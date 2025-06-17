import os
import json
import logging
import requests
import dataclasses
import re
from typing import List, Dict, Optional
from langdetect import detect_langs
from langdetect.lang_detect_exception import LangDetectException

# === Configuration ===
DEBUG = os.environ.get("DEBUG", "false").lower() == "true"
if DEBUG:
    logging.basicConfig(level=logging.DEBUG)

AZURE_SEARCH_PERMITTED_GROUPS_COLUMN = os.environ.get("AZURE_SEARCH_PERMITTED_GROUPS_COLUMN")

# === JSON Encoding ===
class JSONEncoder(json.JSONEncoder):
    """Custom JSON encoder to handle dataclasses."""
    def default(self, obj):
        if dataclasses.is_dataclass(obj):
            return dataclasses.asdict(obj)
        return super().default(obj)

# === Streaming Utilities ===
async def format_as_ndjson(response):
    """
    Format response as NDJSON (newline-delimited JSON) for streaming.

    Args:
        response: Async iterable yielding response data.

    Yields:
        str: JSON string followed by newline.
    """
    try:
        async for event in response:
            yield json.dumps(event, cls=JSONEncoder) + "\n"
    except Exception as error:
        logging.exception("❌ Exception while generating NDJSON stream: %s", error)
        yield json.dumps({"error": str(error)}) + "\n"

# === Azure AD Group Utilities ===
def parse_multi_columns(columns: str) -> List[str]:
    """
    Parse a string of columns into a list, handling '|' or ',' separators.

    Args:
        columns (str): String of column names separated by '|' or ','.

    Returns:
        List[str]: List of column names.
    """
    return columns.split("|") if "|" in columns else columns.split(",")

def fetchUserGroups(userToken: str, nextLink: Optional[str] = None) -> List[Dict]:
    """
    Fetch user groups from Microsoft Graph API.

    Args:
        userToken (str): Azure AD user token.
        nextLink (str, optional): URL for paginated results.

    Returns:
        List[Dict]: List of group objects with 'id' fields.
    """
    endpoint = nextLink or "https://graph.microsoft.com/v1.0/me/transitiveMemberOf?$select=id"
    headers = {"Authorization": f"Bearer {userToken}"}
    try:
        response = requests.get(endpoint, headers=headers)
        if response.status_code != 200:
            logging.error(f"❌ Error fetching user groups: {response.status_code} {response.text}")
            return []
        data = response.json()
        groups = data.get("value", [])
        if "@odata.nextLink" in data:
            groups.extend(fetchUserGroups(userToken, data["@odata.nextLink"]))
        return groups
    except Exception as e:
        logging.error(f"❌ Exception in fetchUserGroups: {e}")
        return []

def generateFilterString(userToken: str) -> str:
    """
    Generate a filter string for Azure Search based on user groups.

    Args:
        userToken (str): Azure AD user token.

    Returns:
        str: Filter string for Azure Search.
    """
    user_groups = fetchUserGroups(userToken)
    if not user_groups:
        logging.debug("✅ No user groups found")
        return ""
    group_ids = ", ".join(group["id"] for group in user_groups)
    return f"{AZURE_SEARCH_PERMITTED_GROUPS_COLUMN}/any(g:search.in(g, '{group_ids}'))"

# === Response Formatting ===
def format_non_streaming_response(chatCompletion, history_metadata: Dict, apim_request_id: str) -> Dict:
    """
    Format a non-streaming chat completion response.

    Args:
        chatCompletion: Chat completion object from Azure OpenAI.
        history_metadata (Dict): Metadata for conversation history.
        apim_request_id (str): APIM request ID.

    Returns:
        Dict: Formatted response with choices and messages.
    """
    response_obj = {
        "id": chatCompletion.id,
        "model": chatCompletion.model,
        "created": chatCompletion.created,
        "object": chatCompletion.object,
        "choices": [{"messages": []}],
        "history_metadata": history_metadata,
        "apim-request-id": apim_request_id,
    }
    if chatCompletion.choices:
        message = chatCompletion.choices[0].message
        if message:
            if hasattr(message, "context"):
                response_obj["choices"][0]["messages"].append({
                    "role": "tool",
                    "content": json.dumps(message.context)
                })
            response_obj["choices"][0]["messages"].append({
                "role": "assistant",
                "content": message.content
            })
    return response_obj

def format_stream_response(chatCompletionChunk, history_metadata: Dict, apim_request_id: str) -> Dict:
    """
    Format a streaming chat completion chunk.

    Args:
        chatCompletionChunk: Chat completion chunk from Azure OpenAI.
        history_metadata (Dict): Metadata for conversation history.
        apim_request_id (str): APIM request ID.

    Returns:
        Dict: Formatted response with choices and messages.
    """
    response_obj = {
        "id": chatCompletionChunk.id,
        "model": chatCompletionChunk.model,
        "created": chatCompletionChunk.created,
        "object": chatCompletionChunk.object,
        "choices": [{"messages": []}],
        "history_metadata": history_metadata,
        "apim-request-id": apim_request_id,
    }
    if chatCompletionChunk.choices:
        delta = chatCompletionChunk.choices[0].delta
        if delta:
            if hasattr(delta, "context"):
                return append_msg(response_obj, {"role": "tool", "content": json.dumps(delta.context)})
            if delta.role == "assistant" and hasattr(delta, "context"):
                return append_msg(response_obj, {"role": "assistant", "context": delta.context})
            if delta.tool_calls:
                return append_msg(response_obj, {
                    "role": "tool",
                    "tool_calls": {
                        "id": delta.tool_calls[0].id,
                        "function": {
                            "name": delta.tool_calls[0].function.name,
                            "arguments": delta.tool_calls[0].function.arguments,
                        },
                        "type": delta.tool_calls[0].type
                    },
                    "context": json.dumps(delta.context) if hasattr(delta, "context") else None
                })
            if delta.content:
                return append_msg(response_obj, {"role": "assistant", "content": delta.content})
    return response_obj

def append_msg(resp_obj: Dict, msg: Dict) -> Dict:
    """
    Append a message to the response object's choices.

    Args:
        resp_obj (Dict): Response object to modify.
        msg (Dict): Message to append.

    Returns:
        Dict: Updated response object.
    """
    resp_obj["choices"][0]["messages"].append(msg)
    return resp_obj

# === PromptFlow Support ===
def format_pf_non_streaming_response(
    chatCompletion: Dict,
    history_metadata: Dict,
    response_field_name: str,
    citations_field_name: str,
    message_uuid: Optional[str] = None
) -> Dict:
    """
    Format a non-streaming PromptFlow response.

    Args:
        chatCompletion (Dict): PromptFlow response data.
        history_metadata (Dict): Metadata for conversation history.
        response_field_name (str): Field name for response content.
        citations_field_name (str): Field name for citations.
        message_uuid (str, optional): Unique message ID.

    Returns:
        Dict: Formatted response with choices and messages.
    """
    if chatCompletion is None:
        return {"error": "No response received from PromptFlow endpoint."}
    if "error" in chatCompletion:
        return {"error": chatCompletion["error"]}
    messages = []
    if response_field_name in chatCompletion:
        messages.append({"role": "assistant", "content": chatCompletion[response_field_name]})
    if citations_field_name in chatCompletion:
        messages.append({"role": "tool", "content": json.dumps({"citations": chatCompletion[citations_field_name]})})
    return {
        "id": chatCompletion.get("id", ""),
        "model": "",
        "created": "",
        "object": "",
        "history_metadata": history_metadata,
        "choices": [{"messages": messages}],
    }

def convert_to_pf_format(input_json: Dict, request_field_name: str, response_field_name: str) -> List[Dict]:
    """
    Convert input JSON to PromptFlow format.

    Args:
        input_json (Dict): Input JSON with messages.
        request_field_name (str): Field name for request content.
        response_field_name (str): Field name for response content.

    Returns:
        List[Dict]: List of PromptFlow-formatted entries.
    """
    output_json = []
    for message in input_json.get("messages", []):
        if message["role"] == "user":
            output_json.append({
                "inputs": {request_field_name: message["content"]},
                "outputs": {response_field_name: ""}
            })
        elif message["role"] == "assistant" and output_json:
            output_json[-1]["outputs"][response_field_name] = message["content"]
    return output_json

# === General Utilities ===
def comma_separated_string_to_list(s: str) -> List[str]:
    """
    Convert a comma-separated string to a list.

    Args:
        s (str): Comma-separated string.

    Returns:
        List[str]: List of stripped values.
    """
    return s.strip().replace(" ", "").split(",")

# === Metadata Extraction ===
COUNTRY_PHONE_CODES = {
    "afghanistan": "+93",
    "albania": "+355",
    "algeria": "+213",
    "andorra": "+376",
    "angola": "+244",
    "argentina": "+54",
    "armenia": "+374",
    "australia": "+61",
    "austria": "+43",
    "azerbaijan": "+994",
    "bahamas": "+1",
    "bahrain": "+973",
    "bangladesh": "+880",
    "belarus": "+375",
    "belgium": "+32",
    "belize": "+501",
    "benin": "+229",
    "bhutan": "+975",
    "bolivia": "+591",
    "bosnia": "+387",
    "botswana": "+267",
    "brazil": "+55",
    "brunei": "+673",
    "bulgaria": "+359",
    "burkina faso": "+226",
    "burundi": "+257",
    "cambodia": "+855",
    "cameroon": "+237",
    "canada": "+1",
    "cape verde": "+238",
    "chad": "+235",
    "chile": "+56",
    "china": "+86",
    "colombia": "+57",
    "comoros": "+269",
    "congo": "+242",
    "costa rica": "+506",
    "croatia": "+385",
    "cuba": "+53",
    "cyprus": "+357",
    "czech republic": "+420",
    "denmark": "+45",
    "djibouti": "+253",
    "dominica": "+1",
    "dominican republic": "+1",
    "ecuador": "+593",
    "egypt": "+20",
    "el salvador": "+503",
    "equatorial guinea": "+240",
    "eritrea": "+291",
    "estonia": "+372",
    "eswatini": "+268",
    "ethiopia": "+251",
    "fiji": "+679",
    "finland": "+358",
    "france": "+33",
    "gabon": "+241",
    "gambia": "+220",
    "to": "+1",
    "georgia": "+995",
    "germany": "+49",
    "ghana": "+233",
    "greece": "+30",
    "greenland": "+299",
    "guatemala": "+502",
    "guinea": "+224",
    "guyana": "+592",
    "haiti": "+509",
    "honduras": "+504",
    "hong kong": "+852",
    "hungary": "+36",
    "iceland": "+354",
    "india": "+91",
    "indonesia": "+62",
    "iran": "+98",
    "iraq": "+964",
    "ireland": "+353",
    "israel": "+972",
    "italy": "+39",
    "jamaica": "+1",
    "japan": "+81",
    "jordan": "+962",
    "kazakhstan": "+7",
    "kenya": "+254",
    "kiribati": "+686",
    "korea": "+82",
    "kuwait": "+965",
    "kyrgyzstan": "+996",
    "laos": "+856",
    "latvia": "+371",
    "lebanon": "+61",
    "lesotho": "+266",
    "liberia": "+231",
    "libya": "+218",
    "liechtenstein": "+423",
    "lithuania": "+370",
    "luxembourg": "+351",
    "madagascar": "261",
    "malawi": "+265",
    "malaysia": "+60",
    "maldives": "+960",
    "mali": "+223",
    "malta": "+356",
    "marshall islands": "+92",
    "mauritania": "22",
    "mauritius": "110",
    "mexico": "+52",
    "micronesia": "+691",
    "moldova": "+373",
    "monaco": "+377",
    "mongolia": "+976",
    "montenegro": "+382",
    "morocco": "+212",
    "mozambique": "+258",
    "myanmar": "+95",
    "namibia": "+264",
    "nepal": "+977",
    "netherlands": "+31",
    "new zealand": "+64",
    "nicaragua": "505",
    "niger": "+227",
    "nigeria": "+234",
    "north macedonia": "+389",
    "norway": "+47",
    "oman": "+968",
    "pakistan": "+92",
    "palau": "+680",
    "panama": "+507",
    "papua new guinea": "+675",
    "paraguay": "+595",
    "peru": "+51",
    "philippines": "+63",
    "poland": "+48",
    "portugal": "+351",
    "qatar": "+974",
    "romania": "+40",
    "russia": "+7",
    "rwanda": "+250",
    "saint lucia": "+1",
    "samoa": "+685",
    "san marino": "+378",
    "saudi arabia": "+966",
    "senegal": "+221",
    "serbia": "+381",
    "seychelles": "+248",
    "singapore": "+65",
    "slovakia": "+386",
    "slovenia": "+49",
    "somalia": "+252",
    "south africa": "+27",
    "south sudan": "+211",
    "spain": "+34",
    "sri lanka": "+94",
    "sudan": "+249",
    "suriname": "+597",
    "sweden": "+46",
    "switzerland": "+41",
    "syria": "+963",
    "taiwan": "+886",
    "tajikistan": "+992",
    "tanzania": "+255",
    "thailand": "+66",
    "timor-leste": "+670",
    "togo": "+228",
    "tonga": "+676",
    "trinidad": "+1",
    "tunisia": "+216",
    "turkey": "+90",
    "turkmenistan": "+993",
    "uganda": "+256",
    "ukraine": "+380",
    "united arab emirates": "+971",
    "united kingdom": "+44",
    "uk": "+44",
    "united states": "+1",
    "usa": "+1",
    "uruguay": "+598",
    "uzbekistan": "+998",
    "vanuatu": "+678",
    "venezuela": "+58",
    "vietnam": "+84",
    "yemen": "+967",
    "zambia": "+260",
    "zimbabwe": "+263",
    "ind": "+91",
    "aus": "+61",
    "us": "+1"
}

def extract_metadata_from_message(text: str) -> Dict[str, Optional[str]]:
    """
    Extract metadata (country, language, phone) from user input text.

    Args:
        text (str): User input text.

    Returns:
        Dict[str, Optional[str]]: Dictionary with country, language, phone, and confidence.
    """
    text_lower = text.lower()
    detected_country = next((k for k in COUNTRY_PHONE_CODES if k in text_lower), None)
    country_code = COUNTRY_PHONE_CODES.get(detected_country)
    phone_match = re.search(r'(\+\d{1,3})?\s?\d{7,15}|\b\d{10}\b', text)
    phone_raw = phone_match.group(0).strip() if phone_match else None
    phone = None
    if phone_raw:
        digits_only = re.sub(r'\D', '', phone_raw)
        phone = f"+{digits_only}" if phone_raw.startswith("+") else (f"{country_code}{digits_only}" if country_code else None)
    try:
        langs = detect_langs(text)
        lang_code = langs[0].lang if langs else "en"  # Default to English if detection fails
        if lang_code == "id" and detected_country == "india":
            lang_code = "en"
        confidence = langs[0].prob if langs else 0.5  # Use detection probability as confidence
    except LangDetectException:
        lang_code = "en"
        confidence = 0.5
    return {
        "country": detected_country.title() if detected_country else None,
        "language": lang_code,
        "phone": phone,
        "confidence": confidence
    }

def needs_form(metadata: Dict[str, Optional[str]]) -> bool:
    """
    Check if metadata is incomplete and requires a form response.

    Args:
        metadata (Dict[str, Optional[str]]): Metadata dictionary with country, language, phone.

    Returns:
        bool: True if language is missing, False otherwise.
    """
    return not metadata.get("language")
