import os
import json
import logging
import requests
import dataclasses
import re
from typing import List
from langdetect import detect_langs
from langdetect.lang_detect_exception import LangDetectException

DEBUG = os.environ.get("DEBUG", "false")
if DEBUG.lower() == "true":
    logging.basicConfig(level=logging.DEBUG)

AZURE_SEARCH_PERMITTED_GROUPS_COLUMN = os.environ.get("AZURE_SEARCH_PERMITTED_GROUPS_COLUMN")


class JSONEncoder(json.JSONEncoder):
    def default(self, o):
        if dataclasses.is_dataclass(o):
            return dataclasses.asdict(o)
        return super().default(o)


async def format_as_ndjson(r):
    try:
        async for event in r:
            yield json.dumps(event, cls=JSONEncoder) + "\n"
    except Exception as error:
        logging.exception("Exception while generating response stream: %s", error)
        yield json.dumps({"error": str(error)})


def parse_multi_columns(columns: str) -> list:
    if "|" in columns:
        return columns.split("|")
    else:
        return columns.split(",")


def fetchUserGroups(userToken, nextLink=None):
    if nextLink:
        endpoint = nextLink
    else:
        endpoint = "https://graph.microsoft.com/v1.0/me/transitiveMemberOf?$select=id"

    headers = {"Authorization": "bearer " + userToken}
    try:
        r = requests.get(endpoint, headers=headers)
        if r.status_code != 200:
            logging.error(f"Error fetching user groups: {r.status_code} {r.text}")
            return []

        r = r.json()
        if "@odata.nextLink" in r:
            nextLinkData = fetchUserGroups(userToken, r["@odata.nextLink"])
            r["value"].extend(nextLinkData)

        return r["value"]
    except Exception as e:
        logging.error(f"Exception in fetchUserGroups: {e}")
        return []


def generateFilterString(userToken):
    userGroups = fetchUserGroups(userToken)
    if not userGroups:
        logging.debug("No user groups found")

    group_ids = ", ".join([obj["id"] for obj in userGroups])
    return f"{AZURE_SEARCH_PERMITTED_GROUPS_COLUMN}/any(g:search.in(g, '{group_ids}'))"


def format_non_streaming_response(chatCompletion, history_metadata, apim_request_id):
    response_obj = {
        "id": chatCompletion.id,
        "model": chatCompletion.model,
        "created": chatCompletion.created,
        "object": chatCompletion.object,
        "choices": [{"messages": []}],
        "history_metadata": history_metadata,
        "apim-request-id": apim_request_id,
    }

    if len(chatCompletion.choices) > 0:
        message = chatCompletion.choices[0].message
        if message:
            if hasattr(message, "context"):
                response_obj["choices"][0]["messages"].append(
                    {"role": "tool", "content": json.dumps(message.context)}
                )
            response_obj["choices"][0]["messages"].append(
                {"role": "assistant", "content": message.content}
            )
            return response_obj

    return {}


def format_stream_response(chatCompletionChunk, history_metadata, apim_request_id):
    response_obj = {
        "id": chatCompletionChunk.id,
        "model": chatCompletionChunk.model,
        "created": chatCompletionChunk.created,
        "object": chatCompletionChunk.object,
        "choices": [{"messages": []}],
        "history_metadata": history_metadata,
        "apim-request-id": apim_request_id,
    }

    if len(chatCompletionChunk.choices) > 0:
        delta = chatCompletionChunk.choices[0].delta
        if delta:
            if hasattr(delta, "context"):
                messageObj = {"role": "tool", "content": json.dumps(delta.context)}
                response_obj["choices"][0]["messages"].append(messageObj)
                return response_obj
            if delta.role == "assistant" and hasattr(delta, "context"):
                messageObj = {"role": "assistant", "context": delta.context}
                response_obj["choices"][0]["messages"].append(messageObj)
                return response_obj
            if delta.tool_calls:
                messageObj = {
                    "role": "tool",
                    "tool_calls": {
                        "id": delta.tool_calls[0].id,
                        "function": {
                            "name": delta.tool_calls[0].function.name,
                            "arguments": delta.tool_calls[0].function.arguments,
                        },
                        "type": delta.tool_calls[0].type,
                    },
                }
                if hasattr(delta, "context"):
                    messageObj["context"] = json.dumps(delta.context)
                response_obj["choices"][0]["messages"].append(messageObj)
                return response_obj
            else:
                if delta.content:
                    messageObj = {"role": "assistant", "content": delta.content}
                    response_obj["choices"][0]["messages"].append(messageObj)
                    return response_obj

    return {}


def format_pf_non_streaming_response(chatCompletion, history_metadata, response_field_name, citations_field_name, message_uuid=None):
    if chatCompletion is None:
        logging.error("chatCompletion object is None - Increase PROMPTFLOW_RESPONSE_TIMEOUT parameter")
        return {"error": "No response received from promptflow endpoint."}
    if "error" in chatCompletion:
        logging.error(f"Error in promptflow response api: {chatCompletion['error']}")
        return {"error": chatCompletion["error"]}

    logging.debug(f"chatCompletion: {chatCompletion}")
    try:
        messages = []
        if response_field_name in chatCompletion:
            messages.append({"role": "assistant", "content": chatCompletion[response_field_name]})
        if citations_field_name in chatCompletion:
            citation_content = {"citations": chatCompletion[citations_field_name]}
            messages.append({"role": "tool", "content": json.dumps(citation_content)})

        response_obj = {
            "id": chatCompletion["id"],
            "model": "",
            "created": "",
            "object": "",
            "history_metadata": history_metadata,
            "choices": [{"messages": messages}],
        }
        return response_obj
    except Exception as e:
        logging.error(f"Exception in format_pf_non_streaming_response: {e}")
        return {}


def convert_to_pf_format(input_json, request_field_name, response_field_name):
    output_json = []
    logging.debug(f"Input json: {input_json}")
    for message in input_json["messages"]:
        if message:
            if message["role"] == "user":
                new_obj = {"inputs": {request_field_name: message["content"]}, "outputs": {response_field_name: ""}}
                output_json.append(new_obj)
            elif message["role"] == "assistant" and len(output_json) > 0:
                output_json[-1]["outputs"][response_field_name] = message["content"]
    logging.debug(f"PF formatted response: {output_json}")
    return output_json


def comma_separated_string_to_list(s: str) -> List[str]:
    return s.strip().replace(' ', '').split(',')


import re
from langdetect import detect_langs
from langdetect.lang_detect_exception import LangDetectException

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
  "lebanon": "+961",
  "lesotho": "+266",
  "liberia": "+231",
  "libya": "+218",
  "liechtenstein": "+423",
  "lithuania": "+370",
  "luxembourg": "+352",
  "madagascar": "+261",
  "malawi": "+265",
  "malaysia": "+60",
  "maldives": "+960",
  "mali": "+223",
  "malta": "+356",
  "marshall islands": "+692",
  "mauritania": "+222",
  "mauritius": "+230",
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
  "nicaragua": "+505",
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
  "slovakia": "+421",
  "slovenia": "+386",
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

def extract_metadata_from_message(text: str):
    """
    Extracts metadata such as country, language, and phone number from a given text input.
    """
    text_lower = text.lower()

    # 🌍 Country detection
    detected_country = next((k for k in COUNTRY_PHONE_CODES if k in text_lower), None)
    country_code = COUNTRY_PHONE_CODES.get(detected_country)

    # 📞 Phone number detection
    phone_match = re.search(r'(\+\d{1,3})?\s?\d{7,15}|\b\d{10}\b', text)
    phone_raw = phone_match.group(0).strip() if phone_match else None

    phone = None
    if phone_raw:
        digits_only = re.sub(r'\D', '', phone_raw)
        if phone_raw.startswith("+"):
            phone = "+" + digits_only
        elif country_code:
            phone = country_code + digits_only

    # 🌐 Language detection
    try:
        langs = detect_langs(text)
        lang_code = langs[0].lang if langs else None
        if lang_code == "id" and detected_country == "india":
            lang_code = "en"
    except LangDetectException:
        lang_code = None

    return {
        "country": detected_country.title() if detected_country else None,
        "language": lang_code,
        "phone": phone
    }

def needs_form(metadata: dict):
    """
    Checks if any of the required metadata fields are missing.
    """
    required = ["country", "language", "phone"]
    return any(metadata.get(k) is None for k in required)
