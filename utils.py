import os
import json
import logging
import requests
import dataclasses
import re
from pathlib import Path
from typing import List
from langdetect import detect_langs
from langdetect.lang_detect_exception import LangDetectException

DEBUG = os.environ.get("DEBUG", "false")
if DEBUG.lower() == "true":
    logging.basicConfig(level=logging.DEBUG)

AZURE_SEARCH_PERMITTED_GROUPS_COLUMN = os.environ.get("AZURE_SEARCH_PERMITTED_GROUPS_COLUMN")

# Load country → phone code mapping
with open(Path(__file__).parent / "country_phone_code_map.json", "r") as f:
    COUNTRY_PHONE_CODES = json.load(f)


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
    return columns.split("|") if "|" in columns else columns.split(",")


def fetchUserGroups(userToken, nextLink=None):
    endpoint = nextLink or "https://graph.microsoft.com/v1.0/me/transitiveMemberOf?$select=id"
    headers = {"Authorization": "bearer " + userToken}
    try:
        r = requests.get(endpoint, headers=headers)
        if r.status_code != 200:
            logging.error(f"Error fetching user groups: {r.status_code} {r.text}")
            return []

        r = r.json()
        if "@odata.nextLink" in r:
            r["value"].extend(fetchUserGroups(userToken, r["@odata.nextLink"]))
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

    if chatCompletion.choices:
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

    if chatCompletionChunk.choices:
        delta = chatCompletionChunk.choices[0].delta
        if delta:
            if hasattr(delta, "context"):
                response_obj["choices"][0]["messages"].append(
                    {"role": "tool", "content": json.dumps(delta.context)}
                )
                return response_obj
            if delta.role == "assistant" and hasattr(delta, "context"):
                response_obj["choices"][0]["messages"].append(
                    {"role": "assistant", "context": delta.context}
                )
                return response_obj
            if delta.tool_calls:
                tc = delta.tool_calls[0]
                response_obj["choices"][0]["messages"].append(
                    {
                        "role": "tool",
                        "tool_calls": {
                            "id": tc.id,
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments
                            },
                            "type": tc.type
                        },
                        "context": json.dumps(delta.context) if hasattr(delta, "context") else None
                    }
                )
                return response_obj
            if delta.content:
                response_obj["choices"][0]["messages"].append(
                    {"role": "assistant", "content": delta.content}
                )
                return response_obj

    return {}


def format_pf_non_streaming_response(chatCompletion, history_metadata, response_field_name, citations_field_name, message_uuid=None):
    if chatCompletion is None:
        logging.error("chatCompletion object is None - Increase PROMPTFLOW_RESPONSE_TIMEOUT parameter")
        return {
            "error": "No response received from promptflow endpoint"
        }

    if "error" in chatCompletion:
        logging.error(f"Error in promptflow response api: {chatCompletion['error']}")
        return {"error": chatCompletion["error"]}

    messages = []
    if response_field_name in chatCompletion:
        messages.append({"role": "assistant", "content": chatCompletion[response_field_name]})
    if citations_field_name in chatCompletion:
        messages.append({"role": "tool", "content": json.dumps({"citations": chatCompletion[citations_field_name]})})

    return {
        "id": chatCompletion["id"],
        "model": "",
        "created": "",
        "object": "",
        "history_metadata": history_metadata,
        "choices": [{"messages": messages}]
    }


def convert_to_pf_format(input_json, request_field_name, response_field_name):
    output_json = []
    for message in input_json["messages"]:
        if message:
            if message["role"] == "user":
                output_json.append({
                    "inputs": {request_field_name: message["content"]},
                    "outputs": {response_field_name: ""}
                })
            elif message["role"] == "assistant" and output_json:
                output_json[-1]["outputs"][response_field_name] = message["content"]
    return output_json


def comma_separated_string_to_list(s: str) -> List[str]:
    return s.strip().replace(' ', '').split(',')


def extract_metadata_from_message(text: str):
    """
    Extracts country, language, and phone number from the input text.
    Supports auto-detecting countries + abbreviations and formats phone with correct prefix.
    """
    text_lower = text.lower()

    # 🌍 Country detection
    detected_country = next((k for k in COUNTRY_PHONE_CODES if k in text_lower), None)
    country_code = COUNTRY_PHONE_CODES.get(detected_country)

    # 📞 Phone detection
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
    Returns True if any required metadata field (country, language, phone) is missing.
    """
    required = ["country", "language", "phone"]
    return any(metadata.get(k) is None for k in required)
