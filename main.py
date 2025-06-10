from fastapi import FastAPI, Request, Form, BackgroundTasks, HTTPException, Response
from fastapi.responses import PlainTextResponse, StreamingResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List, Optional
import os
import httpx
import re
import json
import uuid
import time
import asyncio
from dotenv import load_dotenv
from twilio.rest import Client
from azure.cosmos import CosmosClient, PartitionKey
from azure_search import search_articles
from utils import extract_metadata_from_message, needs_form
import xml.sax.saxutils as saxutils
from elevenlabs.client import ElevenLabs
from elevenlabs import VoiceSettings
from azure.storage.blob import BlobServiceClient, ContentSettings
from cosmos_client import update_preferences

# Load environment variables
load_dotenv(dotenv_path=".env.production")

AZURE_OPENAI_ENDPOINT = os.environ.get("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_KEY = os.environ.get("AZURE_OPENAI_KEY")
AZURE_OPENAI_MODEL = os.environ.get("AZURE_OPENAI_MODEL")
AZURE_OPENAI_API_VERSION = os.environ.get("AZURE_OPENAI_PREVIEW_API_VERSION", "2024-05-01-preview")
AZURE_WHISPER_MODEL = os.environ.get("AZURE_WHISPER_MODEL", "whisper-1")

TWILIO_ACCOUNT_SID = os.environ.get("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = os.environ.get("TWILIO_AUTH_TOKEN")
TWILIO_PHONE_NUMBER = os.environ.get("TWILIO_PHONE_NUMBER", "whatsapp:+447488880990")
TWILIO_VOICE_NUMBER = os.environ.get("TWILIO_VOICE_NUMBER")

ELEVENLABS_API_KEY = os.environ.get("ELEVENLABS_API_KEY")
ELEVENLABS_VOICE_ID = os.environ.get("ELEVENLABS_VOICE_ID", "21m00Tcm4TlvDq8ikWAM")

AZURE_STORAGE_CONNECTION_STRING = os.environ.get("AZURE_STORAGE_CONNECTION_STRING")

required_vars = [
    "AZURE_OPENAI_ENDPOINT", "AZURE_OPENAI_KEY", "AZURE_OPENAI_MODEL",
    "TWILIO_ACCOUNT_SID", "TWILIO_AUTH_TOKEN",
    "AZURE_STORAGE_CONNECTION_STRING",
    "COSMOS_DB_ENDPOINT", "COSMOS_DB_KEY"
]
for var in required_vars:
    if not os.environ.get(var):
        raise EnvironmentError(f"Missing required environment variable: {var}")

app = FastAPI()

conversation_history = {}

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://kind-island-057bb3903.6.azurestaticapps.net"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")

class Message(BaseModel):
    role: str
    content: str

class ConversationRequest(BaseModel):
    messages: Optional[List[Message]] = []
    session_id: Optional[str] = None

class FeedbackRequest(BaseModel):
    user_id: str
    fact_id: str
    liked: bool

@app.get("/")
async def root():
    return {"message": "Backend online!"}

async def transcribe_audio(audio_url: str) -> str:
    """
    Transcribe audio from the given URL using Azure OpenAI Whisper.
    """
    print(f"🎙️ Starting transcription for audio URL: {audio_url}")
    try:
        async with httpx.AsyncClient(timeout=30, follow_redirects=True) as client:
            print(f"📥 Downloading audio from {audio_url}")
            response = await client.get(audio_url, auth=(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN))
            if response.is_redirect:
                print(f"🔄 Redirected to: {response.headers.get('location')}")
            response.raise_for_status()
            audio_content = response.content
            print(f"✅ Audio downloaded successfully, size: {len(audio_content)} bytes")

            headers = {
                "Content-Type": "multipart/form-data",
                "api-key": AZURE_OPENAI_KEY
            }
            files = {
                "file": ("audio.mp3", audio_content, "audio/mpeg")
            }
            data = {
                "model": AZURE_WHISPER_MODEL
            }
            print(f"📤 Sending audio to Whisper model: {AZURE_WHISPER_MODEL}")
            response = await client.post(
                f"{AZURE_OPENAI_ENDPOINT}/openai/deployments/{AZURE_WHISPER_MODEL}/audio/transcriptions?api-version={AZURE_OPENAI_API_VERSION}",
                headers={"api-key": AZURE_OPENAI_KEY},
                files=files,
                data=data
            )
            response.raise_for_status()
            result = response.json()
            print(f"🎙️ Full transcription response: {result}")
            return result.get("text", "")
    except Exception as e:
        print(f"❌ Error transcribing audio: {str(e)}")
        return ""

async def detect_implicit_liking(session_id: str, conversation_history: List[dict]) -> dict:
    """
    Detect implicit user liking based on conversation patterns using Azure OpenAI.

    Args:
        session_id (str): The session ID or user ID (e.g., WhatsApp number).
        conversation_history (List[dict]): The conversation history for the session.

    Returns:
        dict: {
            "is_liked": bool,
            "fact_id": str or None,
            "confidence": float,
            "topic": str or None,
            "suggested_question": str or None
        }
    """
    print(f"🧠 Detecting implicit liking for session: {session_id}")
    try:
        if not conversation_history or len([m for m in conversation_history if m["role"] == "user"]) < 2:
            return {
                "is_liked": False,
                "fact_id": None,
                "confidence": 0.0,
                "topic": None,
                "suggested_question": None
            }

        # Extract user messages and their timestamps (if available)
        user_messages = [m["content"] for m in conversation_history if m["role"] == "user"]
        recent_messages = user_messages[-5:]  # Analyze last 5 user messages for context

        # Prepare prompt for OpenAI
        openai_body = {
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are an implicit liking detection assistant. Analyze the conversation history to determine if the user implicitly likes a topic. "
                        "Consider the following signals: "
                        "- Deeper or follow-up questions on the same topic (e.g., asking for more details or clarifications). "
                        "- Repeated questions about the same topic across messages. "
                        "- Positive sentiment (e.g., 'this is interesting', 'cool'). "
                        "- High message frequency (multiple messages in a short time). "
                        "- Increasing question complexity (e.g., moving from general to specific). "
                        "Also, identify the main topic and generate a concise fact_id (e.g., 'fact_emf_sustainability'). "
                        "Suggest a relevant follow-up question to keep the user engaged. "
                        "Respond in JSON format: { \"is_liked\": boolean, \"fact_id\": string or null, \"confidence\": float (0.0-1.0), \"topic\": string or null, \"suggested_question\": string or null }. "
                        "Examples: "
                        "- ['What is EMF?', 'How does EMF affect health?'] -> { \"is_liked\": true, \"fact_id\": \"fact_emf_health\", \"confidence\": 0.8, \"topic\": \"EMF health\", \"suggested_question\": \"Would you like to know about EMF safety guidelines?\" } "
                        "- ['Hi', 'Nice!'] -> { \"is_liked\": true, \"fact_id\": null, \"confidence\": 0.6, \"topic\": null, \"suggested_question\": null } "
                        "- ['What is Fique?'] -> { \"is_liked\": false, \"fact_id\": null, \"confidence\": 0.0, \"topic\": null, \"suggested_question\": null }"
                    )
                },
                {"role": "user", "content": "\n".join(recent_messages)}
            ],
            "temperature": 0.3,
            "max_tokens": 150,
            "top_p": 1,
            "frequency_penalty": 0,
            "presence_penalty": 0,
            "stream": False,
        }
        headers = {"Content-Type": "application/json", "api-key": AZURE_OPENAI_KEY}
        async with httpx.AsyncClient(timeout=15) as client:
            response = await client.post(
                f"{AZURE_OPENAI_ENDPOINT}/openai/deployments/{AZURE_OPENAI_MODEL}/chat/completions?api-version={AZURE_OPENAI_API_VERSION}",
                headers=headers,
                json=openai_body
            )
            response.raise_for_status()
            result = response.json()
            reply = result["choices"][0]["message"]["content"]
            liking_data = json.loads(reply)
            print(f"✅ Implicit liking detection response: {liking_data}")
            return liking_data
    except Exception as e:
        print(f"❌ Error in detect_implicit_liking: {str(e)}")
        return {
            "is_liked": False,
            "fact_id": None,
            "confidence": 0.0,
            "topic": None,
            "suggested_question": None
        }

async def conversation_logic(messages, metadata):
    try:
        session_id = metadata.get("phone", str(uuid.uuid4()))  # Use phone as session_id for WhatsApp
        cleaned_messages = [{"role": m["role"], "content": m["content"]} for m in messages]
        user_question = cleaned_messages[-1]["content"]
        print(f"🧠 Calling OpenAI with messages: {cleaned_messages}")

        # Detect implicit liking
        liking_data = await detect_implicit_liking(session_id, messages)
        print(f"🔔 Implicit liking result: {liking_data}")

        # Update preferences if liking detected
        if liking_data["is_liked"] and liking_data["fact_id"]:
            try:
                update_preferences(
                    user_id=session_id,
                    fact_id=liking_data["fact_id"],
                    liked=True,
                    confidence=liking_data["confidence"]
                )
            except Exception as e:
                print(f"❌ Failed to update preferences: {str(e)}")

        # Extract keywords for search
        def extract_keywords(text):
            words = re.findall(r'\w+', text.lower())
            stopwords = {"the", "of", "and", "in", "on", "is", "to", "a"}
            keywords = [w for w in words if w not in stopwords]
            return " ".join(keywords[:10])

        keywords = extract_keywords(user_question)
        search_contexts = search_articles(keywords) or []
        print(f"🔍 Search query: {keywords}, Articles returned: {len(search_contexts)}")

        fallback_phrases = [
            "yes", "yeah", "sure", "go ahead", "please do", "try general", "fallback", "try again",
            "use gpt", "search online", "search web", "do it", "okay", "alright", "continue",
            "that’s fine", "proceed", "give me an answer", "show me anyway", "tell genarally"
        ]
        fallback_flag = any(
            any(phrase in m["content"].lower() for phrase in fallback_phrases)
            for m in messages[-2:]
        )

        if not search_contexts and not fallback_flag:
            response_content = "🤖 I couldn’t find anything in our articles. Would you like to try a general answer instead?"
            if liking_data["suggested_question"]:
                response_content += f"\n\n💡 How about: {liking_data['suggested_question']}"
            return [{"role": "assistant", "content": response_content}]

        if alleviated_flag or search_contexts:
            if search_contexts:
                context_block = "\n\n".join([
                    f"{item['snippet']}\n\nSource: {item['title']} ({item['url']})"
                    for item in search_contexts if isinstance(item, dict) and all(k in item for k in ["snippet", "title", "url"])
                ]) or "\n\n".join([str(item) for item in search_contexts])
                cleaned_messages[-1] = {
                    "role": "user",
                    "content": f"""Use the following context to answer the question. Cite the article title and URL explicitly.\n\nContext:\n{context_block}\n\nQuestion:\n{user_question}"""
                }
            else:
                cleaned_messages[-1] = {
                    "role": "user",
                    "content": f"Answer the following question in a friendly, conversational tone: {user_question}"
                }

        headers = {"Content-Type": "application/json", "api-key": AZURE_OPENAI_KEY}
        body = {
            "messages": cleaned_messages,
            "temperature": 0.7,
            "top_p": 0.95,
            "frequency_penalty": 0,
            "presence_penalty": 0,
            "max_tokens": 500,
            "stream": False,
        }

        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.post(
                f"{AZURE_OPENAI_ENDPOINT}/openai/deployments/{AZURE_OPENAI_MODEL}/chat/completions?api-version={AZURE_OPENAI_API_VERSION}",
                headers=headers,
                json=body
            )
            response.raise_for_status()

        result = response.json()
        print(f"✅ OpenAI response: {result}")

        if not result or not result.get("choices") or not result["choices"][0].get("message"):
            print("⚠️ OpenAI response is empty or malformed")
            response_content = "⚠️ I couldn’t generate a response."
            if liking_data["suggested_question"]:
                response_content += f"\n\n💡 How about: {liking_data['suggested_question']}"
            return [{"role": "assistant", "content": response_content}]

        # Append suggested question to the response
        response_content = result["choices"][0]["message"]["content"]
        if liking_data["suggested_question"]:
            response_content += f"\n\n💡 Next question: {liking_data['suggested_question']}"

        return [{"role": "assistant", "content": response_content}]

    except Exception as e:
        print(f"❌ Error in conversation_logic: {str(e)}")
        import traceback
        traceback.print_exc()
        response_content = "⚠️ Sorry, there was an error processing your message."
        if liking_data.get("suggested_question"):
            response_content += f"\n\n💡 How about: {liking_data['suggested_question']}"
        return [{"role": "assistant", "content": response_content}]

async def detect_feedback(user_input: str) -> dict:
    """
    Use Azure OpenAI to detect feedback in a WhatsApp message and extract fact_id and liked status.
    """
    print(f"🧠 Detecting feedback in message: {user_input}")
    try:
        openai_body = {
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a feedback detection assistant. Analyze the user message to determine if it contains feedback "
                        "about a fact (e.g., liking or disliking a fact with an ID like 'fact001' or referring to the previous response with 'I like this'). "
                        "Respond in JSON format: { \"is_feedback\": boolean, \"fact_id\": string or null, \"liked\": boolean or null }. "
                        "If the message is 'I like this', 'I love this', 'this is great', or similar positive feedback, set is_feedback to true, fact_id to null, and liked to true. "
                        "If the message is 'I dislike this', 'I hate this', 'this is bad', 'I dont like this', or similar negative feedback, set is_feedback to true, fact_id to null, and liked to false. "
                        "For explicit feedback (e.g., 'I liked fact001'), extract the fact_id. "
                        "If no feedback is detected, return { \"is_feedback\": false, \"fact_id\": null, \"liked\": null }. "
                        "Examples: "
                        "- 'I liked fact001' -> { \"is_feedback\": true, \"fact_id\": \"fact001\", \"liked\": true } "
                        "- 'I like this' -> { \"is_feedback\": true, \"fact_id\": null, \"liked\": true } "
                        "- 'I dont like this' -> { \"is_feedback\": true, \"fact_id\": null, \"liked\": false } "
                        "- 'fact002 was bad' -> { \"is_feedback\": true, \"fact_id\": \"fact002\", \"liked\": false } "
                        "- 'hello' -> { \"is_feedback\": false, \"fact_id\": null, \"liked\": null }"
                    )
                },
                {"role": "user", "content": user_input}
            ],
            "temperature": 0,
            "max_tokens": 100,
            "top_p": 1,
            "frequency_penalty": 0,
            "presence_penalty": 0,
            "stream": False,
        }
        headers = {"Content-Type": "application/json", "api-key": AZURE_OPENAI_KEY}
        async with httpx.AsyncClient(timeout=15) as client:
            response = await client.post(
                f"{AZURE_OPENAI_ENDPOINT}/openai/deployments/{AZURE_OPENAI_MODEL}/chat/completions?api-version={AZURE_OPENAI_API_VERSION}",
                headers=headers,
                json=openai_body
            )
            response.raise_for_status()
            result = response.json()
            reply = result["choices"][0]["message"]["content"]
            feedback_data = json.loads(reply)
            print(f"✅ Feedback detection response: {feedback_data}")
            return feedback_data
    except Exception as e:
        print(f"❌ Error in detect_feedback: {str(e)}")
        return {"is_feedback": False, "fact_id": None, "liked": None}

async def extract_fact_id(previous_message: str) -> str:
    """
    Use Azure OpenAI to extract or generate a fact_id from the previous assistant message.
    """
    print(f"🧠 Extracting fact_id from previous message: {previous_message}")
    try:
        openai_body = {
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a fact ID extraction assistant. Analyze the provided message to extract an explicit fact ID (e.g., 'fact001') if present. "
                        "If no explicit fact ID is found, generate a concise fact ID based on the main topic or subject of the message (e.g., 'fact_emf_sustainability' for a message about EMF sustainability, 'fact_donald_trump' for a message about Donald Trump). "
                        "Use lowercase and underscores for generated IDs. Respond in JSON format: { \"fact_id\": string or null }. "
                        "If no fact ID can be determined or generated, return { \"fact_id\": null }. "
                        "Examples: "
                        "- 'Here is fact001: ...' -> { \"fact_id\": \"fact001\" } "
                        "- 'EMF sustainability refers to... Source: ...' -> { \"fact_id\": \"fact_emf_sustainability\" } "
                        "- 'Donald Trump is a former President...' -> { \"fact_id\": \"fact_donald_trump\" } "
                        "- 'This is a generic response.' -> { \"fact_id\": null }"
                    )
                },
                {"role": "user", "content": previous_message}
            ],
            "temperature": 0,
            "max_tokens": 50,
            "top_p": 1,
            "frequency_penalty": 0,
            "presence_penalty": 0,
            "stream": False,
        }
        headers = {"Content-Type": "application/json", "api-key": AZURE_OPENAI_KEY}
        async with httpx.AsyncClient(timeout=15) as client:
            response = await client.post(
                f"{AZURE_OPENAI_ENDPOINT}/openai/deployments/{AZURE_OPENAI_MODEL}/chat/completions?api-version={AZURE_OPENAI_API_VERSION}",
                headers=headers,
                json=openai_body
            )
            response.raise_for_status()
            result = response.json()
            reply = result["choices"][0]["message"]["content"]
            fact_data = json.loads(reply)
            fact_id = fact_data.get("fact_id")
            print(f"✅ Fact ID extraction response: {fact_data}")
            return fact_id
    except Exception as e:
        print(f"❌ Error in extract_fact_id: {str(e)}")
        return None

async def process_feedback(user_id: str, fact_id: str, liked: bool, feedback_response: str) -> tuple[bool, str]:
    """
    Process feedback by calling update_preferences and updating feedback_response.
    """
    try:
        update_preferences(user_id, fact_id, liked, confidence=1.0)  # Explicit feedback has confidence 1.0
        feedback_response = f"✅ Recorded your {'like' if liked else 'dislike'} for fact {fact_id}."
        print(f"✅ Feedback processed for {user_id}: fact_id={fact_id}, liked={liked}")
        return True, feedback_response
    except Exception as e:
        print(f"❌ Error processing feedback: {str(e)}")
        feedback_response = "⚠️ Sorry, I couldn’t save your feedback."
        return False, feedback_response

@app.post("/conversation")
async def conversation_endpoint(request: Request):
    try:
        payload = await request.json()
        messages_data = payload.get("messages", [])
        session_id = payload.get("session_id", str(uuid.uuid4()))
        print(f"📩 Received payload: {payload}, Session ID: {session_id}")

        if not messages_data:
            return StreamingResponse(
                stream_response(
                    [{"role": "assistant", "content": "👋 Welcome! Before we begin, could you please tell me your *country* and *phone number*?"}],
                    session_id
                ),
                media_type="text/event-stream"
            )

        valid_messages = [
            Message(role=msg["role"], content=msg["content"])
            for msg in messages_data
            if isinstance(msg, dict) and msg.get("role") and msg.get("content")
        ]

        if not valid_messages:
            return StreamingResponse(
                stream_response(
                    [{"role": "assistant", "content": "⚠️ Invalid message format."}],
                    session_id
                ),
                media_type="text/event-stream"
            )

        if session_id not in conversation_history:
            conversation_history[session_id] = []

        latest_user_message = {"role": valid_messages[-1].role, "content": valid_messages[-1].content, "timestamp": time.time()}
        conversation_history[session_id].append(latest_user_message)
        print(f"📜 Conversation history for session {session_id}: {conversation_history[session_id]}")

        all_user_text = " ".join([m.content for m in valid_messages if m.role == "user"])
        metadata = {
            "phone": payload.get("phone"),
            "country": payload.get("country"),
            "language": payload.get("language")
        }

        if not metadata["phone"] or not metadata["country"]:
            async with httpx.AsyncClient(timeout=10) as client:
                meta_response = await client.post(
                    f"{request.base_url}extract_metadata",
                    json={"text": all_user_text}
                )
            if meta_response.status_code == 200:
                metadata = meta_response.json()
                print(f"📋 Extracted metadata: {metadata}")

        if len(conversation_history[session_id]) == 1 and needs_form(metadata):
            missing_fields = [k for k, v in metadata.items() if v is None]
            return StreamingResponse(
                stream_response(
                    [{"role": "assistant", "content": f"⚠️ I need more info to help you: missing {', '.join(missing_fields)}. Could you please provide it?"}],
                    session_id
                ),
                media_type="text/event-stream"
            )

        if len(conversation_history[session_id]) == 2 and (metadata["phone"] and metadata["country"]):
            response_message = {
                "role": "assistant",
                "content": (
                    f"""✅ Got it! Here's what I understood:\n\n"""
                    f"- Country: {metadata['country']}\n"
                    f"- Language: {metadata['language']}\n"
                    f"- Phone: {metadata['phone']}\n\n"
                    f"📘 Now, what would you like to ask about Fique?\n\n"
                    f"💡 Suggested: What is Fique used for?"
                )
            }
            conversation_history[session_id].append(response_message)
            return StreamingResponse(
                stream_response([response_message], session_id),
                media_type="text/event-stream"
            )

        result = await conversation_logic(conversation_history[session_id], metadata)
        conversation_history[session_id].append({"role": "assistant", "content": result[0]["content"], "timestamp": time.time()})
        return StreamingResponse(
            stream_response(result, session_id),
            media_type="text/event-stream"
        )

    except Exception as e:
        print(f"❌ Error during conversation: {str(e)}")
        import traceback
        traceback.print_exc()
        return StreamingResponse(
            stream_response(
                [{"role": "assistant", "content": "⚠️ Sorry, there was an error processing your message."}],
                session_id
            ),
            media_type="text/event-stream"
        )

async def cleanup_audio(audio_path, delay=300):
    time.sleep(delay)
    if os.path.exists(audio_path):
        os.remove(audio_path)
        print(f"🗑️ Deleted audio file: {audio_path}")

@app.get("/audio/{filename}")
async def serve_audio(filename: str):
    start_time = time.time()
    audio_path = os.path.join("static/audio", filename)
    if not os.path.exists(audio_path):
        print(f"❌ Audio file not found: {audio_path}")
        raise HTTPException(status_code=404, detail="Audio file not found")
    print(f"🎵 Serving audio file: {audio_path}")
    response = FileResponse(audio_path, media_type="audio/mpeg")
    end_time = time.time()
    print(f"⏱️ Audio serving took {end_time - start_time:.2f} seconds")
    return response

async def upload_to_public_hosting(audio_path: str, audio_filename: str) -> str:
    """
    Uploads the audio file to Azure Blob Storage and returns a public URL.
    """
    start_time = time.time()
    try:
        blob_service_client = BlobServiceClient.from_connection_string(AZURE_STORAGE_CONNECTION_STRING)
        container_name = "audio-files"
        blob_name = audio_filename

        blob_client = blob_service_client.get_blob_client(container=container_name, blob=blob_name)

        content_settings = ContentSettings(content_type="audio/mpeg")

        with open(audio_path, "rb") as f:
            await asyncio.to_thread(blob_client.upload_blob, f, overwrite=True, content_settings=content_settings)

        public_url = f"https://{blob_service_client.account_name}.blob.core.windows.net/{container_name}/{blob_name}"

        async with httpx.AsyncClient(timeout=5) as client:
            response = await client.head(public_url)
            if response.status_code != 200:
                print(f"❌ Blob URL is not accessible: {response.status_code}")
                raise Exception(f"Uploaded audio is not publicly accessible: {response.status_code}")

    except Exception as e:
        print(f"❌ Azure Blob Storage upload error: {str(e)}")
        raise
    finally:
        blob_service_client.close()

    end_time = time.time()
    print(f"📤 Uploaded {audio_path} to Azure Blob Storage in {end_time - start_time:.2f} seconds")
    print(f"✅ Public URL: {public_url}")
    return public_url

@app.post("/twilio-webhook")
async def handle_whatsapp(request: Request, background_tasks: BackgroundTasks, From: str = Form(...), Body: str = Form(...)):
    try:
        form = await request.form()
        user_input = Body.strip().lower()
        media_url = form.get("MediaUrl0")
        media_content_type = form.get("MediaContentType0")

        print(f"📩 Received WhatsApp message from {From}: Body={Body}, Media={media_url}")

        # Initialize feedback variables
        feedback_processed = False
        feedback_response = ""

        # Detect explicit feedback
        feedback_data = await detect_feedback(user_input)
        print(f"🔔 Feedback detection result: {feedback_data}")
        if feedback_data.get("is_feedback", False):
            fact_id = feedback_data.get("fact_id")
            liked = feedback_data.get("liked")
            if fact_id is None and liked is not None:
                print(f"🔔 Contextual feedback detected: liked={liked}")
                if From in conversation_history and conversation_history[From]:
                    last_assistant_message = next(
                        (msg["content"] for msg in reversed(conversation_history[From]) if msg["role"] == "assistant"),
                        None
                    )
                    if last_assistant_message:
                        fact_id = await extract_fact_id(last_assistant_message)
                        if fact_id:
                            feedback_processed, feedback_response = await process_feedback(From, fact_id, liked, feedback_response)
                        else:
                            feedback_response = "⚠️ Couldn’t identify the fact you’re referring to. Please specify, e.g., 'I liked fact001'."
                    else:
                        feedback_response = "⚠️ No previous response to refer to. Please specify the fact ID."
                else:
                    feedback_response = "⚠️ No conversation history found. Please specify the fact ID."
            elif fact_id is not None and liked is not None:
                print(f"🔔 Explicit feedback detected: fact_id={fact_id}, liked={liked}")
                feedback_processed, feedback_response = await process_feedback(From, fact_id, liked, feedback_response)

        voice_phrases = ["send voice", "reply in audio", "voice answer", "audio response"]
        is_voice_request = any(phrase in user_input for phrase in voice_phrases)
        is_voice_message = media_url and media_content_type and "audio" in media_content_type.lower()

        if is_voice_message and media_url:
            print(f"🎙️ Initiating transcription for voice message")
            transcribed_text = await transcribe_audio(media_url)
            user_input = transcribed_text.strip().lower() if transcribed_text else "[Voice message]"
            print(f"🎙️ Transcribed voice message: {user_input}")

            # Check for feedback in transcribed text
            feedback_data = await detect_feedback(user_input)
            if feedback_data.get("is_feedback", False):
                fact_id = feedback_data.get("fact_id")
                liked = feedback_data.get("liked")
                if fact_id is None and liked is not None:
                    if From in conversation_history and conversation_history[From]:
                        last_assistant_message = next(
                            (msg["content"] for msg in reversed(conversation_history[From]) if msg["role"] == "assistant"),
                            None
                        )
                        if last_assistant_message:
                            fact_id = await extract_fact_id(last_assistant_message)
                            if fact_id:
                                feedback_processed, feedback_response = await process_feedback(From, fact_id, liked, feedback_response)
                            else:
                                feedback_response = "⚠️ Couldn’t identify the fact you’re referring to."
                        else:
                            feedback_response = "⚠️ No previous response to refer to."
                    else:
                        feedback_response = "⚠️ No conversation history found."
                elif fact_id is not None and liked is not None:
                    feedback_processed, feedback_response = await process_feedback(From, fact_id, liked, feedback_response)

        if user_input == "reset":
            if From in conversation_history:
                del conversation_history[From]
                print(f"🔄 Reset conversation history for {From}")
            text_reply = "✅ Conversation reset. Let’s start fresh!"
        else:
            if From not in conversation_history:
                conversation_history[From] = []

            conversation_history[From].append({"role": "user", "content": user_input, "timestamp": time.time()})
            print(f"📜 Full conversation history for {From}: {conversation_history[From]}")

            metadata = {"phone": From, "country": "auto", "language": "en"}
            try:
                if not feedback_processed:
                    result = await conversation_logic(conversation_history[From], metadata)
                    text_reply = result[0]["content"]
                else:
                    text_reply = feedback_response
                if feedback_processed and user_input not in ["i like this", "i dislike this", "i dont like this"]:
                    result = await conversation_logic(conversation_history[From], metadata)
                    text_reply = f"{feedback_response}\n\n{result[0]['content']}"
                conversation_history[From].append({"role": "assistant", "content": text_reply, "timestamp": time.time()})
            except Exception as e:
                print(f"❌ Error in twilio-webhook conversation_logic: {str(e)}")
                text_reply = "⚠️ Sorry, something went wrong."
                if feedback_processed:
                    text_reply = f"{feedback_response}\n\n{text_reply}"

        if not TWILIO_ACCOUNT_SID or not TWILIO_AUTH_TOKEN:
            print(f"❌ Twilio credentials missing")
            return PlainTextResponse("Twilio credentials missing", status_code=500)

        client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
        if (is_voice_request or is_voice_message) and ELEVENLABS_API_KEY and ELEVENLABS_VOICE_ID:
            text_reply = text_reply[:150].strip()
            try:
                elevenlabs_client = ElevenLabs(api_key=ELEVENLABS_API_KEY)
                audio_dir = "static/audio"
                os.makedirs(audio_dir, exist_ok=True)
                audio_filename = f"{uuid.uuid4()}.mp3"
                audio_path = os.path.join(audio_dir, audio_filename)

                audio_bytes = elevenlabs_client.text_to_speech.convert(
                    voice_id=ELEVENLABS_VOICE_ID,
                    text=text_reply,
                    voice_settings=VoiceSettings(
                        stability=0.5,
                        similarity_boost=0.75
                    ),
                    output_format="mp3_44100_128"
                )

                with open(audio_path, "wb") as f:
                    for chunk in audio_bytes:
                        if chunk:
                            f.write(chunk)

                file_size_mb = os.path.getsize(audio_path) / (1024 * 1024)
                print(f"📏 Audio file size: {file_size_mb:.2f} MB")
                if file_size_mb > 16:
                    raise Exception("Audio file exceeds WhatsApp 16MB limit")

                audio_url = await upload_to_public_hosting(audio_path, audio_filename)
                message = client.messages.create(
                    media_url=[audio_url],
                    from_=TWILIO_PHONE_NUMBER,
                    to=From
                )
                updated_message = client.messages(message.sid).fetch()
                print(f"📤 Audio message status: {updated_message.status}, SID: {message.sid}")
                if updated_message.status in ["failed", "undelivered"]:
                    message = client.messages.create(
                        body=text_reply,
                        from_=TWILIO_PHONE_NUMBER,
                        to=From
                    )
                    print(f"✅ Fallback text message sent to {From}, SID: {message.sid}")
                else:
                    background_tasks.add_task(cleanup_audio, audio_path)
                    print(f"✅ Audio message sent to {From}, SID: {message.sid}")
            except Exception as e:
                print(f"❌ Error generating audio: {str(e)}")
                message = client.messages.create(
                    body=text_reply,
                    from_=TWILIO_PHONE_NUMBER,
                    to=From
                )
                print(f"✅ Fallback text message sent to {From}, SID: {message.sid}")
        else:
            message = client.messages.create(
                body=text_reply,
                from_=TWILIO_PHONE_NUMBER,
                to=From
            )
            print(f"✅ Text message sent to {From}, SID: {message.sid}")

        return PlainTextResponse("")

    except Exception as e:
        print(f"❌ Critical error in twilio-webhook: {str(e)}")
        import traceback
        traceback.print_exc()
        return PlainTextResponse("", status_code=500)

@app.post("/feedback")
async def process_feedback_endpoint(feedback: FeedbackRequest):
    try:
        if not (feedback.user_id.startswith("whatsapp:") or feedback.user_id.startswith("uuid-")):
            raise HTTPException(status_code=400, detail="Invalid user_id format")
        update_preferences(feedback.user_id, feedback.fact_id, feedback.liked, confidence=1.0)
        return {"status": "success"}
    except Exception as e:
        print(f"❌ Error in process_feedback_endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/extract_metadata")
async def extract_metadata_via_openai(request: Request):
    try:
        data = await request.json()
        user_input = data.get("text")
        if not user_input:
            return {"error": "No text provided."}
        openai_data = {
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a metadata extraction assistant. Extract phone number, country, and language from the text below. "
                        "Respond strictly in JSON format: { \"phone\": \"\", \"country\": \"\", \"language\": \"\", \"confidence\": 0.95 }"
                    )
                },
                {"role": "user", "content": user_input}
            ],
            "temperature": 0,
            "max_tokens": 100,
            "top_p": 1,
            "frequency_penalty": 0,
            "presence_penalty": 0,
            "stream": False,
        }
        headers = {
            "Content-Type": "application/json",
            "api-key": AZURE_OPENAI_KEY
        }
        async with httpx.AsyncClient(timeout=15) as client:
            response = await client.post(
                f"{AZURE_OPENAI_ENDPOINT}/openai/deployments/{AZURE_OPENAI_MODEL}/chat/completions?api-version={AZURE_OPENAI_API_VERSION}",
                headers=headers,
                json=openai_data
            )
            response.raise_for_status()
            result = response.json()
            reply = result["choices"][0]["message"]["content"]
            return json.loads(reply)
    except Exception as e:
        print(f"❌ Error in extract_metadata_via_openai: {str(e)}")
        return {
            "error": "Failed to parse OpenAI response",
            "details": str(e)
        }

@app.post("/voice", response_class=Response)
async def voice():
    twiml = '''<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Gather input="speech" action="/process_speech" method="POST" timeout="5">
        <Say>Hello, you're connected to the Fique AI assistant. Please say something after the beep.</Say>
    </Gather>
    <Say>Sorry, I didn't catch that. Goodbye!</Say>
</Response>'''
    return Response(content=twiml, media_type="application/xml")

@app.post("/process_speech", response_class=Response)
async def process_speech(request: Request, background_tasks: BackgroundTasks):
    try:
        form = await request.form()
        user_input = form.get("SpeechResult", "").strip()
        phone = form.get("From", "unknown")

        if not user_input:
            twiml = '''<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Say>Sorry, I didn't hear anything. Please try again.</Say>
</Response>'''
            return Response(content=twiml, media_type="application/xml")

        if phone not in conversation_history:
            conversation_history[phone] = []
        conversation_history[phone].append({"role": "user", "content": user_input, "timestamp": time.time()})

        metadata = {"phone": phone, "country": "auto", "language": "en"}
        result = await conversation_logic(conversation_history[phone], metadata)
        reply_text = saxutils.escape(result[0]["content"])

        conversation_history[phone].append({"role": "assistant", "content": reply_text, "timestamp": time.time()})

        print(f"📞 Voice interaction from {phone}: Input={user_input}, Reply={reply_text}")

        twiml = f'''<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Say>{reply_text}</Say>
</Response>'''
        return Response(content=twiml, media_type="application/xml")
    except Exception as e:
        print(f"❌ Error in process_speech: {str(e)}")
        twiml = '''<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Say>Sorry, an error occurred. Please try again later.</Say>
</Response>'''
        return Response(content=twiml, media_type="application/xml")

async def stream_response(messages, session_id):
    response_data = {
        "choices": [{"messages": messages}],
        "session_id": session_id
    }
    yield json.dumps(response_data) + "\n"
    yield "\n"

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("app:app", host="0.0.0.0", port=port)
