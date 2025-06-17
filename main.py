from fastapi import FastAPI, Request, Form, BackgroundTasks, HTTPException, Response
from fastapi.responses import PlainTextResponse, StreamingResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict
from typing import List, Optional, Dict
import os
import httpx
import re
import json
import uuid
import time
import asyncio
from dotenv import load_dotenv
from twilio.rest import Client
from azure.storage.blob import BlobServiceClient, ContentSettings
from azure.cosmos import CosmosClient, PartitionKey
from utils import extract_metadata_from_message, needs_form
import xml.sax.saxutils as saxutils
from elevenlabs.client import ElevenLabs
from elevenlabs import VoiceSettings
from cryptography.fernet import Fernet, InvalidToken
import base64
import logging
from contextlib import AsyncExitStack

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv(dotenv_path=".env.production")

# Configuration
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
COSMOS_DB_ENDPOINT = os.environ.get("COSMOS_DB_ENDPOINT")
COSMOS_DB_KEY = os.environ.get("COSMOS_DB_KEY")
FERNET_KEY = os.environ.get("FERNET_KEY")
MAX_AUDIO_SIZE_MB = float(os.environ.get("MAX_AUDIO_SIZE_MB", 16))
HTTP_TIMEOUT = float(os.environ.get("HTTP_TIMEOUT", 30))

# Validate environment variables
required_vars = [
    "AZURE_OPENAI_ENDPOINT", "AZURE_OPENAI_KEY", "AZURE_OPENAI_MODEL",
    "TWILIO_ACCOUNT_SID", "TWILIO_AUTH_TOKEN", "AZURE_STORAGE_CONNECTION_STRING",
    "COSMOS_DB_ENDPOINT", "COSMOS_DB_KEY", "FERNET_KEY"
]
for var in required_vars:
    if not os.environ.get(var):
        raise EnvironmentError(f"Missing required environment variable: {var}")

# Initialize Fernet for encryption
fernet = Fernet(FERNET_KEY.encode())

# Initialize FastAPI app
app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://kind-island-057bb3903.6.azurestaticapps.net"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Simple in-memory rate limiting
request_counts = {}
RATE_LIMIT = 10  # Requests per minute per IP
RATE_LIMIT_WINDOW = 60  # Seconds

# Pydantic models
class Message(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    role: str
    content: str

class ConversationRequest(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    messages: Optional[List[Message]] = []
    session_id: Optional[str] = None
    phone: Optional[str] = None
    country: Optional[str] = None
    language: Optional[str] = None

class FeedbackRequest(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    user_id: str
    fact_id: str
    liked: bool

# Cosmos DB utilities
async def get_conversation_history(session_id: str) -> List[Dict]:
    try:
        client = CosmosClient(COSMOS_DB_ENDPOINT, COSMOS_DB_KEY)
        database = client.get_database_client("ConversationsDB")
        container = database.get_container_client("Conversations")
        query = "SELECT * FROM c WHERE c.session_id = @session_id"
        items = [item async for item in container.query_items(
            query=query,
            parameters=[{"name": "@session_id", "value": session_id}],
            partition_key=session_id
        )]
        return items[0].get("messages", []) if items else []
    except Exception as e:
        logger.error(f"Failed to fetch conversation history for {session_id}: {e}")
        return []

async def append_conversation_history(session_id: str, message: Dict):
    try:
        client = CosmosClient(COSMOS_DB_ENDPOINT, COSMOS_DB_KEY)
        database = client.get_database_client("ConversationsDB")
        container = database.get_container_client("Conversations")
        existing = await get_conversation_history(session_id)
        existing.append(message)
        await container.upsert_item({
            "id": session_id,
            "session_id": session_id,
            "messages": existing
        })
        logger.info(f"Appended message to history for {session_id}")
    except Exception as e:
        logger.error(f"Failed to append conversation history for {session_id}: {e}")

async def reset_conversation_history(session_id: str):
    try:
        client = CosmosClient(COSMOS_DB_ENDPOINT, COSMOS_DB_KEY)
        database = client.get_database_client("ConversationsDB")
        container = database.get_container_client("Conversations")
        await container.delete_item(item=session_id, partition_key=session_id)
        logger.info(f"Reset conversation history for {session_id}")
    except Exception as e:
        logger.error(f"Failed to reset conversation history for {session_id}: {e}")

# Encryption/Decryption utilities
def encrypt_data(data: str) -> str:
    try:
        return fernet.encrypt(data.encode()).decode()
    except Exception as e:
        logger.error(f"Encryption error: {e}")
        raise HTTPException(status_code=500, detail="Encryption failed")

def decrypt_data(encrypted_data: str) -> str:
    try:
        return fernet.decrypt(encrypted_data.encode()).decode()
    except InvalidToken:
        logger.error("Decryption error: Invalid token")
        raise HTTPException(status_code=400, detail="Invalid encrypted data")
    except Exception as e:
        logger.error(f"Decryption error: {e}")
        raise HTTPException(status_code=500, detail="Decryption failed")

def encrypt_binary(data: bytes) -> bytes:
    try:
        return fernet.encrypt(data)
    except Exception as e:
        logger.error(f"Binary encryption error: {e}")
        raise HTTPException(status_code=500, detail="Binary encryption failed")

def decrypt_binary(encrypted_data: bytes) -> bytes:
    try:
        return fernet.decrypt(encrypted_data)
    except InvalidToken:
        logger.error("Binary decryption error: Invalid token")
        raise HTTPException(status_code=400, detail="Invalid encrypted binary data")
    except Exception as e:
        logger.error(f"Binary decryption error: {e}")
        raise HTTPException(status_code=500, detail="Binary decryption failed")

def normalize_metadata(metadata: Dict) -> Dict:
    return {
        "phone": encrypt_data(metadata["phone"]) if metadata.get("phone") and not metadata["phone"].startswith("enc:") else metadata.get("phone"),
        "country": encrypt_data(metadata["country"]) if metadata.get("country") and not metadata["country"].startswith("enc:") else metadata.get("country"),
        "language": encrypt_data(metadata["language"]) if metadata.get("language") and not metadata["language"].startswith("enc:") else metadata.get("language")
    }

# Root endpoint
@app.get("/")
async def root():
    return {"message": "Backend online!"}

# Audio transcription
async def transcribe_audio(audio_url: str) -> str:
    logger.info(f"Transcribing audio: {audio_url}")
    try:
        async with httpx.AsyncClient(timeout=HTTP_TIMEOUT, follow_redirects=True) as client:
            response = await client.get(audio_url, auth=(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN))
            response.raise_for_status()
            audio_content = response.content
            files = {"file": ("audio.mp3", audio_content, "audio/mpeg")}
            data = {"model": AZURE_WHISPER_MODEL}
            response = await client.post(
                f"{AZURE_OPENAI_ENDPOINT}/openai/deployments/{AZURE_WHISPER_MODEL}/audio/transcriptions?api-version={AZURE_OPENAI_API_VERSION}",
                headers={"api-key": AZURE_OPENAI_KEY},
                files=files,
                data=data
            )
            response.raise_for_status()
            return response.json().get("text", "")
    except Exception as e:
        logger.error(f"Transcription error: {e}")
        return ""

# Implicit liking detection
async def detect_implicit_liking(session_id: str, conversation_history: List[Dict]) -> Dict:
    logger.info(f"Detecting implicit liking for session: {session_id}")
    try:
        if len([m for m in conversation_history if m["role"] == "user"]) < 2:
            return {"is_liked": False, "fact_id": None, "confidence": 0.0, "topic": None, "suggested_question": None}
        user_messages = [decrypt_data(m["content"]) for m in conversation_history if m["role"] == "user"][-5:]
        openai_body = {
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "Analyze conversation for implicit liking (e.g., deeper questions, positive sentiment). "
                        "Generate fact_id and suggest a follow-up question. "
                        "Return JSON: { \"is_liked\": bool, \"fact_id\": str|null, \"confidence\": float, \"topic\": str|null, \"suggested_question\": str|null }."
                    )
                },
                {"role": "user", "content": "\n".join(user_messages)}
            ],
            "temperature": 0.3,
            "max_tokens": 150,
            "stream": False
        }
        headers = {"Content-Type": "application/json", "api-key": AZURE_OPENAI_KEY}
        async with httpx.AsyncClient(timeout=HTTP_TIMEOUT) as client:
            response = await client.post(
                f"{AZURE_OPENAI_ENDPOINT}/openai/deployments/{AZURE_OPENAI_MODEL}/chat/completions?api-version={AZURE_OPENAI_API_VERSION}",
                headers=headers,
                json=openai_body
            )
            response.raise_for_status()
            return json.loads(response.json()["choices"][0]["message"]["content"])
    except Exception as e:
        logger.error(f"Implicit liking error: {e}")
        return {"is_liked": False, "fact_id": None, "confidence": 0.0, "topic": None, "suggested_question": None}

# Conversation logic
async def conversation_logic(messages: List[Dict], metadata: Dict) -> List[Dict]:
    session_id = metadata.get("phone", str(uuid.uuid4()))
    user_question = decrypt_data(messages[-1]["content"]).strip().lower()
    logger.info(f"Processing question: {user_question}")

    liking_data = await detect_implicit_liking(session_id, messages)
    if liking_data["is_liked"] and liking_data["fact_id"]:
        try:
            from cosmos_client import update_preferences
            update_preferences(user_id=session_id, fact_id=liking_data["fact_id"], liked=True, confidence=liking_data["confidence"])
            logger.info(f"Updated preferences: {liking_data['fact_id']}")
        except Exception as e:
            logger.error(f"Preference update error: {e}")

    def extract_keywords(text: str) -> str:
        words = re.findall(r'\w+', text.lower())
        stopwords = {"the", "of", "and", "in", "on", "is", "to", "a", "about"}
        return " ".join([w for w in words if w not in stopwords][:4])

    keywords = extract_keywords(user_question)
    from azure_search import search_articles
    search_contexts = search_articles(keywords) or []
    logger.info(f"Search results: {len(search_contexts)} articles")

    fallback_phrases = ["yes", "sure", "go ahead", "general", "try again", "okay", "continue"]
    fallback_flag = any(
        phrase in decrypt_data(m["content"]).lower() for phrase in fallback_phrases
        for m in messages[-2:] if m["role"] == "user"
    )

    cleaned_messages = [{"role": m["role"], "content": decrypt_data(m["content"])} for m in messages]
    response_content = ""

    if user_question in ["hi", "hello", "hey"]:
        response_content = f"{user_question.capitalize()}! How can I help you today? 💡 Curious about EMF sustainability?"
    elif not search_contexts and not fallback_flag:
        response_content = f"🤖 No articles found for '{keywords}'. Want a general answer? 💡 Or ask about EMF sustainability!"
        return [{"role": "assistant", "content": encrypt_data(response_content)}]
    else:
        if search_contexts:
            context_block = "\n\n".join([f"{item['snippet']}\nSource: {item['title']} ({item['url']})" for item in search_contexts])
            cleaned_messages[-1]["content"] = (
                f"Answer conversationally, citing sources where relevant. Focus on EMF context if applicable:\n\n{context_block}\n\nQuestion: {user_question}"
            )
        else:
            cleaned_messages[-1]["content"] = f"Answer conversationally, focusing on EMF context if relevant: {user_question}"

        headers = {"Content-Type": "application/json", "api-key": AZURE_OPENAI_KEY}
        body = {"messages": cleaned_messages, "temperature": 0.7, "max_tokens": 500, "stream": False}
        async with httpx.AsyncClient(timeout=HTTP_TIMEOUT) as client:
            response = await client.post(
                f"{AZURE_OPENAI_ENDPOINT}/openai/deployments/{AZURE_OPENAI_MODEL}/chat/completions?api-version={AZURE_OPENAI_API_VERSION}",
                headers=headers,
                json=body
            )
            response.raise_for_status()
            result = response.json()

        if not result.get("choices"):
            response_content = "⚠️ Couldn’t generate a response. 💡 Try asking about EMF sustainability!"
        else:
            response_content = result["choices"][0]["message"]["content"]
            if search_contexts:
                source_links = "\n".join([f"- [{item['title']}]({item['url']})" for item in search_contexts])
                response_content += f"\n\n**Sources**:\n{source_links}"

    if liking_data["suggested_question"]:
        response_content += f"\n\n💡 Next question: {liking_data['suggested_question']}"
    return [{"role": "assistant", "content": encrypt_data(response_content)}]

# Feedback detection
async def detect_feedback(user_input: str) -> Dict:
    logger.info(f"Detecting feedback: {user_input}")
    try:
        openai_body = {
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "Detect feedback (e.g., liking/disliking a fact). "
                        "Return JSON: { \"is_feedback\": bool, \"fact_id\": str|null, \"liked\": bool|null }. "
                        "Examples: 'I like fact001' -> { \"is_feedback\": true, \"fact_id\": \"fact001\", \"liked\": true }, "
                        "'I like this' -> { \"is_feedback\": true, \"fact_id\": null, \"liked\": true }"
                    )
                },
                {"role": "user", "content": user_input}
            ],
            "temperature": 0,
            "max_tokens": 100,
            "stream": False
        }
        headers = {"Content-Type": "application/json", "api-key": AZURE_OPENAI_KEY}
        async with httpx.AsyncClient(timeout=HTTP_TIMEOUT) as client:
            response = await client.post(
                f"{AZURE_OPENAI_ENDPOINT}/openai/deployments/{AZURE_OPENAI_MODEL}/chat/completions?api-version={AZURE_OPENAI_API_VERSION}",
                headers=headers,
                json=openai_body
            )
            response.raise_for_status()
            return json.loads(response.json()["choices"][0]["message"]["content"])
    except Exception as e:
        logger.error(f"Feedback detection error: {e}")
        return {"is_feedback": False, "fact_id": None, "liked": None}

# Fact ID extraction
async def extract_fact_id(previous_message: str) -> Optional[str]:
    logger.info(f"Extracting fact_id: {previous_message[:50]}...")
    try:
        openai_body = {
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "Extract or generate fact_id (e.g., 'fact_emf_sustainability'). "
                        "Return JSON: { \"fact_id\": str|null }."
                    )
                },
                {"role": "user", "content": previous_message}
            ],
            "temperature": 0,
            "max_tokens": 50,
            "stream": False
        }
        headers = {"Content-Type": "application/json", "api-key": AZURE_OPENAI_KEY}
        async with httpx.AsyncClient(timeout=HTTP_TIMEOUT) as client:
            response = await client.post(
                f"{AZURE_OPENAI_ENDPOINT}/openai/deployments/{AZURE_OPENAI_MODEL}/chat/completions?api-version={AZURE_OPENAI_API_VERSION}",
                headers=headers,
                json=openai_body
            )
            response.raise_for_status()
            return json.loads(response.json()["choices"][0]["message"]["content"]).get("fact_id")
    except Exception as e:
        logger.error(f"Fact ID error: {e}")
        return None

# Process feedback
async def process_feedback(user_id: str, fact_id: str, liked: bool) -> tuple[bool, str]:
    try:
        from cosmos_client import update_preferences
        update_preferences(user_id, fact_id, liked, confidence=1.0)
        return True, f"✅ Recorded your {'like' if liked else 'dislike'} for fact {fact_id}."
    except Exception as e:
        logger.error(f"Feedback processing error: {e}")
        return False, "⚠️ Couldn’t save feedback."

# Conversation endpoint
@app.post("/conversation")
async def conversation_endpoint(request: Request):
    try:
        # Basic rate limiting
        client_ip = request.client.host
        current_time = time.time()
        if client_ip in request_counts:
            requests, start_time = request_counts[client_ip]
            if current_time - start_time < RATE_LIMIT_WINDOW:
                if requests >= RATE_LIMIT:
                    raise HTTPException(status_code=429, detail="Rate limit exceeded")
                request_counts[client_ip][0] += 1
            else:
                request_counts[client_ip] = [1, current_time]
        else:
            request_counts[client_ip] = [1, current_time]

        payload = await request.json()
        messages_data = payload.get("messages", [])
        session_id = payload.get("session_id", str(uuid.uuid4()))
        logger.info(f"Processing conversation payload for session: {session_id}")

        if not messages_data:
            response = [{"role": "assistant", "content": encrypt_data("👋 Hi! How can I help you today? 💡 Curious about EMF sustainability?")}]
            await append_conversation_history(session_id, response[0])
            return StreamingResponse(stream_response(response, session_id), media_type="text/event-stream")

        valid_messages = [
            Message(role=msg["role"], content=msg["content"])
            for msg in messages_data if isinstance(msg, dict) and msg.get("role") and msg.get("content")
        ]
        if not valid_messages:
            response = [{"role": "assistant", "content": encrypt_data("⚠️ Invalid message format.")}]
            await append_conversation_history(session_id, response[0])
            return StreamingResponse(stream_response(response, session_id), media_type="text/event-stream")

        await append_conversation_history(session_id, {"role": "user", "content": encrypt_data(valid_messages[-1].content), "timestamp": time.time()})

        metadata = normalize_metadata({
            "phone": payload.get("phone"),
            "country": payload.get("country"),
            "language": payload.get("language", "en")
        })

        conversation_history = await get_conversation_history(session_id)
        if not (metadata["phone"] and metadata["country"]) and not session_id.startswith("whatsapp:"):
            all_user_text = " ".join(decrypt_data(m.content) for m in valid_messages if m.role == "user")
            extracted = extract_metadata_from_message(all_user_text)
            metadata.update({
                "phone": metadata["phone"] or extracted["phone"],
                "country": metadata["country"] or extracted["country"],
                "language": metadata["language"] or extracted["language"] or "en"
            })
            metadata = normalize_metadata(metadata)
            logger.info(f"Extracted metadata: {metadata}")

        if session_id.startswith("whatsapp:") and len(conversation_history) == 1 and needs_form(metadata):
            response = [{"role": "assistant", "content": encrypt_data("⚠️ Please provide your language (e.g., English).")}]
            await append_conversation_history(session_id, response[0])
            return StreamingResponse(stream_response(response, session_id), media_type="text/event-stream")

        if session_id.startswith("whatsapp:") and len(conversation_history) == 2 and metadata["phone"]:
            response = {
                "role": "assistant",
                "content": encrypt_data(
                    f"✅ Got it!\n- Language: {decrypt_data(metadata['language'])}\n- Phone: {decrypt_data(metadata['phone'])}\n"
                    f"📘 How can I help you today? 💡 Curious about EMF sustainability?"
                )
            }
            await append_conversation_history(session_id, {"role": "assistant", "content": response["content"], "timestamp": time.time()})
            return StreamingResponse(stream_response([response], session_id), media_type="text/event-stream")

        result = await conversation_logic(conversation_history, metadata)
        await append_conversation_history(session_id, {"role": "assistant", "content": result[0]["content"], "timestamp": time.time()})
        return StreamingResponse(stream_response(result, session_id), media_type="text/event-stream")

    except Exception as e:
        logger.error(f"Conversation error: {e}")
        response = [{"role": "assistant", "content": encrypt_data("⚠️ Error processing your message.")}]
        await append_conversation_history(session_id, response[0])
        return StreamingResponse(stream_response(response, session_id), media_type="text/event-stream")

# Audio cleanup
async def cleanup_audio(audio_path: str, delay: int = 300):
    await asyncio.sleep(delay)
    if os.path.exists(audio_path):
        os.remove(audio_path)
        logger.info(f"Deleted audio: {audio_path}")

# Serve audio
@app.get("/audio/{filename}")
async def get_audio(filename: str):
    audio_path = os.path.join("static/audio", filename)
    if not os.path.exists(audio_path):
        logger.error(f"Audio not found: {audio_path}")
        raise HTTPException(status_code=404, detail="Audio not found")
    
    try:
        with open(audio_path, "rb") as f:
            encrypted_audio = f.read()
        decrypted_audio = decrypt_binary(encrypted_audio)
        
        temp_path = os.path.join("static/audio", f"temp_{filename}")
        with open(temp_path, "wb") as f:
            f.write(decrypted_audio)
        
        logger.info(f"Serving decrypted audio: {temp_path}")
        response = FileResponse(temp_path, media_type="audio/mpeg")
        
        asyncio.create_task(cleanup_audio(temp_path, delay=10))
        return response
    except Exception as e:
        logger.error(f"Audio decryption error: {e}")
        raise HTTPException(status_code=500, detail="Failed to decrypt audio")

# Upload audio
async def upload_audio_file(audio_path: str, audio_filename: str) -> str:
    async with AsyncExitStack() as stack:
        blob_service_client = await stack.enter_async_context(
            BlobServiceClient.from_connection_string(AZURE_STORAGE_CONNECTION_STRING)
        )
        container_name = "audio-files"
        blob_client = blob_service_client.get_blob_client(container=container_name, blob=audio_filename)
        
        with open(audio_path, "rb") as f:
            audio_data = f.read()
        if len(audio_data) / (1024 * 1024) > MAX_AUDIO_SIZE_MB:
            raise Exception(f"Audio file exceeds {MAX_AUDIO_SIZE_MB}MB limit")
        
        encrypted_audio = encrypt_binary(audio_data)
        
        await blob_client.upload_blob(
            encrypted_audio,
            overwrite=True,
            content_settings=ContentSettings(content_type="audio/mpeg")
        )
        public_url = f"https://{blob_service_client.account_name}.blob.core.windows.net/{container_name}/{audio_filename}"
        async with httpx.AsyncClient(timeout=HTTP_TIMEOUT) as client:
            response = await client.head(public_url)
            response.raise_for_status()
        logger.info(f"Uploaded encrypted audio: {public_url}")
        return public_url

# Twilio webhook
@app.post("/twilio-webhook")
async def handle_whatsapp(request: Request, background_tasks: BackgroundTasks, From: str = Form(...), Body: str = Form(...)):
    try:
        form = await request.form()
        user_input = Body.strip().lower()
        media_url = form.get("MediaUrl0")
        media_content_type = form.get("MediaContentType", "")
        logger.info(f"WhatsApp from {From}: {user_input}, Content-Type: {media_url}")

        feedback_data = await detect_feedback(user_input)
        feedback_processed = False
        feedback_response = ""

        if feedback_data.get("is_feedback"):
            fact_id = feedback_data.get("fact_id")
            liked = feedback_data.get("liked")
            if fact_id is None and liked is not None:
                conversation_history = await get_conversation_history(From)
                last_message = next(
                    (msg["content"] for msg in reversed(conversation_history) if msg["role"] == "assistant"),
                    None
                )
                if last_message:
                    fact_id = await extract_fact_id(decrypt_data(last_message))
                    if fact_id:
                        feedback_processed, feedback_response = await process_feedback(From, fact_id, liked)
            elif fact_id and liked is not None:
                feedback_processed, feedback_response = await process_feedback(From, fact_id, liked)

        if media_url and "audio" in media_content_type.lower():
            transcribed_text = await transcribe_audio(media_url)
            user_input = transcribed_text.strip().lower() or "[Audio message]"
            logger.info(f"Transcription: {user_input}")
            feedback_data = await detect_feedback(user_input)
            if feedback_data.get("is_feedback"):
                fact_id = feedback_data.get("fact_id")
                liked = feedback_data.get("liked")
                conversation_history = await get_conversation_history(From)
                if fact_id is None and liked is not None:
                    last_message = next(
                        (msg["content"] for msg in reversed(conversation_history) if msg["role"] == "assistant"),
                        None
                    )
                    if last_message:
                        fact_id = await extract_fact_id(decrypt_data(last_message))
                        if fact_id:
                            feedback_processed, feedback_response = await process_feedback(From, fact_id, liked)

        if user_input == "reset":
            await reset_conversation_history(From)
            return PlainTextResponse("✅ Conversation reset!")

        await append_conversation_history(From, {"role": "user", "content": encrypt_data(user_input), "timestamp": time.time()})

        metadata = normalize_metadata({"phone": From, "country": None, "language": "en"})
        conversation_history = await get_conversation_history(From)
        text_reply = decrypt_data(feedback_response) if feedback_processed else (await conversation_logic(conversation_history, metadata))[0]["content"]
        text_reply = decrypt_data(text_reply)
        if feedback_processed and user_input not in ["i like this", "i dont like this"]:
            result = await conversation_logic(conversation_history, metadata)
            text_reply = f"{feedback_response}\n\n{decrypt_data(result[0]['content'])}"

        await append_conversation_history(From, {"role": "assistant", "content": encrypt_data(text_reply), "timestamp": time.time()})

        client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
        voice_phrases = ["send voice", "reply in audio", "voice"]
        is_voice_request = any(phrase in user_input for phrase in voice_phrases)

        if (is_voice_request or media_url) and ELEVENLABS_API_KEY:
            text_reply = text_reply[:500].strip()
            try:
                elevenlabs_client = ElevenLabs(api_key=ELEVENLABS_API_KEY)
                audio_dir = os.path.join("static", "audio")
                os.makedirs(audio_dir, exist_ok=True)
                audio_filename = f"{uuid.uuid4()}.mp3"
                audio_path = os.path.join("static", "audio", audio_filename)

                audio_files = elevenlabs_client.text_to_speech.convert(
                    voice_id=ELEVENLABS_VOICE_ID,
                    text=text_reply,
                    voice_settings=VoiceSettings(stability=0.25, similarity_boost=0.75),
                    output_format="mp3_44100_128"
                )
                with open(audio_path, "wb") as audio:
                    for chunk in audio_files:
                        audio.write(chunk)

                if os.path.getsize(audio_path) / (1024 * 1024) > MAX_AUDIO_SIZE_MB:
                    raise Exception(f"Audio file exceeds {MAX_AUDIO_SIZE_MB}MB limit")

                audio_url = await upload_audio_file(audio_path, audio_filename)
                message = client.messages.create(
                    media_url=[audio_url],
                    from_=TWILIO_PHONE_NUMBER,
                    to=From
                )
                logger.info(f"Audio sent: SID={message.sid}")
                background_tasks.add_task(cleanup_audio, audio_path)
            except Exception as e:
                logger.error(f"Audio error: {e}")
                message = client.messages.create(
                    body=text_reply,
                    from_=TWILIO_PHONE_NUMBER,
                    to=From
                )
                logger.info(f"Fallback text sent: SID={message.sid}")
        else:
            message = client.messages.create(
                body=text_reply,
                from_=TWILIO_PHONE_NUMBER,
                to=From
            )
            logger.info(f"Text sent: SID={message.sid}")

        return PlainTextResponse("")

    except Exception as e:
        logger.error(f"Twilio error: {e}")
        return PlainTextResponse("", status_code=500)

# Feedback endpoint
@app.post("/feedback")
async def feedback_endpoint(feedback: FeedbackRequest):
    try:
        if not feedback.user_id.startswith(("whatsapp:", "uuid-")):
            raise HTTPException(status_code=400, detail="Invalid user_id")
        from cosmos_client import update_preferences
        update_preferences(feedback.user_id, feedback.fact_id, feedback.liked, confidence=1.0)
        return {"status": "success"}
    except Exception as e:
        logger.error(f"Feedback error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Metadata extraction
@app.post("/extract_metadata")
async def extract_metadata_endpoint(request: Request):
    try:
        data = await request.json()
        user_input = data.get("text")
        if not user_input:
            return {
                "phone": None,
                "country": None,
                "language": encrypt_data("en"),
                "confidence": 0.5
            }
        metadata = extract_metadata_from_message(user_input)
        encrypted_metadata = normalize_metadata({
            "phone": metadata["phone"],
            "country": metadata["country"],
            "language": metadata["language"] or "en"
        })
        encrypted_metadata["confidence"] = metadata.get("confidence", 0.5)
        logger.info(f"Extracted metadata: {encrypted_metadata}")
        return encrypted_metadata
    except Exception as e:
        logger.error(f"Metadata error: {e}")
        return {
            "phone": None,
            "country": None,
            "language": encrypt_data("en"),
            "confidence": 0.5
        }

# Voice endpoint
@app.post("/voice")
async def voice_response():
    twiml = '''<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Gather input="speech" action="/voices" method="POST">
        <Say>Hello, you're connected to the Fique AI assistant. Say something after the beep.</Say>
    </Gather>
    <Say>Sorry, I didn’t catch that. Goodbye.</Say>
</Response>'''
    return Response(content=twiml, media_type="text/xml")

# Process speech
@app.post("/voices")
async def process_speech(request: Request, voices: BackgroundTasks):
    try:
        form = await request.form()
        user_input = form.get("SpeechResult", "").strip()
        phone = form.get("From", "unknown")

        if not user_input:
            twiml = '''<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Say>Sorry, I didn’t hear anything. Try again.</Say>
</Response>'''
            return Response(content=twiml, media_type="text/xml")

        await append_conversation_history(phone, {"role": "user", "content": encrypt_data(user_input), "timestamp": time.time()})
        metadata = normalize_metadata({"phone": phone, "country": None, "language": "en"})
        conversation_history = await get_conversation_history(phone)
        result = await conversation_logic(conversation_history, metadata)
        reply_text = saxutils.escape(decrypt_data(result[0]["content"]))

        await append_conversation_history(phone, {"role": "muted", "content": encrypt_data(reply_text), "timestamp": time.time()})
        logger.info(f"Voice: Input={user_input}, Reply={reply_text}")

        twiml = f'''<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Say>{reply_text}</Say>
</Response>'''
        return Response(content=twiml, media_type="text/xml")
    except Exception as e:
        logger.error(f"Voice error: {e}")
        twiml = '''<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Say>Sorry, an error occurred. Try again later.</Say>
</Response>'''
        return Response(content=twiml, media_type="text/xml")

# Stream response
async def stream_response(messages: List[Dict], session_id: str):
    decrypted_messages = []
    for m in messages:
        try:
            decrypted_messages.append({"role": m["role"], "content": decrypt_data(m["content"])})
        except Exception as e:
            logger.error(f"Decryption error in stream_response: {e}")
            decrypted_messages.append({"role": m["role"], "content": "⚠️ Failed to decrypt message"})
    response_data = {"choices": [{"messages": decrypted_messages}], "session_id": session_id}
    yield json.dumps(response_data) + "\n"
    yield "\n"

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
