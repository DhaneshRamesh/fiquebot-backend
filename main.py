from fastapi import FastAPI, Request, Form, BackgroundTasks, HTTPException, Response
from fastapi.responses import PlainTextResponse, StreamingResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
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
from azure_search import search_articles
from utils import extract_metadata_from_message, needs_form
import xml.sax.saxutils as saxutils
from elevenlabs.client import ElevenLabs
from elevenlabs import VoiceSettings
from cosmos_client import update_preferences

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

# Validate environment variables
required_vars = [
    "AZURE_OPENAI_ENDPOINT", "AZURE_OPENAI_KEY", "AZURE_OPENAI_MODEL",
    "TWILIO_ACCOUNT_SID", "TWILIO_AUTH_TOKEN", "AZURE_STORAGE_CONNECTION_STRING",
    "COSMOS_DB_ENDPOINT", "COSMOS_DB_KEY"
]
for var in required_vars:
    if not os.environ.get(var):
        raise EnvironmentError(f"Missing required environment variable: {var}")

# Initialize FastAPI app
app = FastAPI()
conversation_history: Dict[str, List[Dict]] = {}

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://kind-island-057bb3903.6.azurestaticapps.net"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Pydantic models
class Message(BaseModel):
    role: str
    content: str

class ConversationRequest(BaseModel):
    messages: Optional[List[Message]] = []
    session_id: Optional[str] = None
    phone: Optional[str] = None
    country: Optional[str] = None
    language: Optional[str] = None

class FeedbackRequest(BaseModel):
    user_id: str
    fact_id: str
    liked: bool

# Root endpoint
@app.get("/")
async def root():
    return {"message": "Backend online!"}

# Audio transcription
async def transcribe_audio(audio_url: str) -> str:
    """Transcribe audio using Azure OpenAI Whisper."""
    print(f"🎙️ Transcribing audio: {audio_url}")
    try:
        async with httpx.AsyncClient(timeout=30, follow_redirects=True) as client:
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
        print(f"❌ Transcription error: {e}")
        return ""

# Implicit liking detection
async def detect_implicit_liking(session_id: str, conversation_history: List[Dict]) -> Dict:
    """Detect implicit liking based on conversation patterns."""
    print(f"🧠 Detecting implicit liking for session: {session_id}")
    try:
        if len([m for m in conversation_history if m["role"] == "user"]) < 2:
            return {"is_liked": False, "fact_id": None, "confidence": 0.0, "topic": None, "suggested_question": None}
        user_messages = [m["content"] for m in conversation_history if m["role"] == "user"][-5:]
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
        async with httpx.AsyncClient(timeout=15) as client:
            response = await client.post(
                f"{AZURE_OPENAI_ENDPOINT}/openai/deployments/{AZURE_OPENAI_MODEL}/chat/completions?api-version={AZURE_OPENAI_API_VERSION}",
                headers=headers,
                json=openai_body
            )
            response.raise_for_status()
            return json.loads(response.json()["choices"][0]["message"]["content"])
    except Exception as e:
        print(f"❌ Implicit liking error: {e}")
        return {"is_liked": False, "fact_id": None, "confidence": 0.0, "topic": None, "suggested_question": None}

# Conversation logic
async def conversation_logic(messages: List[Dict], metadata: Dict) -> List[Dict]:
    """Process conversation with search and implicit liking."""
    session_id = metadata.get("phone", str(uuid.uuid4()))
    user_question = messages[-1]["content"].strip().lower()
    print(f"🗣️ Question: {user_question}")

    liking_data = await detect_implicit_liking(session_id, messages)
    if liking_data["is_liked"] and liking_data["fact_id"]:
        try:
            update_preferences(user_id=session_id, fact_id=liking_data["fact_id"], liked=True, confidence=liking_data["confidence"])
            print(f"✅ Updated preferences: {liking_data['fact_id']}")
        except Exception as e:
            print(f"❌ Preference update error: {e}")

    def extract_keywords(text: str) -> str:
        words = re.findall(r'\w+', text.lower())
        stopwords = {"the", "of", "and", "in", "on", "is", "to", "a", "about"}
        return " ".join([w for w in words if w not in stopwords][:4])

    keywords = extract_keywords(user_question)
    search_contexts = search_articles(keywords) or []
    print(f"🔍 Search results: {len(search_contexts)} articles")

    fallback_phrases = ["yes", "sure", "go ahead", "general", "try again", "okay", "continue"]
    fallback_flag = any(
        phrase in m["content"].lower() for phrase in fallback_phrases
        for m in messages[-2:] if m["role"] == "user"
    )

    cleaned_messages = [{"role": m["role"], "content": m["content"]} for m in messages]
    response_content = ""

    if user_question in ["hi", "hello", "hey"]:
        response_content = f"{user_question.capitalize()}! How can I help you today? 💡 Curious about EMF sustainability?"
    elif not search_contexts and not fallback_flag:
        response_content = f"🤖 No articles found for '{keywords}'. Want a general answer? 💡 Or ask about EMF sustainability!"
        return [{"role": "assistant", "content": response_content}]
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
        async with httpx.AsyncClient(timeout=30) as client:
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
    return [{"role": "assistant", "content": response_content}]

# Feedback detection
async def detect_feedback(user_input: str) -> Dict:
    """Detect feedback in user input."""
    print(f"🧠 Detecting feedback: {user_input}")
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
        async with httpx.AsyncClient(timeout=15) as client:
            response = await client.post(
                f"{AZURE_OPENAI_ENDPOINT}/openai/deployments/{AZURE_OPENAI_MODEL}/chat/completions?api-version={AZURE_OPENAI_API_VERSION}",
                headers=headers,
                json=openai_body
            )
            response.raise_for_status()
            return json.loads(response.json()["choices"][0]["message"]["content"])
    except Exception as e:
        print(f"❌ Feedback detection error: {e}")
        return {"is_feedback": False, "fact_id": None, "liked": None}

# Fact ID extraction
async def extract_fact_id(previous_message: str) -> Optional[str]:
    """Extract or generate fact_id from assistant message."""
    print(f"🧠 Extracting fact_id: {previous_message[:50]}...")
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
        async with httpx.AsyncClient(timeout=15) as client:
            response = await client.post(
                f"{AZURE_OPENAI_ENDPOINT}/openai/deployments/{AZURE_OPENAI_MODEL}/chat/completions?api-version={AZURE_OPENAI_API_VERSION}",
                headers=headers,
                json=openai_body
            )
            response.raise_for_status()
            return json.loads(response.json()["choices"][0]["message"]["content"]).get("fact_id")
    except Exception as e:
        print(f"❌ Fact ID error: {e}")
        return None

# Process feedback
async def process_feedback(user_id: str, fact_id: str, liked: bool) -> tuple[bool, str]:
    """Save feedback and return response."""
    try:
        update_preferences(user_id, fact_id, liked, confidence=1.0)
        return True, f"✅ Recorded your {'like' if liked else 'dislike'} for fact {fact_id}."
    except Exception as e:
        print(f"❌ Feedback processing error: {e}")
        return False, "⚠️ Couldn’t save feedback."

# Conversation endpoint
@app.post("/conversation")
async def conversation_endpoint(request: Request):
    """Handle conversation requests."""
    try:
        payload = await request.json()
        messages_data = payload.get("messages", [])
        session_id = payload.get("session_id", str(uuid.uuid4()))
        print(f"📩 Payload: {payload}, Session: {session_id}")

        if not messages_data:
            return StreamingResponse(
                stream_response([{"role": "assistant", "content": "👋 Hi! How can I help you today? 💡 Curious about EMF sustainability?"}], session_id),
                media_type="text/event-stream"
            )

        valid_messages = [
            Message(role=msg["role"], content=msg["content"])
            for msg in messages_data if isinstance(msg, dict) and msg.get("role") and msg.get("content")
        ]
        if not valid_messages:
            return StreamingResponse(
                stream_response([{"role": "assistant", "content": "⚠️ Invalid message format."}], session_id),
                media_type="text/event-stream"
            )

        conversation_history.setdefault(session_id, []).append(
            {"role": "user", "content": valid_messages[-1].content, "timestamp": time.time()}
        )
        print(f"📜 History for {session_id}: {conversation_history[session_id]}")

        metadata = {
            "phone": payload.get("phone"),
            "country": payload.get("country"),
            "language": payload.get("language", "en")
        }
        all_user_text = " ".join(m.content for m in valid_messages if m.role == "user")

        if not (metadata["phone"] and metadata["country"]) and not session_id.startswith("whatsapp:"):
            extracted = extract_metadata_from_message(all_user_text)
            metadata.update({
                "phone": metadata["phone"] or extracted["phone"],
                "country": metadata["country"] or extracted["country"],
                "language": metadata["language"] or extracted["language"] or "en"
            })
            print(f"✅ Metadata: {metadata}")

        if session_id.startswith("whatsapp:") and len(conversation_history[session_id]) == 1 and needs_form(metadata):
            return StreamingResponse(
                stream_response([{"role": "assistant", "content": "⚠️ Please provide your language (e.g., English)."}], session_id),
                media_type="text/event-stream"
            )

        if session_id.startswith("whatsapp:") and len(conversation_history[session_id]) == 2 and metadata["phone"]:
            response = {
                "role": "assistant",
                "content": (
                    f"✅ Got it!\n- Language: {metadata['language']}\n- Phone: {metadata['phone']}\n"
                    f"📘 How can I help you today? 💡 Curious about EMF sustainability?"
                )
            }
            conversation_history[session_id].append({"role": "assistant", "content": response["content"], "timestamp": time.time()})
            return StreamingResponse(stream_response([response], session_id), media_type="text/event-stream")

        result = await conversation_logic(conversation_history[session_id], metadata)
        conversation_history[session_id].append({"role": "assistant", "content": result[0]["content"], "timestamp": time.time()})
        return StreamingResponse(stream_response(result, session_id), media_type="text/event-stream")

    except Exception as e:
        print(f"❌ Conversation error: {e}")
        return StreamingResponse(
            stream_response([{"role": "assistant", "content": "⚠️ Error processing your message."}], session_id),
            media_type="text/event-stream"
        )

# Audio cleanup
async def cleanup_audio(audio_path: str, delay: int = 300):
    """Delete audio file after delay."""
    await asyncio.sleep(delay)
    if os.path.exists(audio_path):
        os.remove(audio_path)
        print(f"🗑️ Deleted audio: {audio_path}")

# Serve audio
@app.get("/audio/{filename}")
async def get_audio(filename: str):
    audio_path = os.path.join("static/audio", filename)
    if not os.path.exists(audio_path):
        print(f"❌ Audio not found: {audio_path}")
        raise HTTPException(status_code=404, detail="Audio not found")
    print(f"🎵 Serving audio: {audio_path}")
    return FileResponse(audio_path, media_type="audio/mpeg")

# Upload audio
async def upload_audio_file(audio_path: str, audio_filename: str) -> str:
    """Upload audio to Azure Blob Storage."""
    try:
        blob_service_client = BlobServiceClient.from_connection_string(AZURE_STORAGE_CONNECTION_STRING)
        container_name = "audio-files"
        blob_client = blob_service_client.get_blob_client(container=container_name, blob=audio_filename)
        with open(audio_path, "rb") as f:
            blob_client.upload_blob(f, overwrite=True, content_settings=ContentSettings(content_type="audio/mpeg"))
        public_url = f"https://{blob_service_client.account_name}.blob.core.windows.net/{container_name}/{audio_filename}"
        async with httpx.AsyncClient(timeout=5) as client:
            response = await client.head(public_url)
            response.raise_for_status()
        print(f"📤 Uploaded audio: {public_url}")
        return public_url
    finally:
        blob_service_client.close()

# Twilio webhook
@app.post("/twilio-webhook")
async def handle_whatsapp(request: Request, background_tasks: BackgroundTasks, From: str = Form(...), Body: str = Form(...)):
    """Handle WhatsApp messages."""
    try:
        form = await request.form()
        user_input = Body.strip().lower()
        media_url = form.get("MediaUrl0")
        media_content_type = form.get("MediaContentType", "")
        print(f"📖 WhatsApp from {From}: {user_input}, Content-Type: {media_url}")

        feedback_data = await detect_feedback(user_input)
        feedback_processed = False
        feedback_response = ""

        if feedback_data.get("is_feedback"):
            fact_id = feedback_data.get("fact_id")
            liked = feedback_data.get("liked")
            if fact_id is None and liked is not None:
                last_message = next(
                    (msg["content"] for msg in reversed(conversation_history.get(From, [])) if msg["role"] == "assistant"),
                    None
                )
                if last_message:
                    fact_id = await extract_fact_id(last_message)
                    if fact_id:
                        feedback_processed, feedback_response = await process_feedback(From, fact_id, liked)
            elif fact_id and liked is not None:
                feedback_processed, feedback_response = await process_feedback(From, fact_id, liked)

        if media_url and "audio" in media_content_type.lower():
            transcribed_text = await transcribe_audio(media_url)
            user_input = transcribed_text.strip().lower() or "[Audio message]"
            print(f"🎙️ Transcription: {user_input}")
            feedback_data = await detect_feedback(user_input)
            if feedback_data.get("is_feedback"):
                fact_id = feedback_data.get("fact_id")
                liked = feedback_data.get("liked")
                if fact_id is None and liked is not None:
                    last_message = next(
                        (msg["content"] for msg in reversed(conversation_history.get(From, [])) if msg["role"] == "assistant"),
                        None
                    )
                    if last_message:
                        fact_id = await extract_fact_id(last_message)
                        if fact_id:
                            feedback_processed, feedback_response = await process_feedback(From, fact_id, liked)

        if user_input == "reset":
            if From in conversation_history:
                del conversation_history[From]
                print(f"✅ Reset history for {From}")
            return PlainTextResponse("✅ Conversation reset!")

        conversation_history.setdefault(From, []).append({"role": "user", "content": user_input, "timestamp": time.time()})

        metadata = {"phone": From, "country": None, "language": "en"}
        text_reply = feedback_response if feedback_processed else (await conversation_logic(conversation_history[From], metadata))[0]["content"]
        if feedback_processed and user_input not in ["i like this", "i dont like this"]:
            result = await conversation_logic(conversation_history[From], metadata)
            text_reply = f"{feedback_response}\n\n{result[0]['content']}"

        conversation_history[From].append({"role": "assistant", "content": text_reply, "timestamp": time.time()})

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
                audio_path = os.path.join(audio_dir, audio_filename)

                audio_files = elevenlabs_client.text_to_speech.convert(
                    voice_id=ELEVENLABS_VOICE_ID,
                    text=text_reply,
                    voice_settings=VoiceSettings(stability=0.25, similarity_boost=0.75),
                    output_format="mp3_44100_128"
                )
                with open(audio_path, "wb") as audio:
                    for chunk in audio_files:
                        audio.write(chunk)

                if os.path.getsize(audio_path) / (1024 * 1024) > 16:
                    raise Exception("Audio file too large")

                audio_url = await upload_audio_file(audio_path, audio_filename)
                message = client.messages.create(
                    media_url=[audio_url],
                    from_=TWILIO_PHONE_NUMBER,
                    to=From
                )
                print(f"📤 Audio sent: SID={message.sid}")
                background_tasks.add_task(cleanup_audio, audio_path)
            except Exception as e:
                print(f"❌ Audio error: {e}")
                message = client.messages.create(
                    body=text_reply,
                    from_=TWILIO_PHONE_NUMBER,
                    to=From
                )
                print(f"✅ Fallback text sent: SID={message.sid}")
        else:
            message = client.messages.create(
                body=text_reply,
                from_=TWILIO_PHONE_NUMBER,
                to=From
            )
            print(f"✅ Text sent: SID={message.sid}")

        return PlainTextResponse("")

    except Exception as e:
        print(f"❌ Twilio error: {e}")
        return PlainTextResponse("", status_code=500)

# Feedback endpoint
@app.post("/feedback")
async def feedback_endpoint(feedback: FeedbackRequest):
    """Process explicit feedback."""
    try:
        if not feedback.user_id.startswith(("whatsapp:", "uuid-")):
            raise HTTPException(status_code=400, detail="Invalid user_id")
        update_preferences(feedback.user_id, feedback.fact_id, feedback.liked, confidence=1.0)
        return {"status": "success"}
    except Exception as e:
        print(f"❌ Feedback error: {e}")
        return HTTPException(status_code=500, detail=str(e))

# Metadata extraction
@app.post("/extract_metadata")
async def extract_metadata_endpoint(request: Request):
    """Extract metadata from user input."""
    try:
        data = await request.json()
        user_input = data.get("text")
        if not user_input:
            return {"phone": None, "country": None, "language": "en", "confidence": 0.5}
        metadata = extract_metadata_from_message(user_input)
        print(f"📖 Metadata: {metadata}")
        return metadata
    except Exception as e:
        print(f"❌ Metadata error: {e}")
        return {"phone": None, "country": None, "language": "en", "confidence": 0.5}

# Voice endpoint
@app.post("/voice")
async def voice_response():
    """Handle Twilio voice calls."""
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
    """Process speech input from Twilio."""
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

        conversation_history.setdefault(phone, []).append({"role": "user", "content": user_input, "timestamp": time.time()})
        metadata = {"voice": phone, "content": None, "language": "en"}
        result = await conversation_logic(conversation_history[phone], metadata)
        reply_text = saxutils.escape(result[0]["content"])

        conversation_history[phone].append({"role": "muted", "content": reply_text, "timestamp": time.time()})
        print(f"📞 Voice: Input={user_input}, Reply={reply_text}")

        twiml = f'''<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Say>{reply_text}</Say>
</Response>'''
        return Response(content=twiml, media_type="text/xml")
    except Exception as e:
        print(f"❌ Voice error: {e}")
        twiml = '''<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Say>Sorry, an error occurred. Try again later.</Say>
</Response>'''
        return Response(content=twiml, media_type="text/xml")

# Stream response
async def stream_response(messages: List[Dict], session_id: str):
    """Stream response as NDJSON."""
    response_data = {"choices": [{"messages": messages}], "session_id": session_id}
    yield json.dumps(response_data) + "\n"
    yield "\n"

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
