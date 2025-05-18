from fastapi import FastAPI, Request, Form, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List, Optional
import os
import httpx
import re
import json
import uuid
from dotenv import load_dotenv
from twilio.rest import Client
from azure_search import search_articles
from utils import extract_metadata_from_message, needs_form
import xml.sax.saxutils as saxutils
from elevenlabs.client import ElevenLabs
from elevenlabs import VoiceSettings

# Load environment variables
load_dotenv(dotenv_path=".env.production")

AZURE_OPENAI_ENDPOINT = os.environ.get("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_KEY = os.environ.get("AZURE_OPENAI_KEY")
AZURE_OPENAI_MODEL = os.environ.get("AZURE_OPENAI_MODEL")
AZURE_OPENAI_API_VERSION = os.environ.get("AZURE_OPENAI_PREVIEW_API_VERSION", "2024-05-01-preview")

# Twilio credentials
TWILIO_ACCOUNT_SID = os.environ.get("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = os.environ.get("TWILIO_AUTH_TOKEN")
TWILIO_PHONE_NUMBER = os.environ.get("TWILIO_PHONE_NUMBER", "whatsapp:+447488880990")
TWILIO_VOICE_NUMBER = os.environ.get("TWILIO_VOICE_NUMBER")

# ElevenLabs credentials
ELEVENLABS_API_KEY = os.environ.get("ELEVENLABS_API_KEY")
ELEVENLABS_VOICE_ID = os.environ.get("ELEVENLABS_VOICE_ID", "rachel")

# Validate required environment variables
required_vars = [
    "AZURE_OPENAI_ENDPOINT", "AZURE_OPENAI_KEY", "AZURE_OPENAI_MODEL",
    "TWILIO_ACCOUNT_SID", "TWILIO_AUTH_TOKEN"
]
for var in required_vars:
    if not os.environ.get(var):
        raise EnvironmentError(f"Missing required environment variable: {var}")

app = FastAPI()

# In-memory storage for conversation history (key: phone number or session ID, value: list of messages)
conversation_history = {}

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://kind-island-057bb3903.6.azurestaticapps.net"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static directory for audio files
app.mount("/static", StaticFiles(directory="static"), name="static")

class Message(BaseModel):
    role: str
    content: str

class ConversationRequest(BaseModel):
    messages: Optional[List[Message]] = []
    session_id: Optional[str] = None

@app.get("/")
async def root():
    return {"message": "Backend online!"}

async def conversation_logic(messages, metadata):
    try:
        cleaned_messages = [{"role": m["role"], "content": m["content"]} for m in messages]
        user_question = cleaned_messages[-1]['content']
        print(f"🧠 Calling OpenAI with messages: {cleaned_messages}")
        
        def extract_keywords(text):
            words = re.findall(r'\w+', text.lower())
            stopwords = {"what", "are", "the", "of", "and", "in", "on", "is", "to", "a", "how", "do", "does"}
            keywords = [w for w in words if w not in stopwords]
            return " ".join(keywords[:5])
        
        keywords = extract_keywords(user_question)
        search_contexts = search_articles(keywords) or []
        print(f"🔍 Search context: {search_contexts}")
        
        fallback_phrases = [
            "yes", "yeah", "sure", "go ahead", "please do", "try general", "fallback", "try again",
            "use gpt", "search online", "search web", "do it", "okay", "alright", "continue",
            "that’s fine", "proceed", "give me an answer", "show me anyway"
        ]
        fallback_flag = any(
            any(phrase in m['content'].lower() for phrase in fallback_phrases)
            for m in messages[-2:]
        )
        
        if not search_contexts and not fallback_flag:
            return [{"role": "assistant", "content": "🤖 I couldn’t find anything in our articles. Would you like me to try a general answer instead?"}]
        
        if search_contexts:
            context_block = "\n\n".join([
                f"{item['snippet']}\n\nSource: {item['title']} ({item['url']})"
                for item in search_contexts if isinstance(item, dict) and all(k in item for k in ["snippet", "title", "url"])
            ]) or "\n\n".join([str(item) for item in search_contexts])
            cleaned_messages[-1] = {
                "role": "user",
                "content": f"""Use the following context to answer the question. Cite the article title and URL explicitly.\n\nContext:\n{context_block}\n\nQuestion:\n{user_question}"""
            }
        
        headers = {"Content-Type": "application/json", "api-key": AZURE_OPENAI_KEY}
        body = {
            "messages": cleaned_messages,
            "temperature": 0.7,
            "top_p": 0.95,
            "frequency_penalty": 0,
            "presence_penalty": 0,
            "max_tokens": 1000,
            "stream": False,
        }
        
        async with httpx.AsyncClient(timeout=15) as client:
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
            return [{"role": "assistant", "content": "⚠️ I couldn’t generate a response."}]
        
        return [result["choices"][0]["message"]]
    
    except Exception as e:
        print(f"❌ Error in conversation_logic: {e}")
        import traceback
        traceback.print_exc()
        return [{"role": "assistant", "content": "⚠️ Sorry, there was an error processing your message."}]

async def stream_response(messages, session_id):
    # Simulate streaming by sending the response as newline-separated JSON objects
    response_data = {
        "choices": [{"messages": messages}],
        "session_id": session_id
    }
    # Send the response as a single chunk followed by an empty line to signal end
    yield json.dumps(response_data) + "\n"
    yield "\n"

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
        
        # Filter out invalid messages
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
        
        # Initialize or retrieve conversation history for this session
        if session_id not in conversation_history:
            conversation_history[session_id] = []
        
        # Add new user message to history
        latest_user_message = {"role": valid_messages[-1].role, "content": valid_messages[-1].content}
        conversation_history[session_id].append(latest_user_message)
        print(f"📜 Conversation history for session {session_id}: {conversation_history[session_id]}")
        
        all_user_text = " ".join([m.content for m in valid_messages if m.role == "user"])
        metadata = {
            "phone": payload.get("phone"),
            "country": payload.get("country"),
            "language": payload.get("language")
        }
        
        # Extract metadata if missing
        if not metadata["phone"] or not metadata["country"]:
            async with httpx.AsyncClient(timeout=10) as client:
                meta_response = await client.post(
                    f"{request.base_url}extract_metadata",
                    json={"text": all_user_text}
                )
            if meta_response.status_code == 200:
                metadata = meta_response.json()
                print(f"📋 Extracted metadata: {metadata}")
        
        # Ask for metadata only on the first message if incomplete
        if len(conversation_history[session_id]) == 1 and needs_form(metadata):
            missing_fields = [k for k, v in metadata.items() if v is None]
            return StreamingResponse(
                stream_response(
                    [{"role": "assistant", "content": f"⚠️ I need more info to help you: missing {', '.join(missing_fields)}. Could you please provide it?"}],
                    session_id
                ),
                media_type="text/event-stream"
            )
        
        # Confirm metadata on the second message if provided
        if len(conversation_history[session_id]) == 2 and (metadata["phone"] and metadata["country"]):
            response_message = {
                "role": "assistant",
                "content": (
                    f"""✅ Got it! Here's what I understood:\n\n"""
                    f"- Country: {metadata['country']}\n"
                    f"- Language: {metadata['language']}\n"
                    f"- Phone: {metadata['phone']}\n\n"
                    f"📘 Now, what would you like to ask about Fique?"
                )
            }
            conversation_history[session_id].append(response_message)
            return StreamingResponse(
                stream_response([response_message], session_id),
                media_type="text/event-stream"
            )
        
        # Proceed to conversation logic for subsequent messages
        result = await conversation_logic(conversation_history[session_id], metadata)
        conversation_history[session_id].append({"role": "assistant", "content": result[0]["content"]})
        return StreamingResponse(
            stream_response(result, session_id),
            media_type="text/event-stream"
        )
    
    except Exception as e:
        print(f"❌ Error during conversation: {e}")
        return StreamingResponse(
            stream_response(
                [{"role": "assistant", "content": "⚠️ Sorry, there was an error processing your message."}],
                session_id
            ),
            media_type="text/event-stream"
        )

async def cleanup_audio(audio_path, delay=300):
    import time
    time.sleep(delay)
    if os.path.exists(audio_path):
        os.remove(audio_path)
        print(f"🗑️ Deleted audio file: {audio_path}")

@app.post("/twilio-webhook")
async def handle_whatsapp(request: Request, background_tasks: BackgroundTasks, From: str = Form(...), Body: str = Form(...)):
    form = await request.form()
    user_input = Body.strip().lower()
    media_url = form.get("MediaUrl0")
    media_content_type = form.get("MediaContentType0")

    print(f"📩 Received WhatsApp message from {From}: Body={Body}, Media={media_url}")

    voice_phrases = ["send voice", "reply in audio", "voice answer", "audio response"]
    is_voice_request = any(phrase in user_input for phrase in voice_phrases)
    is_voice_message = media_url and media_content_type and "audio" in media_content_type.lower()

    if user_input == "reset":
        if From in conversation_history:
            del conversation_history[From]
            print(f"🔄 Reset conversation history for {From}")
        text_reply = "✅ Conversation reset. Let’s start fresh! Please tell me your country and phone number."
    else:
        if From not in conversation_history:
            conversation_history[From] = []
        
        conversation_history[From].append({"role": "user", "content": user_input or "[Voice message]"})
        print(f"📜 Conversation history for {From}: {len(conversation_history[From])} messages")
        
        metadata = {"phone": From, "country": "auto", "language": "en"}
        try:
            result = await conversation_logic(conversation_history[From], metadata)
            text_reply = result[0]["content"]
            conversation_history[From].append({"role": "assistant", "content": text_reply})
        except Exception as e:
            print(f"❌ Error in twilio-webhook: {e}")
            text_reply = "⚠️ Sorry, something went wrong."

    if not TWILIO_ACCOUNT_SID or not TWILIO_AUTH_TOKEN:
        print(f"❌ Twilio credentials missing: SID={TWILIO_ACCOUNT_SID}, Token={TWILIO_AUTH_TOKEN}")
        return PlainTextResponse("Twilio credentials missing", status_code=500)
    
    try:
        client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
        if (is_voice_request or is_voice_message) and ELEVENLABS_API_KEY and ELEVENLABS_VOICE_ID:
            elevenlabs_client = ElevenLabs(api_key=ELEVENLABS_API_KEY)
            audio_filename = f"{uuid.uuid4()}.mp3"
            audio_path = os.path.join("static/audio", audio_filename)
            
            audio_bytes = elevenlabs_client.text_to_speech.convert(
                voice_id=ELEVENLABS_VOICE_ID,
                text=text_reply,
                voice_settings=VoiceSettings(
                    stability=0.5,
                    similarity_boost=0.5
                )
            )
            
            with open(audio_path, "wb") as f:
                for chunk in audio_bytes:
                    if chunk:
                        f.write(chunk)
            
            audio_url = f"{os.environ.get('RENDER_DOMAIN')}/static/audio/{audio_filename}"
            message = client.messages.create(
                media_url=[audio_url],
                from_=TWILIO_PHONE_NUMBER,
                to=From
            )
            background_tasks.add_task(cleanup_audio, audio_path)
            print(f"✅ Audio message sent to {From}, SID: {message.sid}, URL: {audio_url}")
        else:
            message = client.messages.create(
                body=text_reply,
                from_=TWILIO_PHONE_NUMBER,
                to=From
            )
            print(f"✅ Text message sent to {From}, SID: {message.sid}")
    except Exception as e:
        print(f"❌ Error sending message via Twilio REST API: {str(e)}")
        return PlainTextResponse("", status_code=500)
    
    return PlainTextResponse("")

@app.post("/extract_metadata")
async def extract_metadata_via_openai(request: Request):
    data = await request.json()
    user_input = data.get("text")
    if not user_input:
        return {"error": "No text provided."}
    openai_body = {
        "messages": [
            {
                "role": "system",
                "content": "You are a metadata extraction assistant. Extract phone number, country, and language from the text below. Respond strictly in this JSON format: { \"phone\": \"\", \"country\": \"\", \"language\": \"\", \"confidence\": 0.95 }"
            },
            {"role": "user", "content": user_input}
        ],
        "temperature": 0,
        "max_tokens": 200,
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
            json=openai_body
        )
    try:
        result = response.json()
        reply = result["choices"][0]["message"]["content"]
        return json.loads(reply)
    except Exception as e:
        return {
            "error": "Failed to parse OpenAI response.",
            "raw": reply,
            "details": str(e)
        }

@app.post("/voice", response_class=Response)
async def voice():
    twiml = '''<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Gather input="speech" action="/process_speech" method="POST" timeout="5">
        <Say>Hello, you're now connected to the Fique AI assistant. Please say something after the beep.</Say>
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
        conversation_history[phone].append({"role": "user", "content": user_input})

        metadata = {"phone": phone, "country": "auto", "language": "en"}
        result = await conversation_logic(conversation_history[phone], metadata)
        reply_text = saxutils.escape(result[0]["content"])

        conversation_history[phone].append({"role": "assistant", "content": reply_text})

        print(f"📞 Voice interaction from {phone}: Input={user_input}, Reply={reply_text}")

        twiml = f'''<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Say>{reply_text}</Say>
</Response>'''
        return Response(content=twiml, media_type="application/xml")
    except Exception as e:
        print(f"❌ Error in process_speech: {e}")
        twiml = '''<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Say>Sorry, an error occurred. Please try again later.</Say>
</Response>'''
        return Response(content=twiml, media_type="application/xml")

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
