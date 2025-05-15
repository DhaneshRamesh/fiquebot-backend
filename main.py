
from fastapi import FastAPI, Request, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel
from typing import List, Optional
import os
import httpx
import re
from dotenv import load_dotenv
from azure_search import search_articles
from utils import extract_metadata_from_message, needs_form

load_dotenv(dotenv_path=".env.production")

AZURE_OPENAI_ENDPOINT = os.environ.get("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_KEY = os.environ.get("AZURE_OPENAI_KEY")
AZURE_OPENAI_MODEL = os.environ.get("AZURE_OPENAI_MODEL")
AZURE_OPENAI_API_VERSION = os.environ.get("AZURE_OPENAI_PREVIEW_API_VERSION", "2024-05-01-preview")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://kind-island-057bb3903.6.azurestaticapps.net"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Message(BaseModel):
    role: str
    content: str

class ConversationRequest(BaseModel):
    messages: Optional[List[Message]] = []

@app.get("/")
async def root():
    return {"message": "Backend online!"}

@app.post("/conversation")
async def conversation_logic(messages, metadata):
    try:
        cleaned_messages = [{"role": m["role"], "content": m["content"]} for m in messages]
        user_question = cleaned_messages[-1]['content']
        def extract_keywords(text):
            words = re.findall(r'\w+', text.lower())
            stopwords = {"what", "are", "the", "of", "and", "in", "on", "is", "to", "a", "how", "do", "does"}
            keywords = [w for w in words if w not in stopwords]
            return " ".join(keywords[:5])
        keywords = extract_keywords(user_question)
        search_contexts = search_articles(keywords)
        fallback_phrases = ["yes", "yeah", "sure", "go ahead", "please do", "try general", "fallback", "try again",
            "use gpt", "search online", "search web", "do it", "okay", "alright", "continue",
            "that’s fine", "proceed", "give me an answer", "show me anyway"]
        fallback_flag = any(
            any(phrase in m['content'].lower() for phrase in fallback_phrases)
            for m in messages[-2:]
        )
        if not search_contexts and not fallback_flag:
            return [{"role": "assistant", "content": "🤖 I couldn’t find anything in our articles. Would you like me to try a general answer instead?"}]
        if search_contexts:
            context_block = "\n\n".join([
                f"{item['snippet']}\n\nSource: {item['title']} ({item['url']})"
                for item in search_contexts
            ]) if isinstance(search_contexts[0], dict) else "\n\n".join(search_contexts)
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
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.post(
                f"{AZURE_OPENAI_ENDPOINT}/openai/deployments/{AZURE_OPENAI_MODEL}/chat/completions?api-version={AZURE_OPENAI_API_VERSION}",
                headers=headers,
                json=body
            )
        result = response.json()
        return [result["choices"][0]["message"]]
    except Exception as e:
        print("❌ Error in run_chatbot_logic:", str(e))
        import traceback; traceback.print_exc()
        return [{"role": "assistant", "content": "⚠️ Sorry, there was an error processing your message."}]
    
    import traceback
    try:
        payload = await request.json()
        messages_data = payload.get("messages", [])
    
        if not messages_data:
            return {
                "choices": [{
                    "messages": [{
                        "role": "assistant",
                        "content": "👋 Welcome! Before we begin, could you please tell me your *country* and *phone number*?"
                    }]
                }]
            }
        valid_messages = [
            Message(role=msg["role"], content=msg["content"])
            for msg in messages_data
            if isinstance(msg, dict) and msg.get("role") and msg.get("content")
        ]
    
        if not valid_messages:
            return {"choices": [{"messages": [{"role": "assistant", "content": "⚠️ Invalid message format."}]}]}
    
        cleaned_messages = [{"role": m.role, "content": m.content} for m in valid_messages]
        user_question = valid_messages[-1].content
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
    
        if needs_form(metadata):
            missing_fields = [k for k, v in metadata.items() if v is None]
            return {
                "choices": [{
                    "messages": [{
                        "role": "assistant",
                        "content": f"⚠️ I need more info to help you: missing {', '.join(missing_fields)}. Could you please provide it?"
                    }]
                }]
            }
    
        if len(valid_messages) < 3:
            return {
                "choices": [{
                    "messages": [{
                        "role": "assistant",
                        "content": (
    f"""✅ Got it! Here's what I understood:
    
    
    "
                            f"- Country: {metadata['country']}
    "
                            f"- Language: {metadata['language']}
    "
                            f"- Phone: {metadata['phone']}
    
    "
    f"📘 Now, what would you like to ask about Fique?"
    """
                        )
                    }]
                }]
            }
    
        fallback_phrases = [
            "yes", "yeah", "sure", "go ahead", "please do", "try general", "fallback", "try again",
            "use gpt", "search online", "search web", "do it", "okay", "alright", "continue",
            "that’s fine", "proceed", "give me an answer", "show me anyway"
        ]
    
        fallback_flag = any(
            any(phrase in m.content.lower() for phrase in fallback_phrases)
            for m in valid_messages[-2:]
        )
    
        def extract_keywords(text):
            words = re.findall(r'\w+', text.lower())
            stopwords = {"what", "are", "the", "of", "and", "in", "on", "is", "to", "a", "how", "do", "does"}
            keywords = [w for w in words if w not in stopwords]
            return " ".join(keywords[:5])
    
        keywords = extract_keywords(user_question)
        search_contexts = search_articles(keywords)
    
        if not search_contexts and not fallback_flag:
            return {
                "choices": [{
                    "messages": [{
                        "role": "assistant",
                        "content": "🤖 I couldn’t find anything in our articles. Would you like me to try a general answer instead?"
                    }]
                }]
            }
    
        if search_contexts:
            context_block = "\n\n".join([
                f"{item['snippet']}\n\nSource: {item['title']} ({item['url']})"
                for item in search_contexts
            ]) if isinstance(search_contexts[0], dict) else "\n\n".join(search_contexts)
    
            cleaned_messages[-1] = {
                "role": "user",
                "content": f"""Use the following context to answer the question. Cite the article title and URL explicitly.
    
    Context:
    {context_block}
    
    Question:
    {user_question}"""
            }
    
        headers = {
            "Content-Type": "application/json",
            "api-key": AZURE_OPENAI_KEY
        }
    
        body = {
            "messages": cleaned_messages,
            "temperature": 0.7,
            "top_p": 0.95,
            "frequency_penalty": 0,
            "presence_penalty": 0,
            "max_tokens": 1000,
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
        assistant_message = result["choices"][0]["message"]
    
        return {"choices": [{"messages": [assistant_message]}]}
    
    except Exception as e:
        print(f"❌ Error during OpenAI call: {e}")
        return {"choices": [{"messages": [{"role": "assistant", "content": "⚠️ Sorry, there was an error processing your message."}]}]}
    

@app.post("/conversation")
async def conversation_endpoint(request: Request):
    try:
        payload = await request.json()
        messages = payload.get("messages", [])
        metadata = {
            "phone": payload.get("phone"),
            "country": payload.get("country"),
            "language": payload.get("language", "en")
        }
        return {"choices": [{"messages": await conversation_logic(messages, metadata)}]}
    except Exception as e:
        print(f"❌ Error in conversation endpoint: {e}")
        return {"choices": [{"messages": [{"role": "assistant", "content": "⚠️ Sorry, internal error."}]}]}
@app.post("/twilio-webhook")
async def handle_whatsapp(From: str = Form(...), Body: str = Form(...)):
    print(f"📩 Received WhatsApp message from {From}: {Body}")
    user_input = Body.strip()
    messages = [{"role": "user", "content": user_input}]
    
    response = await run_chatbot_logic(messages, metadata)
        "type": "http",
        "method": "POST",
        "headers": {},
        response = await client.post("http://localhost:8000/chat", json={
            "messages": [{"role": "user", "content": Body}],
            "phone": From,
            "country": "auto",
            "language": "en"
        })

    text_reply = response["choices"][0]["messages"][0]["content"]
        return PlainTextResponse(text_reply)

@app.post("/extract_metadata")
async def extract_metadata_via_openai(request: Request):
    import json
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
            { "role": "user", "content": user_input }
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

    async with httpx.AsyncClient(timeout=30) as client:
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

from fastapi import Form
from fastapi.responses import PlainTextResponse
from twilio.twiml.messaging_response import MessagingResponse
import httpx

@app.post("/whatsapp")
async def whatsapp_webhook(
    From: str = Form(...),
    Body: str = Form(...)
):
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post("http://localhost:8000/chat", json={
                "messages": [{"role": "user", "content": Body}]
            })
            result = response.json()
    reply_text = response[0]["content"]
    except Exception as e:
        reply_text = "⚠️ Sorry, something went wrong."

    reply = MessagingResponse()
    reply.message(reply_text)
    return PlainTextResponse(str(reply))
