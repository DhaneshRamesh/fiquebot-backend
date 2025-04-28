from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import os
import httpx
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Environment Variables
AZURE_OPENAI_ENDPOINT = os.environ.get("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_KEY = os.environ.get("AZURE_OPENAI_KEY")
AZURE_OPENAI_MODEL = os.environ.get("AZURE_OPENAI_MODEL")
AZURE_OPENAI_API_VERSION = os.environ.get("AZURE_OPENAI_PREVIEW_API_VERSION", "2024-05-01-preview")

# Check required environment variables
if not AZURE_OPENAI_ENDPOINT or not AZURE_OPENAI_KEY or not AZURE_OPENAI_MODEL:
    raise ValueError("❌ Missing AZURE_OPENAI_* environment variables. Check your .env file!")

# Create FastAPI app
app = FastAPI()

# Setup CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://kind-island-057bb3903.6.azurestaticapps.net",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Models
class Message(BaseModel):
    role: str
    content: str

class ConversationRequest(BaseModel):
    messages: Optional[List[Message]] = []

# Dummy database for conversation history
mock_db = {"history": []}

# Health check route
@app.get("/")
async def root():
    return {"message": "Backend online!"}

# Conversation endpoint (REAL Azure OpenAI call)
@app.post("/conversation")
async def conversation_api(request: Request):
    try:
        payload = await request.json()
        messages = payload.get("messages", [])

        if not messages:
            return {
                "choices": [
                    {
                        "messages": [
                            {"role": "assistant", "content": "👋 Hello! You haven't said anything yet."}
                        ]
                    }
                ]
            }

        # Clean messages: keep only 'role' and 'content'
        cleaned_messages = [{"role": m["role"], "content": m["content"]} for m in messages]

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

        return {
            "choices": [
                {
                    "messages": [assistant_message]
                }
            ]
        }

    except Exception as e:
        print(f"❌ Error during OpenAI call: {e}")
        if hasattr(e, 'response'):
            print(f"❌ OpenAI API response: {e.response.text}")
        return {
            "choices": [
                {
                    "messages": [
                        {"role": "assistant", "content": "⚠️ Sorry, there was an error processing your message."}
                    ]
                }
            ]
        }

# History endpoints
@app.get("/history/ensure")
async def history_ensure():
    return {"message": "DB working (mocked)"}

@app.get("/frontend_settings")
async def frontend_settings():
    return {"ui": {"theme": "light", "language": "en"}}

@app.get("/history/list")
async def history_list(offset: int = 0):
    return mock_db["history"][offset:]

@app.post("/history/read")
async def history_read(payload: dict):
    conv_id = payload.get("conversation_id")
    for conv in mock_db["history"]:
        if conv["id"] == conv_id:
            return {"messages": conv.get("messages", [])}
    return {"messages": []}

@app.post("/history/generate")
async def history_generate(payload: dict):
    new_conv = {
        "id": f"conv_{len(mock_db['history'])}",
        "title": "New Conversation",
        "createdAt": "2025-04-27",
        "messages": payload.get("messages", [])
    }
    mock_db["history"].append(new_conv)
    return new_conv

@app.post("/history/update")
async def history_update(payload: dict):
    conv_id = payload.get("conversation_id")
    for conv in mock_db["history"]:
        if conv["id"] == conv_id:
            conv["messages"] = payload.get("messages", [])
            return {"status": "updated"}
    return {"error": "Conversation not found"}

@app.delete("/history/delete")
async def history_delete(payload: dict):
    conv_id = payload.get("conversation_id")
    mock_db["history"] = [conv for conv in mock_db["history"] if conv["id"] != conv_id]
    return {"status": "deleted"}

@app.delete("/history/delete_all")
async def history_delete_all():
    mock_db["history"] = []
    return {"status": "all deleted"}

@app.post("/history/clear")
async def history_clear(payload: dict):
    conv_id = payload.get("conversation_id")
    for conv in mock_db["history"]:
        if conv["id"] == conv_id:
            conv["messages"] = []
            return {"status": "cleared"}
    return {"error": "Conversation not found"}

@app.post("/history/rename")
async def history_rename(payload: dict):
    conv_id = payload.get("conversation_id")
    new_title = payload.get("title")
    for conv in mock_db["history"]:
        if conv["id"] == conv_id:
            conv["title"] = new_title
            return {"status": "renamed"}
    return {"error": "Conversation not found"}

@app.post("/history/message_feedback")
async def history_message_feedback(payload: dict):
    return {"status": "feedback logged"}
