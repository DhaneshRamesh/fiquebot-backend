from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional

# Create app
app = FastAPI()

# Setup CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://kind-island-057bb3903.6.azurestaticapps.net",  # Frontend
        "https://fiquebot-backend.onrender.com"                 # Backend
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

# Dummy database to simulate history
mock_db = {
    "history": []
}

# Health check route
@app.get("/")
async def root():
    return {"message": "Backend online!"}

# 🚀 Conversation endpoint with RAW body logging
@app.post("/conversation")
async def conversation_api(request: Request):
    try:
        body = await request.body()
        print("🛜 Raw body received:", body)

        payload = await request.json()
        print("🚀 Received payload:", payload)

        messages = payload.get("messages", [])
        print("📝 Extracted messages:", messages)

        if not messages:
            print("⚠️ No messages found!")
            return {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "👋 Hello! You haven't said anything yet."
                        }
                    }
                ]
            }

        last_message = messages[-1]["content"]
        print("💬 Last user message:", last_message)

        response_content = f"👋 You said: '{last_message}'"

        return {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": response_content
                    }
                }
            ]
        }

    except Exception as e:
        print(f"❌ Error happened while processing conversation: {e}")
        return {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "⚠️ Sorry, there was an error processing your message."
                    }
                }
            ]
        }

# List conversation history
@app.get("/history/list")
async def history_list(offset: int = 0):
    return mock_db["history"][offset:]

# Read a specific conversation
@app.post("/history/read")
async def history_read(payload: dict):
    conv_id = payload.get("conversation_id")
    for conv in mock_db["history"]:
        if conv["id"] == conv_id:
            return {"messages": conv.get("messages", [])}
    return {"messages": []}

# Generate a new conversation
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

# Update an existing conversation
@app.post("/history/update")
async def history_update(payload: dict):
    conv_id = payload.get("conversation_id")
    for conv in mock_db["history"]:
        if conv["id"] == conv_id:
            conv["messages"] = payload.get("messages", [])
            return {"status": "updated"}
    return {"error": "Conversation not found"}

# Delete a specific conversation
@app.delete("/history/delete")
async def history_delete(payload: dict):
    conv_id = payload.get("conversation_id")
    mock_db["history"] = [conv for conv in mock_db["history"] if conv["id"] != conv_id]
    return {"status": "deleted"}

# Delete all conversation history
@app.delete("/history/delete_all")
async def history_delete_all():
    mock_db["history"] = []
    return {"status": "all deleted"}

# Clear messages in a conversation
@app.post("/history/clear")
async def history_clear(payload: dict):
    conv_id = payload.get("conversation_id")
    for conv in mock_db["history"]:
        if conv["id"] == conv_id:
            conv["messages"] = []
            return {"status": "cleared"}
    return {"error": "Conversation not found"}

# Rename a conversation
@app.post("/history/rename")
async def history_rename(payload: dict):
    conv_id = payload.get("conversation_id")
    new_title = payload.get("title")
    for conv in mock_db["history"]:
        if conv["id"] == conv_id:
            conv["title"] = new_title
            return {"status": "renamed"}
    return {"error": "Conversation not found"}

# Health check for CosmosDB status
@app.get("/history/ensure")
async def history_ensure():
    return {"message": "DB working"}

# Frontend settings endpoint
@app.get("/frontend_settings")
async def frontend_settings():
    return {"ui": {"theme": "light", "language": "en"}}

# Log message feedback
@app.post("/history/message_feedback")
async def history_message_feedback(payload: dict):
    return {"status": "feedback logged"}
