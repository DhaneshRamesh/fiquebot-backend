from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List

# Create app
app = FastAPI()

# Setup CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://kind-island-057bb3903.6.azurestaticapps.net"],  # Your Azure Static Web App URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Models
class Message(BaseModel):
    role: str
    content: str

class ConversationRequest(BaseModel):
    messages: List[Message]

# Dummy database to simulate history
mock_db = {
    "history": []
}

# Health check route
@app.get("/")
async def root():
    return {"message": "Backend online!"}

# Chatbot main conversation endpoint
@app.post("/conversation")
async def conversation_api(request: ConversationRequest):
    if not request.messages:
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

    last_message = request.messages[-1].content
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
