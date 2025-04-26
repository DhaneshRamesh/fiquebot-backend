from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://kind-island-057bb3903.6.azurestaticapps.net"],
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

# Endpoints
@app.get("/")
async def root():
    return {"message": "Backend is working."}

@app.post("/conversation")
async def conversation_api(request: ConversationRequest):
    last_message = request.messages[-1].content if request.messages else "No message"
    return {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": f"You said: {last_message}"
                }
            }
        ]
    }
