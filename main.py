from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://kind-island-057bb3903.6.azurestaticapps.net"],  # your frontend
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
    return {"message": "Backend running with conversation endpoint!"}

@app.post("/conversation")
async def conversation_api(req: ConversationRequest):
    last_msg = req.messages[-1].content if req.messages else "Nothing received."
    return {
        "choices": [{
            "message": {
                "role": "assistant",
                "content": f"You said: {last_msg}"
            }
        }]
    }
