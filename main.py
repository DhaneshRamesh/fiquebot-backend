from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://kind-island-057bb3903.6.azurestaticapps.net"], # your frontend link
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

# Root Route
@app.get("/")
async def root():
    return {"message": "Backend working fine!"}

# Real /conversation Route
@app.post("/conversation")
async def conversation_api(request: ConversationRequest):
    if request.messages:
        last_user_message = request.messages[-1].content
    else:
        last_user_message = "Nothing provided."

    return {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": f"👋 You said: '{last_user_message}'"
                }
            }
        ]
    }
