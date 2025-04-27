from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List

# Create app
app = FastAPI()

# Setup CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://kind-island-057bb3903.6.azurestaticapps.net"],  # YOUR Frontend link here!
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define Models
class Message(BaseModel):
    role: str
    content: str

class ConversationRequest(BaseModel):
    messages: List[Message]

# Health check route
@app.get("/")
async def root():
    return {"message": "Backend online!"}

# Chatbot main endpoint
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
