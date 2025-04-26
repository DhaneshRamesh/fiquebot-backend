from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from history.routes import router as history_router
from auth.routes import router as auth_router
from security.routes import router as security_router

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(history_router, prefix="/history")
app.include_router(auth_router, prefix="/.auth")
app.include_router(security_router)

@app.get("/")
async def root():
    return {"message": "Backend running with endpoints!"}
