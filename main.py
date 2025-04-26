from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Or restrict to your Azure domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
