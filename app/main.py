from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.analyze import router as analyze_router

app = FastAPI(
    title="Chiaki",
    description="Local video analysis service for cashier operations",
    version="0.1.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "X-API-Key"],
)
app.include_router(analyze_router)


@app.get("/health", tags=["Health"])
def health_check() -> dict[str, str]:
    return {"status": "ok"}