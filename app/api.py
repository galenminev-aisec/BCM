# app/api.py
import os
import traceback
from typing import List, Any, Dict
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from .rag import ask_bcp_assistant

app = FastAPI(
    title="BCP GRC Copilot API",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    query: str
    filters: Dict[str, Any] | None = None
    top_k: int = 5

class ContextSnippet(BaseModel):
    text: str
    metadata: Dict[str, Any]

class ChatResponse(BaseModel):
    answer: str
    contexts: List[ContextSnippet]

@app.get("/")
def serve_frontend():
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "index.html")
    return FileResponse(path)

@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    try:
        result = ask_bcp_assistant(
            user_query=req.query,
            filters=req.filters,
            top_k=req.top_k,
        )
        contexts = [
            ContextSnippet(text=c["text"], metadata=c.get("metadata", {}))
            for c in result["contexts"]
        ]
        return ChatResponse(
            answer=result["answer"],
            contexts=contexts,
        )
    except Exception as e:
        print("=== ГРЕШКА ===")
        print(traceback.format_exc())
        print("==============")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health():
    return {"status": "ok"}