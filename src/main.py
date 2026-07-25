import uuid
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from dotenv import load_dotenv
from .core.transcript import get_transcript
from .core.vectorstore import build_vectorstore
from .core.qa_chain import create_qa_chain

load_dotenv()

app = FastAPI(title="YouTube Q&A Bot", version="1.0.0") 

STATIC_DIR = Path(__file__).parent.parent / "static"
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# Keyed by session_id, so multiple videos/users don't clobber each other.
sessions: dict[str, dict] = {}


class VideoRequest(BaseModel):
    """Request model for processing a YouTube video."""
    url: str


class QuestionRequest(BaseModel):
    """Request model for asking a question about a processed video."""
    session_id: str
    question: str


def _is_connection_error(e: Exception) -> bool:
    """Check if the exception is related to a connection error."""
    msg = str(e).lower()
    return "connection" in msg or "connect" in msg or "refused" in msg


@app.get("/")
async def serve_frontend():
    return FileResponse(STATIC_DIR / "index.html")


@app.post("/process")
async def process_video(req: VideoRequest):
    """Fetch transcript, build vectorstore, and start a new session."""
    if not req.url or not req.url.strip():
        raise HTTPException(status_code=400, detail="Please provide a YouTube URL.")

    try:
        transcript = get_transcript(req.url)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error fetching transcript: {e}")

    try:
        vectorstore = build_vectorstore(transcript)
        qa_chain = create_qa_chain(vectorstore)
    except Exception as e:
        if _is_connection_error(e):
            raise HTTPException(
                status_code=503,
                detail="Could not reach Ollama. Make sure it's running (`ollama serve`) and the models are pulled.",
            )
        raise HTTPException(status_code=500, detail=f"Failed to build index: {e}")

    session_id = str(uuid.uuid4())
    sessions[session_id] = {"vectorstore": vectorstore, "qa_chain": qa_chain}

    return {
        "status": "success",
        "message": "Video processed successfully.",
        "session_id": session_id,
    }


@app.post("/ask")
async def ask_question(req: QuestionRequest):
    """Answer a question about a previously processed video."""
    if not req.question or not req.question.strip():
        raise HTTPException(status_code=400, detail="Please provide a question.")

    session = sessions.get(req.session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Unknown or expired session_id. Call /process first.")

    try:
        response = session["qa_chain"].invoke({"question": req.question})
    except Exception as e:
        if _is_connection_error(e):
            raise HTTPException(
                status_code=503,
                detail="Could not reach Ollama. Make sure it's running (`ollama serve`).",
            )
        raise HTTPException(status_code=500, detail=f"Failed to get an answer: {e}")

    return {"answer": response["answer"]}