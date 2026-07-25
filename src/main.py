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
    url: str


class QuestionRequest(BaseModel):
    session_id: str
    question: str


@app.get("/")
async def serve_frontend():
    return FileResponse(STATIC_DIR / "index.html")


@app.post("/process")
async def process_video(req: VideoRequest):
    """Fetch transcript, build vectorstore, and start a new session."""
    try:
        transcript = get_transcript(req.url)
        vectorstore = build_vectorstore(transcript)
        qa_chain = create_qa_chain(vectorstore)

        session_id = str(uuid.uuid4())
        sessions[session_id] = {"vectorstore": vectorstore, "qa_chain": qa_chain}

        return {
            "status": "success",
            "message": "Video processed successfully.",
            "session_id": session_id,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ask")
async def ask_question(req: QuestionRequest):
    """Answer a question about a previously processed video."""
    session = sessions.get(req.session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Unknown or expired session_id. Call /process first.")

    qa_chain = session["qa_chain"]
    response = qa_chain.invoke({"question": req.question})
    return {"answer": response["answer"]}