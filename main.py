from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
import os
from .core.transcript import get_transcript
from .core.vectorstore import build_vectorstore
from .core.qa_chain import create_qa_chain

load_dotenv()

app = FastAPI(title="YouTube Q&A Bot", version="1.0.0")

vectorstore = None
qa_chain = None


class VideoRequest(BaseModel):
    url: str


class QuestionRequest(BaseModel):
    question: str


@app.post("/process")
async def process_video(req: VideoRequest):
    """Fetch transcript and build vectorstore."""
    global vectorstore, qa_chain
    try:
        transcript = get_transcript(req.url)
        vectorstore = build_vectorstore(transcript)
        qa_chain = create_qa_chain(vectorstore)
        return {"status": "success", "message": "Video processed successfully."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ask")
async def ask_question(req: QuestionRequest):
    """Answer question about the processed video."""
    global qa_chain
    if qa_chain is None:
        raise HTTPException(status_code=400, detail="No video processed yet.")
    response = qa_chain.invoke({"question": req.question})
    return {"answer": response["answer"]}
