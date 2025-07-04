from fastapi import FastAPI, HTTPException, Request, File, UploadFile, Form
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel, Field
from dotenv import load_dotenv
import os
import pdfplumber
import docx
from typing import Optional

from fastapi.middleware.cors import CORSMiddleware

from backend.embedding_and_matching import match_resume_to_jobs
from backend.feedback_logger import log_feedback

load_dotenv()

app = FastAPI(
    title="JobMatcher AI API",
    description="API for matching resumes to jobs and collecting user feedback.",
    version="2.1.0" # Final Working Version
)

# --- CORS Configuration ---
# Allow all origins for simplicity in local development.
# In production, I would restrict this to my frontend's domain.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Text Extraction Logic ---
def extract_text(file: UploadFile):
    extension = os.path.splitext(file.filename)[1].lower()
    try:
        if extension == ".pdf":
            with pdfplumber.open(file.file) as pdf:
                return "\n".join(p.extract_text() or "" for p in pdf.pages)
        elif extension == ".docx":
            doc = docx.Document(file.file)
            return "\n".join(p.text for p in doc.paragraphs)
        elif extension == ".txt":
            return file.file.read().decode("utf-8")
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported file type: {extension}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")


# --- Request Models ---
class FeedbackPayload(BaseModel):
    resume_text: str
    job_id: str
    is_relevant: int

# --- API Endpoints ---
@app.get("/health", summary="Health Check")
def health_check():
    return {"status": "ok"}

@app.post("/match-resume-file", summary="Match Resume from File")
async def match_resume_from_file(top_k: int, resume_file: UploadFile = File(...)):
    """
    Accepts a resume file (PDF, DOCX, TXT), extracts text, and returns matches.
    """
    if not resume_file:
        raise HTTPException(status_code=400, detail="No resume file provided.")
        
    # --- CLEANUP: Call extract_text only ONCE ---
    resume_text = extract_text(resume_file)
    
    if not resume_text:
        raise HTTPException(status_code=400, detail="Could not extract text from the resume.")

    try:
        # Use the already extracted text
        stub = resume_text[:500].replace("\n", " ")
        results = match_resume_to_jobs(resume_text, top_k)
        return {"resume_stub": stub, "results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred during matching: {str(e)}")

@app.post("/feedback", summary="Receive User Feedback")
async def receive_feedback(payload: FeedbackPayload):
    """
    Receives user feedback (relevant/not relevant) and logs it.
    """
    try:
        log_feedback(payload.resume_text, payload.job_id, payload.is_relevant)
        return {"status": "feedback received successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to log feedback: {str(e)}")
