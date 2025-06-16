from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware
from backend.embedding_and_matching import match_resume_to_jobs
from backend.feedback_logger import log_feedback

load_dotenv()

# FastAPI setup
app = FastAPI(
    title="JobMatcher AI API",
    description="API for matching resumes to best jobs and collecting user feedback.",
    version="1.0.0"
)

# --- CORS Configuration ---
# To tell the backend to allow web browser requests from the default Gradio origin (http://127.0.0.1:7860).

origins = [
    "http://localhost",
    "http://localhost:7860",
    "http://127.0.0.1",
    "http://127.0.0.1:7860",
    # We can add the URL of your deployed Gradio app here in future if needed
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # Allows specific origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods (GET, POST, etc.)
    allow_headers=["*"],  # Allows all headers
)
# --- End of CORS Configuration ---


# Request validation models
class ResumeQuery(BaseModel):
    resume_text: str
    top_k: int = Field(default=5, ge=1, le=50, description="Number of jobs to return (1–50)")

class FeedbackInput(BaseModel):
    resume_text: str
    job_id: str
    is_relevant: int  # 1 = relevant, 0 = not relevant


# API Endpoints
@app.get("/health", summary="Health Check")
def health_check():
    """Check if the API server is running."""
    return {"status": "ok"}

@app.post("/match-jobs", summary="Match Resume to Jobs")
def get_matched_jobs(data: ResumeQuery):
    """
    Accepts resume text and returns top_k semantically similar job postings.
    """
    try:
        results = match_resume_to_jobs(data.resume_text, data.top_k)
        return {"results": results}
    except Exception as e:
        # In a production app, you might want more specific logging here
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/feedback", summary="Receive User Feedback")
def receive_feedback(data: FeedbackInput):
    """
    Receives user feedback (relevant/not relevant) and logs it.
    """
    try:
        log_feedback(data.resume_text, data.job_id, data.is_relevant)
        return {"status": "feedback received successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

