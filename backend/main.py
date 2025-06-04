from fastapi import FastAPI, HTTPException, Body
from pydantic import BaseModel, Field
from backend.embedding_and_matching import match_resume_to_jobs
from backend.feedback_logger import log_feedback

# FastAPI setup
app = FastAPI(title="JobMatcher AI API")

# Request validation
class ResumeQuery(BaseModel):
    resume_text: str
    top_k: int = Field(default=5, ge=1, le=50, description="Number of jobs to return (1–50)")

class FeedbackInput(BaseModel):
    resume_text: str
    job_id: str
    is_relevant: int  # 1 = relevant, 0 = not relevant

@app.get("/health")
def health_check():
    return {"status": "ok"}

# Match-job endpoint
@app.post("/match-jobs")
def get_matched_jobs(data: ResumeQuery):
    try:
        results = match_resume_to_jobs(data.resume_text, data.top_k)
        return {"results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
# Feedback end-point
@app.post("/feedback")
def receive_feedback(data: FeedbackInput):
    try:
        log_feedback(data.resume_text, data.job_id, data.is_relevant)
        return {"status": "feedback received"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
