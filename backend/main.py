from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from backend.matching import match_resume_to_jobs

app = FastAPI(title="JobMatcher AI API")

class ResumeQuery(BaseModel):
    resume_text: str
    top_k: int = Field(default=5, ge=1, le=50, description="Number of jobs to return (1–50)")

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.post("/match-jobs")
def get_matched_jobs(data: ResumeQuery):
    try:
        results = match_resume_to_jobs(data.resume_text, data.top_k)
        return {"results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
