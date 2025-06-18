from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel, Field
from dotenv import load_dotenv

from fastapi.middleware.cors import CORSMiddleware

from backend.embedding_and_matching import match_resume_to_jobs
from backend.feedback_logger import log_feedback

load_dotenv()

app = FastAPI(
    title="JobMatcher AI API",
    description="API for matching resumes to jobs and collecting user feedback.",
    version="1.1.0"
)

# --- CORS Configuration ---
origins = ["http://localhost:7860", "http://127.0.0.1:7860"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Request Models ---
class ResumeQuery(BaseModel):
    resume_text: str
    top_k: int = Field(default=5, ge=1, le=50)

# This is our data contract for the feedback endpoint
class FeedbackPayload(BaseModel):
    resume_text: str
    job_id: str
    is_relevant: int

# --- API Endpoints ---
@app.get("/health", summary="Health Check")
def health_check():
    return {"status": "ok"}

@app.post("/match-jobs", summary="Match Resume to Jobs")
def get_matched_jobs(data: ResumeQuery):
    try:
        results = match_resume_to_jobs(data.resume_text, data.top_k)
        return {"results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

# *** Making the feedback endpoint completely foolproof ***
@app.post("/feedback", summary="Receive User Feedback")
async def receive_feedback(request: Request):
    """
    Receives user feedback (relevant/not relevant) and logs it.
    This version manually parses the request to provide maximum debugging insight.
    """
    try:
        # Step 1: Manually get the raw JSON from the request.
        data = await request.json()
        
        # Step 2: Print the received data to the terminal. This is our source of truth.
        print("\n--- Received Feedback Payload ---")
        print(data)
        print("--------------------------------\n")

        # Step 3: Manually validate the data against our Pydantic model.
        # This will give a clear error if something is wrong.
        feedback_data = FeedbackPayload(**data)
        
        # Step 4: If validation passes, log the feedback.
        log_feedback(feedback_data.resume_text, feedback_data.job_id, feedback_data.is_relevant)
        
        return {"status": "feedback received successfully"}
    
    except Exception as e:
        # This will catch both validation errors and other problems.
        error_detail = f"Failed to process feedback. Error: {str(e)}. Received data: {data}"
        print(f"[ERROR] {error_detail}")
        raise HTTPException(status_code=422, detail=error_detail)

