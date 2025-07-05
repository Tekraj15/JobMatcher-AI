from fastapi import FastAPI, HTTPException, File, UploadFile
from pydantic import BaseModel
from dotenv import load_dotenv
import os
import pdfplumber
import docx
import re
from typing import Optional

from fastapi.middleware.cors import CORSMiddleware

from backend.embedding_and_matching import match_resume_to_jobs
from backend.feedback_logger import log_feedback

load_dotenv()

app = FastAPI(
    title="JobMatcher AI API",
    description="API for matching resumes to jobs and collecting user feedback.",
    version="2.2.0" # Intelligent Parsing Update
)

# --- CORS Configuration ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- NEW: Intelligent Section Extraction Logic ---
def extract_relevant_sections(text: str) -> str:
    """
    Parses resume text to extract content from the most semantically
    important sections, ignoring headers and personal info.
    """
    # Define common section headers. The regex will be case-insensitive.
    # We include variations like 'Work Experience' and 'Employment History'.
    section_headers = [
        'summary', 'objective', 'profile', 'professional summary',
        'experience', 'work experience', 'employment history',
        'skills', 'technical skills', 'competencies',
        'projects', 'personal projects', 'portfolio'
    ]

    # Create a regex pattern to find any of the headers at the start of a line.
    # It looks for the header, followed by optional whitespace, then a colon or newline.
    pattern = re.compile(r'^\s*(' + '|'.join(section_headers) + r')\s*[:\n]', re.IGNORECASE | re.MULTILINE)
    
    matches = list(pattern.finditer(text))

    if not matches:
        # Fallback strategy: If no standard headers are found, return the whole text.
        # The calling function will truncate it. This handles resumes without clear sections.
        print("Warning: No standard sections found in resume. Using full text.")
        return text

    # Extract text for each identified section
    extracted_content = []
    for i, match in enumerate(matches):
        start_pos = match.end()
        # The end of the section is the start of the next section, or the end of the document
        end_pos = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        
        section_text = text[start_pos:end_pos].strip()
        if section_text: # Only add if the section is not empty
            extracted_content.append(section_text)

    # Join the most important parts into a single block of text
    return "\n\n".join(extracted_content)


# --- Text Extraction from File (Now uses the intelligent parser) ---
def extract_text(file: UploadFile):
    extension = os.path.splitext(file.filename)[1].lower()
    raw_text = ""
    try:
        if extension == ".pdf":
            with pdfplumber.open(file.file) as pdf:
                raw_text = "\n".join(p.extract_text() or "" for p in pdf.pages)
        elif extension == ".docx":
            doc = docx.Document(file.file)
            raw_text = "\n".join(p.text for p in doc.paragraphs)
        elif extension == ".txt":
            raw_text = file.file.read().decode("utf-8")
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported file type: {extension}")
        
        # After getting raw text, apply the intelligent parser
        return extract_relevant_sections(raw_text)

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
    if not resume_file:
        raise HTTPException(status_code=400, detail="No resume file provided.")
        
    # This now returns clean, relevant text
    relevant_resume_text = extract_text(resume_file)
    
    if not relevant_resume_text:
        raise HTTPException(status_code=400, detail="Could not extract relevant text from the resume.")

    try:
        # Use the full relevant text for matching
        results = match_resume_to_jobs(relevant_resume_text, top_k)
        
        # Use a larger, cleaner stub for the feedback loop
        stub = relevant_resume_text[:1500].replace("\n", " ")
        
        return {"resume_stub": stub, "results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred during matching: {str(e)}")

@app.post("/feedback", summary="Receive User Feedback")
async def receive_feedback(payload: FeedbackPayload):
    try:
        log_feedback(payload.resume_text, payload.job_id, payload.is_relevant)
        return {"status": "feedback received successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to log feedback: {str(e)}")
