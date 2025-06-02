import csv
import os
from datetime import datetime

FEEDBACK_FILE = "data/feedback.csv"

def log_feedback(resume_text: str, job_id: str, is_relevant: int):
    """
    Append feedback to CSV: [timestamp, resume_text, job_id, is_relevant]
    """
    os.makedirs(os.path.dirname(FEEDBACK_FILE), exist_ok=True)
    is_new = not os.path.exists(FEEDBACK_FILE)

    with open(FEEDBACK_FILE, mode='a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)

        if is_new:
            writer.writerow(["timestamp", "resume_text", "job_id", "is_relevant"])

        writer.writerow([datetime.utcnow().isoformat(), resume_text[:500], job_id, is_relevant]) #resume_text[:500]: only store a partial text for privacy + dedup safety.
