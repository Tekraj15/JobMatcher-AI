# Vector Embedding and Semantic Search Logic
import os
from pinecone import Pinecone
from ml.model import embed_text
from dotenv import load_dotenv

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV")
INDEX_NAME = "jobmatcher-ai-index"

# Initialize Pinecone client
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(INDEX_NAME)

def match_resume_to_jobs(resume_text: str, top_k: int = 5):
    query_vector = embed_text(resume_text)

    result = index.query(
        vector=query_vector.tolist(),
        top_k=top_k,
        include_metadata=True
    )

    matched_jobs = []
    for match in result.get("matches", []):
        metadata = match.get("metadata", {})
        matched_jobs.append({
            "score": round(match["score"], 4),
            "job_title": metadata.get("job_title"),
            "company_name": metadata.get("company_name"),
            "location": metadata.get("location"),
            "last_seen": metadata.get("last_seen"),
            "job_board_url": metadata.get("job_board_url"),
            "description": metadata.get("description"),
        })

    return matched_jobs
