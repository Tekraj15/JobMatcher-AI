# Vector Embedding and Semantic Search Logic
import os
from pinecone import Pinecone
from ml.model import embed_text  # <-- Now correctly using your actual model script
from dotenv import load_dotenv

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV")
INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "jobmatcher-ai-index")

# Initialize Pinecone client
pc = Pinecone(api_key=PINECONE_API_KEY)

# Check if the index exists before trying to connect
if INDEX_NAME not in pc.list_indexes().names():
    raise EnvironmentError(f"Pinecone index '{INDEX_NAME}' does not exist. Please run ml/pipeline.py first.")

index = pc.Index(INDEX_NAME)

def match_resume_to_jobs(resume_text: str, top_k: int = 5):
    """
    Queries Pinecone for the top_k most similar jobs based on the resume text.
    This version correctly retrieves the 'job_id' from the Pinecone match result.
    """
    query_vector = embed_text(resume_text)

    # Ensure the vector is a list of floats for the API call
    if not isinstance(query_vector, list):
        query_vector = query_vector.tolist()

    result = index.query(
        vector=query_vector,
        top_k=top_k,
        include_metadata=True
    )

    matched_jobs = []
    for match in result.get("matches", []):
        metadata = match.get("metadata", {})
        
        # *** THE CRITICAL FIX ***
        # To retrieve the vector's ID from the top-level 'match' object.
        job_details = {
            "job_id": match.get("id"),
            "score": round(match["score"], 4),
            "job_title": metadata.get("job_title"),
            "company_name": metadata.get("company_name"),
            "location": metadata.get("location"),
            "last_seen": metadata.get("last_seen"),
            "job_board_url": metadata.get("job_board_url"),
            "description": metadata.get("description"),
        }
        matched_jobs.append(job_details)

    return matched_jobs
