# Vector Embedding and Semantic Search Logic
import os
import pinecone
from ml.model import embed_text
from dotenv import load_dotenv

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV")
INDEX_NAME = "jobmatcher-index"

def init_pinecone():
    pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)
    return pinecone.Index(INDEX_NAME)

def match_resume_to_jobs(resume_text: str, top_k: int = 5):
    index = init_pinecone()
    query_vector = embed_text(resume_text)

    result = index.query(
        vector=query_vector.tolist(),
        top_k=top_k,
        include_metadata=True
    )

    matched_jobs = []
    for match in result.get("matches", []):
        matched_jobs.append({
            "score": match["score"],
            "job_title": match["metadata"].get("job_title"),
            "company_name": match["metadata"].get("company_name"),
            "location": match["metadata"].get("location"),
        })

    return matched_jobs
