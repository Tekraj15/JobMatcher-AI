# End-to-end pipeline: loads job data, embeds text, and upserts to Pinecone
import os
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv
from ml.model import embed_batch
from pinecone import Pinecone, ServerlessSpec

# Load environment variables
load_dotenv()

# Constants
INDEX_NAME = "jobmatcher-ai-index"

#job_data_path = "data/jobs.parquet"
JOB_DATA_PATH = "data/jobs_2025-06-11_18-30.parquet"  # TEMP: update if needed

# Initialize Pinecone client
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV")

if not PINECONE_API_KEY or not PINECONE_ENV:
    raise EnvironmentError("Missing Pinecone API key or environment variable.")

pc = Pinecone(api_key=PINECONE_API_KEY)

def create_or_get_index(index_name=INDEX_NAME, dimension=768, metric="cosine", region="us-east-1"):
    existing = pc.list_indexes().names()

    if index_name not in existing:
        pc.create_index(
            name=index_name,
            dimension=dimension,
            metric=metric,
            spec=ServerlessSpec(cloud="aws", region=region)
        )
        print(f"✅ Index '{index_name}' created.")
    else:
        print(f"ℹ️ Index '{index_name}' already exists.")

    return pc.Index(index_name)

def embed_and_store_jobs(index):
    df = pd.read_parquet(JOB_DATA_PATH)
    df = df.dropna(subset=["job_title", "description", "job_id"])
    df["text"] = df["job_title"] + " " + df["description"]

    vectors = []

    print(f"🔍 Embedding {len(df)} jobs...")
    for i in tqdm(range(0, len(df), 16)):
        batch_texts = df["text"].iloc[i:i+16].tolist()
        batch_ids = df["job_id"].iloc[i:i+16].astype(str).tolist()
        batch_meta = df[["job_title", "company_name", "location"]].iloc[i:i+16].to_dict(orient="records")
        batch_vectors = embed_batch(batch_texts)

        for id_, vec, meta in zip(batch_ids, batch_vectors, batch_meta):
            vectors.append((id_, vec.tolist(), meta))

    print(f"Uploading {len(vectors)} job vectors to Pinecone...")

    BATCH_SIZE = 100
    for i in range(0, len(vectors), BATCH_SIZE):
        index.upsert(vectors[i:i + BATCH_SIZE])

    print("Upload complete!")

if __name__ == "__main__":
    index = create_or_get_index()
    embed_and_store_jobs(index)
