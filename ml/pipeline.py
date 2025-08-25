# End-to-end pipeline: loads job data, embeds text, and upserts to Pinecone
import os
import pandas as pd
from tqdm import tqdm
import time
from dotenv import load_dotenv
from ml.model import embed_batch
from pinecone import Pinecone, ServerlessSpec
from ml.skill_exp_extraction import extract_skills_and_exp

# Load environment variables
load_dotenv()

# Constants
INDEX_NAME = "jobmatcher-ai-index"

#job_data_path = "data/jobs.parquet"
JOB_DATA_PATH = "data/jobs_2025-08-14_22-30.parquet"  # TEMP: hardcoded value used for updating existing vectors; to be removed later

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
        print(f"Index '{index_name}' created.")
    else:
        print(f"Index '{index_name}' already exists.")

    return pc.Index(index_name)

def embed_and_store_jobs(index):
    df = pd.read_parquet(JOB_DATA_PATH)
    df = df.dropna(subset=["job_title","description","job_id","job_board_url","last_seen"])
    df["text"] = df["job_title"] + " " + df["description"]

    N = len(df)
    print(f"Embedding & upserting {N} jobs in micro-batches…")
    BATCH_SIZE = 20
    SNIPPET_LEN = 400

    for i in tqdm(range(0, N, BATCH_SIZE)):
        batch_df = df.iloc[i:i + BATCH_SIZE]
        texts = batch_df["text"].tolist()
        ids   = batch_df["job_id"].astype(str).tolist()

        # Build metadata with truncated descriptions
        metas = []
        for _, row in batch_df.iterrows():
            desc = row["description"]
            snippet = (desc[:SNIPPET_LEN] + "...") if len(desc) > SNIPPET_LEN else desc
            metas.append({
                "job_title":    row["job_title"],
                "company_name": row["company_name"],
                "location":     row["location"],
                "job_board_url":row["job_board_url"],
                "description":  snippet,
                "last_seen":    row["last_seen"]
            })

        vectors = embed_batch(texts)
        upserts = [(id_, vec.tolist(), meta) for id_, vec, meta in zip(ids, vectors, metas)]

        # Retry logic
        for attempt in range(1, 5):
            try:
                index.upsert(upserts)
                break
            except Exception as e:
                print(f"⚠️ Batch {i//BATCH_SIZE+1} upsert failed (attempt {attempt}): {e}")
                time.sleep(2 ** attempt)
        else:
            print(f"Giving up on batch {i//BATCH_SIZE+1}")

        time.sleep(0.5)

    print("All batches processed.")


def update_job_metadata(index):
    df = pd.read_parquet(JOB_DATA_PATH)
    df = df.dropna(subset=["job_id", "description"])

    N = len(df)
    print(f"Updating metadata for {N} jobs in batches...")
    BATCH_SIZE = 50

    for i in tqdm(range(0, N, BATCH_SIZE)):
        batch_df = df.iloc[i:i + BATCH_SIZE]
        batch_texts = batch_df["description"].tolist()
        batch_ids = batch_df["job_id"].astype(str).tolist()

        # Batched extraction
        batch_extractions = [extract_skills_and_exp(text) for text in batch_texts]

        for job_id, (req_skills, opt_skills, req_exp) in zip(batch_ids, batch_extractions):
            new_metadata = {
                "required_skills": list(req_skills),
                "optional_skills": list(opt_skills),
                "required_experience": req_exp
            }
            try:
                index.update(id=job_id, set_metadata=new_metadata)
            except Exception as e:
                print(f"⚠️ Failed to update metadata for job {job_id}: {e}")

        time.sleep(0.5)

    print("Metadata updates completed.")


if __name__ == "__main__":
    index = create_or_get_index()
    # embed_and_store_jobs(index)  # For full upsert
    update_job_metadata(index)  # For metadata-only updates