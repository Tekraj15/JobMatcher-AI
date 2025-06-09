# End-to-end data flow
# Load jobs.parquet, embed job texts, and store vectors in a vector database (Pinecone)
import pandas as pd
from tqdm import tqdm
import os
import pinecone
from ml.model import embed_batch

# Load job data
job_data_path = "data/jobs.parquet"
index_name = "jobmatcher-index"
pinecone_api_key = os.getenv("PINECONE_API_KEY")
pinecone_env = os.getenv("PINECONE_ENV")

def create_pinecone_index():
    pinecone.init(api_key=pinecone_api_key, environment=pinecone_env)
    if index_name not in pinecone.list_indexes():
        pinecone.create_index(index_name, dimension=768)
    return pinecone.Index(index_name)

def embed_and_store_jobs():
    df = pd.read_parquet(job_data_path)
    df = df.dropna(subset=["job_title", "description"])
    df["text"] = df["job_title"] + " " + df["description"]

    vectors = []
    ids = []
    metadata = []

    index = create_pinecone_index()

    print(f"Embedding {len(df)} jobs...")
    for i in tqdm(range(0, len(df), 16)):
        batch = df["text"].iloc[i:i+16].tolist()
        batch_ids = df["job_id"].iloc[i:i+16].astype(str).tolist()
        batch_meta = df[["job_title", "company_name", "location"]].iloc[i:i+16].to_dict(orient="records")
        batch_vectors = embed_batch(batch)

        for id_, vec, meta in zip(batch_ids, batch_vectors, batch_meta):
            vectors.append((id_, vec.tolist(), meta))

    index.upsert(vectors)
    print(f"Uploaded {len(vectors)} job vectors to Pinecone.")

if __name__ == "__main__":
    embed_and_store_jobs()
