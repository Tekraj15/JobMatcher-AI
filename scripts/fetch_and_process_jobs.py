import os
import json
import pandas as pd
from datetime import datetime
from argparse import ArgumentParser
#from backend.mantiks_job_api import fetch_jobs_pipeline
from dotenv import load_dotenv
from loguru import logger
import hashlib

load_dotenv()

# Create output folder
DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

def get_timestamp():
    return datetime.now().strftime("%Y-%m-%d_%H-%M")

# Create own job_ids
def generate_id_from_url(url: str) -> str:
    return hashlib.md5(url.encode()).hexdigest()

def flatten_jobs_json(json_path, csv_path):
    with open(json_path, "r") as f:
        jobs = json.load(f)

    df = pd.json_normalize(jobs)
    df["job_id"] = df["job_board_url"].apply(generate_id_from_url)
    df.to_csv(csv_path, index=False)
    logger.success(f"Saved flattened CSV to {csv_path}")
    return df

def convert_csv_to_parquet(csv_path, parquet_path):
    df = pd.read_csv(csv_path)
    df.to_parquet(parquet_path, index=False)
    logger.success(f"Saved Parquet dataset to {parquet_path}")
    return df

def run_pipeline(job_titles, location_ids, limit):
    timestamp = get_timestamp()
    #json_path = os.path.join(DATA_DIR, f"jobs_{timestamp}.json")
    json_path = os.path.join(DATA_DIR, "jobs.json")
    csv_path = os.path.join(DATA_DIR, f"jobs_{timestamp}.csv")
    parquet_path = os.path.join(DATA_DIR, f"jobs_{timestamp}.parquet")

    logger.info(f"Fetching jobs for: {job_titles}, locations: {location_ids}")

    #fetch_jobs_pipeline(job_titles, location_ids, limit=limit, save_path=json_path)
    flatten_jobs_json(json_path, csv_path)
    convert_csv_to_parquet(csv_path, parquet_path)

    logger.success("Full job ingestion + processing pipeline completed.")

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--titles", nargs="+", default=["Data scientist", "Machine Learning Engineer", "Software Developer"])
    parser.add_argument("--location_ids", nargs="+", type=int, default=[2867714])
    parser.add_argument("--limit", type=int, default=10)

    args = parser.parse_args()

    run_pipeline(
        job_titles=args.titles,
        location_ids=args.location_ids,
        limit=args.limit
    )
