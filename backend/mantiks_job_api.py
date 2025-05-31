import requests
import json
import os
from time import sleep
from dotenv import load_dotenv

load_dotenv()

# --- Config ---
API_KEY = os.getenv("MANTIKS_API_KEY")
HEADERS = {
    "accept": "application/json",
    "x-api-key": API_KEY
}

# --- Mantiks API Wrappers ---

def search_companies(job_titles, job_locations, job_age=30, limit=10):
    url = "https://api.mantiks.io/company/search"
    params = {
        "job_age_in_days": job_age,
        "job_location_ids": job_locations,
        "job_title": job_titles,
        "job_title_include_all": False,
        "min_company_size": 10,
        "max_company_size": 1000,
        "limit": limit
    }

    response = requests.get(url, headers=HEADERS, params=params)
    response.raise_for_status()
    return response.json().get("companies", [])


def fetch_job_postings(website):
    url = "https://api.mantiks.io/company/jobs"
    params = {
        "website": website,
        "age_in_days": 30
    }

    response = requests.get(url, headers=HEADERS, params=params)
    if response.status_code != 200:
        print(f"[!] Failed to get jobs for {website}")
        return []

    return response.json().get("jobs", [])


def fetch_jobs_pipeline(job_titles, location_ids, limit=10, save_path="data/jobs.json"):
    companies = search_companies(job_titles, location_ids, limit=limit)
    all_jobs = []

    for company in companies:
        website = company.get("website")
        if not website:
            continue

        jobs = fetch_job_postings(website)
        for job in jobs:
            job["company_name"] = company["name"]
            job["company_website"] = website
            all_jobs.append(job)

        sleep(1)

    if not all_jobs:
        print("[!] No jobs found.")
        return

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w") as f:
        json.dump(all_jobs, f, indent=2)

    print(f"[✓] Saved {len(all_jobs)} job listings to {save_path}")
