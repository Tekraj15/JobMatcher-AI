from backend.mantiks_job_api import fetch_jobs_pipeline

fetch_jobs_pipeline(
    job_titles=["Data scientist", "ML engineer", "Data Engineer", "Working Student Data Engineer"],
    location_ids=[2867714],  # e.g., Munich
    limit=10
)
