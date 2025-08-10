# JobMatcher-AI

**JobMatcher AI** is a semantic search job matching platform that uses Transformer models to recommend the best-matching jobs tailored to a candidate's key skill set and experience using Mantiks API + Pinecone Vector DB + Hugging Face.

---

## Motivation

Traditional job search platforms often require job seekers to manually scroll through dozens — sometimes hundreds — of job descriptions, line by line, to evaluate whether their skills and experiences truly align with each opportunity. While keyword-based search engines may return roles with similar titles, they frequently fail to capture deeper alignment:

Do the specific frameworks, tools, or domain expertise match the candidate’s background?

Does the job use a tech stack or architecture that the applicant is confident in?

Are subtle industry requirements being overlooked by a generic search?


This fragmented experience leads to wasted time, decision fatigue, and often, missed opportunities. JobMatcher AI was born from the idea that we can solve this pain point with the help of machine learning.
Instead of asking job seekers to evaluate every job one by one, why not train a model to do it for them — instantly and intelligently?

JobMatcher AI leverages state-of-the-art semantic search models to deeply analyze both resumes and job descriptions. Within seconds, it returns the top-matching jobs, not just by title, but by meaningful skill and context alignment.

JobMatcher AI is a modular end‑to‑end semantic search pipeline: raw job ingestion, vector embedding, Pinecone indexing, FastAPI-based search API, responsive HTML/JS frontend with real‑time feedback, and feedback‑driven fine‑tuning. Collaborators can add new data sources, improve parsing logic, or swap embedding models seamlessly.

##  Key Features

- Upload your resume (PDF/text)
- Retrieve top-k best matching jobs using semantic similarity (e5-base)
- Feedback loop to improve job matching quality
- Real-time FastAPI backend
- Gradio-powered demo UI
- Hugging Face Spaces-ready

---

## Tech Stack

| Layer       | Stack                           |
|-------------|---------------------------------|
| Frontend    | HTML, CSS(Tailwind) and JS      |
| Backend     | FastAPI + Uvicorn               |
| ML Model    | SentenceTransformer (e5-base-v2)|
| Vector Store| Pinecone                        |
| Hosting     | Hugging Face / Render           |

---

## Future Vision
As we enhance the platform, we plan to offer:
A concise, AI-generated match explanation for each job:

  “This job matches your profile because you have experience with Kubernetes, NLP, and data pipelines in healthcare — all of which are core to the role.”

With JobMatcher AI, candidates can stop second-guessing and start applying with confidence, saving time, energy, and unlocking better-fit opportunities.
