import gradio as gr
import requests
import pdfplumber
import docx
import os
import html

# --- Configuration ---
FASTAPI_BASE_URL = os.getenv("API_URL", "http://127.0.0.1:8000")
MATCH_ENDPOINT    = f"{FASTAPI_BASE_URL}/match-jobs"
FEEDBACK_ENDPOINT = f"{FASTAPI_BASE_URL}/feedback"

# --- Helper: text extraction ---
def extract_text_from_file(file_obj):
    if file_obj is None:
        return None, "Error: No file uploaded."
    path, ext = file_obj.name, os.path.splitext(file_obj.name)[1].lower()
    try:
        if ext == ".pdf":
            with pdfplumber.open(path) as pdf:
                return "\n".join(p.extract_text() or "" for p in pdf.pages), None
        elif ext == ".docx":
            doc = docx.Document(path)
            return "\n".join(p.text for p in doc.paragraphs), None
        elif ext == ".txt":
            return open(path, encoding="utf-8").read(), None
        else:
            return None, f"Unsupported filetype: {ext}"
    except Exception as e:
        return None, f"Error processing file: {e}"

# --- Core: call backend, filter by city, build HTML + feedback JS ---
def search_jobs(resume_file, top_k, city_filter):
    # 1) extract
    resume_text, err = extract_text_from_file(resume_file)
    if err:
        return f"⚠️ {err}", ""
    # 2) hit your FastAPI
    try:
        resp = requests.post(
            MATCH_ENDPOINT,
            json={"resume_text": resume_text, "top_k": top_k},
            timeout=120
        )
        resp.raise_for_status()
        jobs = resp.json().get("results", [])
    except Exception:
        return f"🚨 Could not connect to backend at {FASTAPI_BASE_URL}", ""
    if not jobs:
        return "❌ No matches found.", ""

    # 3) optional city‐filter (case‑insensitive substring match)
    if city_filter and city_filter.strip():
        cf = city_filter.strip().lower()
        jobs = [j for j in jobs if cf in (j.get("location","").lower())]
        if not jobs:
            return f"❌ No matches in “{city_filter}”", ""

    # 4) prepare JS + HTML grid
    resume_stub = html.escape(resume_text[:300], quote=True)
    out_html = f"""
    <script>
    function handle_feedback(elem, is_relevant) {{
      const jid = elem.dataset.jobId;
      const stub= elem.dataset.resumeStub;
      // UI ack
      const box = document.getElementById("feedback-"+jid);
      if(box) box.innerHTML = '<p class="feedback-thanks">✅ Thank you!</p>';
      // send
      fetch("{FEEDBACK_ENDPOINT}", {{
        method:"POST",
        headers:{{"Content-Type":"application/json"}},
        body: JSON.stringify({{
          resume_text: stub,
          job_id: jid,
          is_relevant
        }})
      }}).catch(console.error);
    }}
    </script>
    <div class="jobs-grid">
    """

    for j in jobs:
        jid   = html.escape(j.get("job_id",""), quote=True)
        title = html.escape(j.get("job_title","N/A"))
        comp  = html.escape(j.get("company_name","N/A"))
        loc   = html.escape(j.get("location","N/A"))
        desc  = html.escape(j.get("description","")[:250])
        url   = html.escape(j.get("job_board_url","#"), quote=True)
        score = f"{j.get('score',0):.2f}"

        out_html += f"""
        <div class="job-card">
          <h3>{title}</h3>
          <p class="meta">📍 {loc} | 💼 {comp} | ⭐ {score}</p>
          <p class="desc">{desc}…</p>
          <a href="{url}" target="_blank" class="job-link">View Original Post</a>
          <div id="feedback-{jid}" class="feedback-section">
            <button class="feedback-btn relevant"
                    data-job-id="{jid}"
                    data-resume-stub="{resume_stub}"
                    onclick="handle_feedback(this, 1)">
              👍 Relevant
            </button>
            <button class="feedback-btn not-relevant"
                    data-job-id="{jid}"
                    data-resume-stub="{resume_stub}"
                    onclick="handle_feedback(this, 0)">
              👎 Not Relevant
            </button>
          </div>
        </div>
        """

    out_html += "</div>"
    return "✅ Matches found! Filter by city if you like.", out_html

# --- CSS for grid, cards & buttons ---
css = """
.gradio-container { max-width:900px!important; margin:auto!important; }
.jobs-grid {
  display:flex; flex-wrap:wrap; gap:1rem; margin-top:1rem;
}
.job-card {
  background:white; border:1px solid #ddd; border-radius:8px;
  padding:16px; flex:1 1 calc(50% - 1rem); min-width:260px;
  box-shadow:0 2px 6px rgba(0,0,0,0.05);
}
.job-card h3 { margin:0 0 0.5rem; color:#333; }
.job-card .meta { font-size:0.9em; color:#555; margin-bottom:0.5rem; }
.job-card .desc { font-size:0.95em; color:#444; margin-bottom:0.5rem; }
.job-link { display:inline-block; margin-bottom:0.75rem; color:#007BFF; }
.job-link:hover { text-decoration:underline; }
.feedback-section { text-align:right; }
.feedback-btn {
  padding:6px 12px; margin-left:0.5rem; border-radius:20px;
  border:1px solid #ccc; cursor:pointer; transition:0.2s;
}
.feedback-btn.relevant { background:#e6f9f0; color:#1a7f46; }
.feedback-btn.not-relevant { background:#fcebeb; color:#c92a2a; }
.feedback-btn:hover { border-color:#888; }
.feedback-thanks { font-weight:bold; color:#28a745; }
"""

# --- Assemble Gradio UI ---
with gr.Blocks(css=css, title="🤖 JobMatcher AI") as demo:
    gr.Markdown("## 🤖 JobMatcher AI")
    gr.Markdown("# Semantic Search based Revolutionized way of Hunting Jobs with perfect alignment between your expertises and company's requirements.")  
    gr.Markdown("Upload your resume, filter by **city** (optional), set Top‑K, then hit **Search** and give 👍/👎 feedback!")
    with gr.Row():
        with gr.Column(scale=2):
            inp = gr.File(label="Resume (PDF/DOCX/TXT)", file_types=[".pdf",".docx",".txt"])
        with gr.Column(scale=1):
            city = gr.Textbox(label="City / Location (optional)")
            topk = gr.Slider(1,20,value=5,step=1,label="Top‑K")
    btn    = gr.Button("🔍 Search", variant="primary")
    status = gr.Textbox(interactive=False, value="Waiting for your input…")
    out    = gr.HTML()

    btn.click(
        fn=search_jobs,
        inputs=[inp, topk, city],
        outputs=[status, out]
    )

if __name__ == "__main__":
    demo.launch()
