import gradio as gr
import requests
import pdfplumber
import docx
import os
import json
from io import BytesIO

# --- Configuration ---
# Assumes your FastAPI backend is running on the same machine on port 8000.
# Change this if your backend is hosted elsewhere.
FASTAPI_BASE_URL = os.getenv("API_URL", "http://127.0.0.1:8000")
MATCH_ENDPOINT = f"{FASTAPI_BASE_URL}/match-jobs"
FEEDBACK_ENDPOINT = f"{FASTAPI_BASE_URL}/feedback"

# --- Helper Function to Extract Text ---
def extract_text_from_file(file_obj):
    """
    Extracts text from uploaded file (PDF, DOCX, or TXT).
    """
    if file_obj is None:
        return None, "Error: No file uploaded."

    file_path = file_obj.name
    file_extension = os.path.splitext(file_path)[1].lower()
    
    try:
        if file_extension == ".pdf":
            # pdfplumber works directly with the file path
            with pdfplumber.open(file_path) as pdf:
                return "\n".join(page.extract_text() for page in pdf.pages if page.extract_text()), None
        elif file_extension == ".docx":
            # python-docx works with a file-like object
            doc = docx.Document(file_path)
            return "\n".join([para.text for para in doc.paragraphs]), None
        elif file_extension == ".txt":
            with open(file_path, "r", encoding="utf-8") as f:
                return f.read(), None
        else:
            return None, f"Error: Unsupported file type '{file_extension}'. Please upload a PDF, DOCX, or TXT file."
    except Exception as e:
        return None, f"Error processing file: {str(e)}"

# --- Core Gradio Interaction Functions ---

def search_jobs(resume_file, top_k, city_filter):
    """
    This function is called when the user clicks "Search Jobs".
    It sends the resume text to the FastAPI backend and formats the results.
    """
    # 1. Show processing status to the user
    yield "🔍 Extracting text from resume...", gr.HTML(visible=False)

    resume_text, error = extract_text_from_file(resume_file)
    if error:
        return f"⚠️ {error}", gr.HTML(visible=False)

    yield f"✅ Resume processed. Finding the best matches for you...", gr.HTML(visible=False)

    # 2. Prepare the request payload for the FastAPI backend
    payload = {
        "resume_text": resume_text,
        "top_k": top_k
        # Note: The current backend doesn't use a city filter.
        # This would be a future enhancement in `embedding_and_matching.py`.
        # "city": city_filter if city_filter else None
    }

    # 3. Call the backend API
    try:
        response = requests.post(MATCH_ENDPOINT, json=payload, timeout=120) # 120-second timeout
        response.raise_for_status()  # Raise an exception for bad status codes (4xx or 5xx)
        results = response.json().get("results", [])
    except requests.exceptions.RequestException as e:
        error_message = f"🚨 Network Error: Could not connect to the backend at {FASTAPI_BASE_URL}. Is it running?"
        return error_message, gr.HTML(visible=False)
    except Exception as e:
        return f"An unexpected error occurred: {str(e)}", gr.HTML(visible=False)

    if not results:
        return "No job matches found. Try broadening your search or using a different resume.", gr.HTML(visible=False)

    # 4. Format the results into HTML cards
    # Safely escape a snippet of the resume to be passed into the JavaScript feedback function
    resume_stub_js = json.dumps(resume_text[:500])
    
    html_output = ""
    for job in results:
        # NOTE: The job_id is crucial for the feedback mechanism.
        # Ensure your `match_resume_to_jobs` returns a unique ID for each job.
        # If it doesn't, we create one, but a persistent ID is better.
        job_id = job.get("job_id", f"temp_{os.urandom(8).hex()}")

        html_output += f"""
        <div class="job-card">
            <h3>{job.get('job_title', 'N/A')}</h3>
            <p>
                <strong>Match Score:</strong> {job.get('score', 0):.2f} | 
                <strong>Company:</strong> {job.get('company_name', 'N/A')} |
                <strong>Location:</strong> {job.get('location', 'N/A')}
            </p>
            <p class="description">
                {job.get('description', 'No description available.')[:350]}...
            </p>
            <a href="{job.get('job_board_url', '#')}" target="_blank" class="job-link">View Original Post</a>
            <div class="feedback-section" id="feedback-{job_id}">
                <p>Was this suggestion helpful?</p>
                <button class="feedback-btn relevant" onclick="handle_feedback({resume_stub_js}, '{job_id}', 1)">👍 Relevant</button>
                <button class="feedback-btn not-relevant" onclick="handle_feedback({resume_stub_js}, '{job_id}', 0)">👎 Not Relevant</button>
            </div>
        </div>
        """
        
    status_message = (
        "✅ Success! Here are your top job matches. "
        "Your feedback helps improve our recommendations."
    )
    yield status_message, gr.HTML(html_output, visible=True)


# --- Gradio UI Layout and Theming ---

custom_css = """
body { background-color: #F8F9FA; }
.gradio-container { max-width: 900px !important; margin: auto !important; }
.job-card {
    background-color: white;
    padding: 20px;
    border: 1px solid #E0E0E0;
    border-radius: 12px;
    margin-bottom: 20px;
    box-shadow: 0 4px 8px rgba(0,0,0,0.05);
    transition: box-shadow 0.3s ease;
}
.job-card:hover { box-shadow: 0 6px 12px rgba(0,0,0,0.1); }
.job-card h3 { margin-top: 0; color: #343A40; }
.job-card p { color: #495057; }
.job-card .description { font-size: 0.95em; line-height: 1.6; }
.job-link {
    display: inline-block;
    margin-top: 10px;
    text-decoration: none;
    color: #007BFF;
    font-weight: 500;
}
.feedback-section {
    margin-top: 15px;
    padding-top: 15px;
    border-top: 1px solid #f0f0f0;
    text-align: right;
}
.feedback-section p { display: inline; margin-right: 15px; font-size: 0.9em; color: #6c757d; }
.feedback-btn {
    padding: 8px 16px;
    border: 1px solid #dee2e6;
    border-radius: 20px;
    cursor: pointer;
    font-weight: 500;
    transition: all 0.2s ease;
}
.feedback-btn.relevant { background-color: #e6f9f0; color: #1a7f46; }
.feedback-btn.relevant:hover { background-color: #d1f4e0; border-color: #1a7f46; }
.feedback-btn.not-relevant { background-color: #fcebeb; color: #c92a2a; }
.feedback-btn.not-relevant:hover { background-color: #f8d7d7; border-color: #c92a2a; }
.feedback-thanks { color: #28a745; font-weight: bold; }
#status_box {
    padding: 15px;
    border-radius: 8px;
    font-weight: 500;
    text-align: center;
    background-color: #e9ecef;
    border: 1px solid #ced4da;
}
#search_button { background-color: #007BFF; color: white; }
"""

# This JavaScript function runs in the user's browser.
# It makes a direct `fetch` call to your FastAPI `/feedback` endpoint.
js_feedback_handler = f"""
function handle_feedback(resume_stub, job_id, is_relevant) {{
    console.log("Sending feedback for job_id:", job_id, "Relevance:", is_relevant);

    const feedback_payload = {{
        resume_text: resume_stub,
        job_id: job_id,
        is_relevant: is_relevant
    }};

    // Show immediate UI feedback
    const feedback_div = document.getElementById('feedback-' + job_id);
    if (feedback_div) {{
        feedback_div.innerHTML = '<p class="feedback-thanks">✅ Thank you for your feedback!</p>';
    }}

    // Make the API call to the backend
    fetch('{FEEDBACK_ENDPOINT}', {{
        method: 'POST',
        headers: {{
            'Content-Type': 'application/json',
            'Accept': 'application/json'
        }},
        body: JSON.stringify(feedback_payload)
    }})
    .then(response => response.json())
    .then(data => {{
        console.log('Feedback API response:', data);
        // You could update a global status message here if needed
    }})
    .catch(error => {{
        console.error('Error sending feedback:', error);
        // Optionally revert the "thank you" message on error
        if (feedback_div) {{
            feedback_div.innerHTML = '<p style="color:red;">Could not send feedback.</p>';
        }}
    }});
}}
"""

with gr.Blocks(theme=gr.themes.Soft(), title="JobMatcher AI", css=custom_css, js=js_feedback_handler) as demo:
    gr.Markdown("# 🤖 AI-Powered Job Search")
    gr.Markdown("Upload your resume and get a list of job postings that are the best fit for your skills and experience.")

    with gr.Row():
        with gr.Column(scale=2):
            resume_input = gr.File(label="Upload Resume (PDF, DOCX, or TXT)", file_types=[".pdf", ".docx", ".txt"])
        with gr.Column(scale=1):
            topk_slider = gr.Slider(1, 20, value=5, step=1, label="Number of Results")
            city_filter = gr.Textbox(label="Filter by City (Optional)", placeholder="e.g., San Francisco")
            
    search_btn = gr.Button("Search Jobs", variant="primary", elem_id="search_button")
    
    status_display = gr.Textbox(label="Status", interactive=False, elem_id="status_box", value="Please upload a resume and click Search.")
    
    output_display = gr.HTML(visible=False)

    # --- Event Handlers ---
    search_btn.click(
        fn=search_jobs,
        inputs=[resume_input, topk_slider, city_filter],
        outputs=[status_display, output_display]
    )

if __name__ == "__main__":
    print(f"Starting Gradio UI. Make sure your FastAPI backend is running at {FASTAPI_BASE_URL}")
    demo.launch()

