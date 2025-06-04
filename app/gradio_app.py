import gradio as gr
from PyPDF2 import PdfReader
from backend.matching import match_resume_to_jobs

def extract_text_from_pdf(file_obj):
    reader = PdfReader(file_obj)
    return "\n".join([page.extract_text() or "" for page in reader.pages])

# Job matching
def match_resume_file(file, top_k):
    try:
        resume_text = extract_text_from_pdf(file)
        matches = match_resume_to_jobs(resume_text, top_k=top_k)
        if not matches:
            return "No matches found."

        return "\n\n".join(
            [f"🔹 {m['job_title']} at {m['company_name']} ({m['location']})\nScore: {m['score']:.2f}" for m in matches]
        )
    except Exception as e:
        return f"[Error] {str(e)}"

# Demo UI
demo = gr.Interface(
    fn=match_resume_file,
    inputs=[
        gr.File(file_types=[".pdf"], label="Upload your resume PDF"),
        gr.Slider(1, 50, value=5, step=1, label="Number of top matching jobs")
    ],
    outputs="text",
    title="JobMatcher AI - Resume Matching",
    description="Upload your resume as a PDF file and see top job matches based on semantic similarity."
)

if __name__ == "__main__":
    demo.launch()
