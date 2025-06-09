import os
import pandas as pd
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader
from dotenv import load_dotenv
from datetime import datetime

load_dotenv()

MODEL_NAME = "intfloat/e5-base"
SAVE_PATH = f"./finetuned_model/{datetime.now().strftime('%Y%m%d_%H%M')}"

# Reads data/feedback.csv entries
def load_feedback_data(feedback_path="data/feedback.csv"):
    df = pd.read_csv(feedback_path)
    df = df.dropna(subset=["resume_text", "job_id", "is_relevant"])
    return df

# Converts them into InputExample objects
def create_training_examples(df, job_lookup):
    examples = []
    for _, row in df.iterrows():
        resume = row["resume_text"]
        job_id = row["job_id"]
        label = float(row["is_relevant"])
        job_desc = job_lookup.get(job_id)

        if job_desc:
            examples.append(InputExample(texts=[resume, job_desc], label=label))

    return examples

def load_job_descriptions(parquet_path="data/jobs.parquet"):
    job_df = pd.read_parquet(parquet_path)
    job_lookup = dict(zip(job_df["job_id"].astype(str), job_df["description"]))
    return job_lookup

# Fine-tunes e5-base using CosineSimilarityLoss
def fine_tune_model():
    print("[*] Loading data...")
    df = load_feedback_data()
    job_lookup = load_job_descriptions()
    examples = create_training_examples(df, job_lookup)

    if len(examples) < 5:
        print("[!] Not enough feedback data to fine-tune.")
        return

    print(f"[*] Loaded {len(examples)} training pairs.")

    model = SentenceTransformer(MODEL_NAME)
    train_dataloader = DataLoader(examples, shuffle=True, batch_size=16)
    train_loss = losses.CosineSimilarityLoss(model)

    print("[*] Fine-tuning...")
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=2,
        warmup_steps=10,
        output_path=SAVE_PATH
    )

    # Saves the fine-tuned model locally (Later, to be saved to Hugging Face)
    print(f"Fine-tuned model saved to {SAVE_PATH}")
    return SAVE_PATH

if __name__ == "__main__":
    fine_tune_model()
