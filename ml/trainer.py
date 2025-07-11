import os
import pandas as pd
from sentence_transformers import SentenceTransformer, InputExample, losses
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from torch.utils.data import DataLoader
from dotenv import load_dotenv
from datetime import datetime
from sklearn.model_selection import train_test_split
import json

load_dotenv()

# --- Configuration ---
BASE_MODEL_NAME = "intfloat/e5-base-v2"
SAVE_PATH_BASE = "./finetuned_model"  # The fine-tuned model will be saved to a timestamped folder

EVALUATION_REPORT_PATH = os.path.join(SAVE_PATH_BASE, "evaluation_reports") # save the evaluation results

# --- Data Loading Functions ---
def load_feedback_data(feedback_path="data/feedback.csv"):
    if not os.path.exists(feedback_path):
        return None
    df = pd.read_csv(feedback_path)
    df = df.dropna(subset=["resume_text", "job_id", "is_relevant"])
    return df

def load_job_descriptions(parquet_path="data/jobs.parquet"):
    # A fallback in case the main parquet file isn't found
    if not os.path.exists(parquet_path):
        print(f"[Warning] jobs.parquet not found at {parquet_path}. Looking for other parquet files in data/...")
        data_dir = os.path.dirname(parquet_path)
        parquet_files = [f for f in os.listdir(data_dir) if f.endswith('.parquet')]
        if not parquet_files:
            raise FileNotFoundError("No job data found in data/ directory.")
        
        # Use the most recent one
        parquet_path = os.path.join(data_dir, sorted(parquet_files)[-1])
        print(f"Using fallback job data: {parquet_path}")

    job_df = pd.read_parquet(parquet_path)
    job_lookup = dict(zip(job_df["job_id"].astype(str), job_df["description"]))
    return job_lookup

def create_training_examples(df, job_lookup):
    examples = []
    for _, row in df.iterrows():
        resume = row["resume_text"]
        job_id = str(row["job_id"])
        label = float(row["is_relevant"])
        job_desc = job_lookup.get(job_id)
        if job_desc:
            examples.append(InputExample(texts=[resume, job_desc], label=label))
    return examples

# --- Main Training and Evaluation Pipeline ---
def fine_tune_and_evaluate():
    print("[*] Loading data...")
    feedback_df = load_feedback_data()
    if feedback_df is None or len(feedback_df) < 20:
        print("[!] Not enough feedback data to run a full train/evaluation pipeline (need at least 20 samples).")
        return

    job_lookup = load_job_descriptions()
    examples = create_training_examples(feedback_df, job_lookup)

    if len(examples) < 20:
        print(f"[!] Only {len(examples)} valid training pairs found. Need at least 20 to proceed.")
        return

    print(f"[*] Loaded {len(examples)} total training pairs.")

    # --- 1. Train-Test Split ---
    train_samples, dev_samples = train_test_split(examples, test_size=0.2, random_state=42)
    print(f"[*] Training on {len(train_samples)} samples, evaluating on {len(dev_samples)} samples.")

    # --- 2. Prepare for Training
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    model_save_path = os.path.join(SAVE_PATH_BASE, timestamp)
    os.makedirs(model_save_path, exist_ok=True)
    os.makedirs(EVALUATION_REPORT_PATH, exist_ok=True)

    # Load the base model
    model = SentenceTransformer(BASE_MODEL_NAME)
    train_dataloader = DataLoader(train_samples, shuffle=True, batch_size=4)
    train_loss = losses.CosineSimilarityLoss(model)

    # --- 3. Setup the Evaluator ---
    # Compute the Spearman rank correlation between the model's similarity scores and the gold labels (0 or 1) on the development set.
    
    evaluator = EmbeddingSimilarityEvaluator.from_input_examples(
        dev_samples, 
        name='feedback-dev-set'
    )
    
    # --- 4. Fine-Tune the Model ---
    print("\n[*] Fine-tuning the model...")
    
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=2, # 1 epoch is often enough for fine-tuning
        warmup_steps=10,
        output_path=model_save_path,
        evaluator=evaluator,
        evaluation_steps=5, # Evaluate every 5 training steps
        save_best_model=True # Saves the best performing model on the dev set
    )
    
    print(f"\n[*] Fine-tuned model saved to {model_save_path}")

    # --- 5. Final Benchmark Evaluation ---
    print("\n[*] Starting final benchmark evaluation...")
    fine_tuned_model = SentenceTransformer(model_save_path)
    base_model = SentenceTransformer(BASE_MODEL_NAME)

    # Evaluate both models on the same hold-out test set
    base_model_eval = evaluator(base_model)
    fine_tuned_model_eval = evaluator(fine_tuned_model)
    
    # --- 6. Report Results ---
    report = {
        "timestamp": timestamp,
        "base_model": BASE_MODEL_NAME,
        "fine_tuned_model_path": model_save_path,
        "num_training_samples": len(train_samples),
        "num_evaluation_samples": len(dev_samples),
        "evaluation_results": {
            "base_model_spearman_correlation": base_model_eval,
            "fine_tuned_model_spearman_correlation": fine_tuned_model_eval,
            "improvement": fine_tuned_model_eval - base_model_eval
        }
    }

    print("\n" + "="*30)
    print("      EVALUATION COMPLETE")
    print("="*30)
    print(f"Base Model Performance (Spearman Correlation): {report['evaluation_results']['base_model_spearman_correlation']:.4f}")
    print(f"Fine-Tuned Model Performance (Spearman Correlation): {report['evaluation_results']['fine_tuned_model_spearman_correlation']:.4f}")
    print(f"Improvement: {report['evaluation_results']['improvement']:+.4f}")
    print("="*30)

    # Save the detailed report to a file
    report_file_path = os.path.join(EVALUATION_REPORT_PATH, f"report_{timestamp}.json")
    with open(report_file_path, 'w') as f:
        json.dump(report, f, indent=4)
    
    print(f"[*] Detailed evaluation report saved to {report_file_path}")

    if report['evaluation_results']['improvement'] > 0:
        print("[SUCCESS] Fine-tuning resulted in a performance improvement!")
    else:
        print("[INFO] Fine-tuning did not improve performance on the test set. Consider more data or different hyperparameters.")

if __name__ == "__main__":
    fine_tune_and_evaluate()
