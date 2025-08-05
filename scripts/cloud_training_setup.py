#!/usr/bin/env python3
"""
Cloud Training Setup for JobMatcher AI

This script helps set up training on cloud instances (AWS EC2, GCP Compute Engine)
and automatically downloads the trained model back to your local pipeline.

Prerequisites:
- AWS CLI or GCP SDK configured
- SSH key pair set up
"""

import os
import subprocess
import json
from datetime import datetime

# Cloud training configurations
CLOUD_CONFIGS = {
    "aws": {
        "instance_type": "g4dn.xlarge",  # GPU instance
        "ami": "ami-0c02fb55956c7d316",  # Deep Learning AMI
        "region": "us-east-1",
        "storage": "30"
    },
    "gcp": {
        "machine_type": "n1-standard-4",
        "gpu_type": "nvidia-tesla-t4",
        "gpu_count": "1",
        "image": "pytorch-latest-gpu",
        "zone": "us-central1-a"
    }
}

def create_training_script():
    """Create the training script to run on cloud instance."""
    
    script_content = '''#!/bin/bash
# Cloud Training Script for JobMatcher AI

echo "Setting up training environment..."

# Install dependencies
pip install sentence-transformers pandas scikit-learn python-dotenv

# Create training directory
mkdir -p ~/jobmatcher_training
cd ~/jobmatcher_training

# Download data files (you'll need to upload these first)
echo "Place your feedback.csv and job data files in this directory"

# Create Python training script
cat > train_model.py << 'EOF'
import os
import pandas as pd
from sentence_transformers import SentenceTransformer, InputExample, losses
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from torch.utils.data import DataLoader
from datetime import datetime
from sklearn.model_selection import train_test_split
import json
import torch

print(f"GPU Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# Configuration
BASE_MODEL_NAME = "intfloat/e5-base-v2"
BATCH_SIZE = 16 if torch.cuda.is_available() else 4
EPOCHS = 3

# Data loading (same as your local version)
def load_feedback_data():
    df = pd.read_csv("feedback.csv")
    df = df.dropna(subset=["resume_text", "job_id", "is_relevant"])
    return df

def load_job_descriptions():
    files = [f for f in os.listdir('.') if 'job' in f and f.endswith(('.parquet', '.csv'))]
    if not files:
        raise FileNotFoundError("No job data found")
    
    file_path = files[0]
    if file_path.endswith('.parquet'):
        job_df = pd.read_parquet(file_path)
    else:
        job_df = pd.read_csv(file_path)
    
    return dict(zip(job_df["job_id"].astype(str), job_df["description"]))

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

# Main training
print("Loading data...")
feedback_df = load_feedback_data()
job_lookup = load_job_descriptions()
examples = create_training_examples(feedback_df, job_lookup)

train_samples, dev_samples = train_test_split(examples, test_size=0.2, random_state=42)

print(f"Training samples: {len(train_samples)}")
print(f"Validation samples: {len(dev_samples)}")

# Model setup
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = SentenceTransformer(BASE_MODEL_NAME, device=device)
train_dataloader = DataLoader(train_samples, shuffle=True, batch_size=BATCH_SIZE)
train_loss = losses.CosineSimilarityLoss(model)

evaluator = EmbeddingSimilarityEvaluator.from_input_examples(
    dev_samples, name='cloud-training'
)

# Training
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
model_save_path = f"./jobmatcher_cloud_{timestamp}"

print("Starting training...")
model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    epochs=EPOCHS,
    warmup_steps=int(len(train_dataloader) * 0.1),
    output_path=model_save_path,
    evaluator=evaluator,
    evaluation_steps=max(1, len(train_dataloader) // 4),
    save_best_model=True,
    show_progress_bar=True
)

print("Training completed!")

# Evaluation
fine_tuned_model = SentenceTransformer(model_save_path)
base_model = SentenceTransformer(BASE_MODEL_NAME)

base_eval = evaluator(base_model)
finetuned_eval = evaluator(fine_tuned_model)

print(f"Base model: {base_eval:.4f}")
print(f"Fine-tuned: {finetuned_eval:.4f}")
print(f"Improvement: {finetuned_eval - base_eval:+.4f}")

# Package for download
import shutil
shutil.make_archive(f"jobmatcher_cloud_{timestamp}", 'zip', model_save_path)
print(f"Model packaged: jobmatcher_cloud_{timestamp}.zip")
EOF

echo "Training setup complete!"
echo "Upload your data files and run: python train_model.py"
'''
    
    with open("cloud_training_setup.sh", "w") as f:
        f.write(script_content)
    
    os.chmod("cloud_training_setup.sh", 0o755)
    print("✅ Created cloud_training_setup.sh")

def launch_aws_instance():
    """Launch AWS EC2 instance for training."""
    
    config = CLOUD_CONFIGS["aws"]
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    instance_name = f"jobmatcher-training-{timestamp}"
    
    # Create instance
    cmd = [
        "aws", "ec2", "run-instances",
        "--image-id", config["ami"],
        "--instance-type", config["instance_type"],
        "--key-name", "your-key-pair",  # Update this
        "--security-groups", "default",
        "--block-device-mappings", f"DeviceName=/dev/sda1,Ebs={{VolumeSize={config['storage']},VolumeType=gp2}}",
        "--tag-specifications", f"ResourceType=instance,Tags=[{{Key=Name,Value={instance_name}}}]"
    ]
    
    print("🚀 Launching AWS EC2 instance...")
    print("Command:", " ".join(cmd))
    print("Note: Update key-pair name and security groups before running!")

def launch_gcp_instance():
    """Launch GCP Compute Engine instance for training."""
    
    config = CLOUD_CONFIGS["gcp"]
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    instance_name = f"jobmatcher-training-{timestamp}"
    
    cmd = [
        "gcloud", "compute", "instances", "create", instance_name,
        "--zone", config["zone"],
        "--machine-type", config["machine_type"],
        "--accelerator", f"type={config['gpu_type']},count={config['gpu_count']}",
        "--image-family", config["image"],
        "--image-project", "deeplearning-platform-release",
        "--maintenance-policy", "TERMINATE",
        "--restart-on-failure"
    ]
    
    print("🚀 Launching GCP Compute Engine instance...")
    print("Command:", " ".join(cmd))
    print("Note: Ensure GPU quotas are available in your project!")

def main():
    print("☁️  JobMatcher AI - Cloud Training Setup")
    print("="*50)
    
    create_training_script()
    
    print("\n📋 Cloud Training Options:")
    print("1. AWS EC2 with GPU")
    print("2. GCP Compute Engine with GPU")
    print("3. Manual setup instructions")
    
    choice = input("\nSelect option (1-3): ").strip()
    
    if choice == "1":
        launch_aws_instance()
    elif choice == "2":
        launch_gcp_instance()
    elif choice == "3":
        print("\n📖 Manual Setup Instructions:")
        print("1. Create a GPU instance on your preferred cloud provider")
        print("2. SSH into the instance")
        print("3. Copy cloud_training_setup.sh to the instance")
        print("4. Run: ./cloud_training_setup.sh")
        print("5. Upload your feedback.csv and job data files")
        print("6. Run: python train_model.py")
        print("7. Download the generated .zip file")
        print("8. Use scripts/integrate_trained_model.py to integrate locally")
    else:
        print("Invalid choice!")
    
    print("\n💰 Estimated Costs (per hour):")
    print("AWS g4dn.xlarge: ~$0.526/hour")
    print("GCP n1-standard-4 + T4: ~$0.35/hour")
    print("Training typically takes 30-60 minutes")

if __name__ == "__main__":
    main() 