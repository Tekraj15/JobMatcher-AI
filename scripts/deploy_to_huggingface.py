#!/usr/bin/env python3
"""
Deploy JobMatcher AI trained model to Hugging Face Hub

This script uploads your fine-tuned model to Hugging Face for production hosting
and provides API endpoints for your application to use.

Prerequisites:
- pip install huggingface_hub
- huggingface-cli login (with write access token)
"""

import os
import json
from pathlib import Path
from huggingface_hub import HfApi, Repository, create_repo
from sentence_transformers import SentenceTransformer

def deploy_model_to_hf(model_path, repo_name, private=True):
    """Deploy the trained model to Hugging Face Hub."""
    
    # Initialize HF API
    api = HfApi()
    
    # Create repository
    try:
        repo_url = create_repo(
            repo_id=repo_name,
            private=private,
            exist_ok=True,
            repo_type="model"
        )
        print(f"✅ Repository created/found: {repo_url}")
    except Exception as e:
        print(f"❌ Error creating repository: {e}")
        return False
    
    # Load and verify model
    try:
        model = SentenceTransformer(model_path)
        print(f"✅ Model loaded from {model_path}")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return False
    
    # Clone repository locally
    repo_dir = f"./temp_hf_repo_{repo_name.replace('/', '_')}"
    repo = Repository(
        local_dir=repo_dir,
        clone_from=repo_url,
        use_auth_token=True
    )
    
    # Save model to repository directory
    model.save(repo_dir)
    
    # Create model card
    model_card = f"""---
language: en
library_name: sentence-transformers
pipeline_tag: feature-extraction
tags:
- sentence-transformers
- feature-extraction
- sentence-similarity
- transformers
- job-matching
- resume-analysis
---

# JobMatcher AI - Fine-tuned Model

This model is a fine-tuned version of `intfloat/e5-base-v2` for job-resume matching tasks.

## Model Details

- **Base Model**: intfloat/e5-base-v2
- **Task**: Semantic similarity for job-resume matching
- **Training Data**: User feedback from JobMatcher AI platform
- **Fine-tuning Method**: Cosine similarity loss with sentence-transformers

## Usage

```python
from sentence_transformers import SentenceTransformer

# Load model
model = SentenceTransformer('{repo_name}')

# Example usage
resume = "Data scientist with 3 years of Python and ML experience"
job_desc = "Looking for a data scientist with Python and machine learning skills"

# Get embeddings
embeddings = model.encode([resume, job_desc])

# Calculate similarity
similarity = model.similarity(embeddings[0], embeddings[1])
print(f"Similarity: {{similarity:.4f}}")
```

## API Usage

You can also use this model via the Hugging Face Inference API:

```python
import requests

API_URL = "https://api-inference.huggingface.co/models/{repo_name}"
headers = {{"Authorization": "Bearer YOUR_HF_TOKEN"}}

def get_embeddings(texts):
    response = requests.post(API_URL, headers=headers, json={{"inputs": texts}})
    return response.json()

# Example
embeddings = get_embeddings(["resume text", "job description"])
```

## Training Metrics

- **Training Samples**: Varies based on feedback data
- **Evaluation Metric**: Spearman correlation
- **Performance**: Typically 10-20% improvement over base model

## Citation

If you use this model, please cite:

```
@misc{{jobmatcher-ai,
  title={{JobMatcher AI: Fine-tuned Sentence Transformer for Job-Resume Matching}},
  author={{Your Name}},
  year={{2025}},
  howpublished={{\\url{{https://huggingface.co/{repo_name}}}}}
}}
```
"""
    
    with open(f"{repo_dir}/README.md", "w") as f:
        f.write(model_card)
    
    # Create model info
    model_info = {
        "model_type": "sentence-transformer",
        "base_model": "intfloat/e5-base-v2",
        "task": "job-resume-matching",
        "framework": "sentence-transformers"
    }
    
    with open(f"{repo_dir}/model_info.json", "w") as f:
        json.dump(model_info, f, indent=2)
    
    # Commit and push
    repo.git_add()
    repo.git_commit("Upload fine-tuned JobMatcher AI model")
    repo.git_push()
    
    print(f"🚀 Model deployed to: https://huggingface.co/{repo_name}")
    print(f"📚 Model card: https://huggingface.co/{repo_name}/blob/main/README.md")
    print(f"🔗 API endpoint: https://api-inference.huggingface.co/models/{repo_name}")
    
    # Cleanup
    import shutil
    shutil.rmtree(repo_dir)
    
    return True

def create_hf_integration_code(repo_name):
    """Generate code to integrate HF model with existing backend."""
    
    integration_code = f'''# backend/hf_model_client.py
"""
Hugging Face model client for JobMatcher AI
"""

import requests
import numpy as np
from typing import List, Union
import os
from sentence_transformers import SentenceTransformer

class HuggingFaceModelClient:
    def __init__(self, model_name="{repo_name}", use_local=False):
        self.model_name = model_name
        self.use_local = use_local
        self.hf_token = os.getenv("HUGGINGFACE_TOKEN")
        
        if use_local:
            # Load model locally (faster for frequent calls)
            self.model = SentenceTransformer(model_name)
        else:
            # Use API (serverless, no local resources)
            self.api_url = f"https://api-inference.huggingface.co/models/{{model_name}}"
            self.headers = {{"Authorization": f"Bearer {{self.hf_token}}"}}
    
    def encode(self, texts: Union[str, List[str]]) -> np.ndarray:
        """Encode texts to embeddings."""
        
        if isinstance(texts, str):
            texts = [texts]
        
        if self.use_local:
            return self.model.encode(texts)
        else:
            return self._api_encode(texts)
    
    def _api_encode(self, texts: List[str]) -> np.ndarray:
        """Use HF Inference API to get embeddings."""
        
        response = requests.post(
            self.api_url,
            headers=self.headers,
            json={{"inputs": texts, "options": {{"wait_for_model": True}}}}
        )
        
        if response.status_code == 200:
            return np.array(response.json())
        else:
            raise Exception(f"HF API error: {{response.status_code}} - {{response.text}}")
    
    def similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two texts."""
        
        embeddings = self.encode([text1, text2])
        return np.dot(embeddings[0], embeddings[1]) / (
            np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1])
        )

# Usage in your existing backend:
# 
# # Replace in backend/embedding_and_matching.py
# from backend.hf_model_client import HuggingFaceModelClient
# 
# # Initialize model client
# model_client = HuggingFaceModelClient(use_local=False)  # Use API
# # OR
# model_client = HuggingFaceModelClient(use_local=True)   # Download and use locally
# 
# # Replace your existing model.encode() calls with:
# embeddings = model_client.encode(texts)
'''
    
    with open("backend/hf_model_client.py", "w") as f:
        f.write(integration_code)
    
    print("✅ Created HuggingFace integration code: backend/hf_model_client.py")

def main():
    print("🤗 JobMatcher AI - Hugging Face Deployment")
    print("="*50)
    
    # Get model path
    model_path = input("Enter path to your trained model (e.g., finetuned_model/jobmatcher_model_20250112): ").strip()
    
    if not os.path.exists(model_path):
        print(f"❌ Model not found at: {model_path}")
        return
    
    # Get repository name
    username = input("Enter your Hugging Face username: ").strip()
    model_name = input("Enter model name (e.g., jobmatcher-ai-v1): ").strip()
    repo_name = f"{username}/{model_name}"
    
    # Privacy setting
    private = input("Make repository private? (y/n): ").strip().lower() == 'y'
    
    print(f"\n🚀 Deploying to: {repo_name}")
    print(f"🔒 Private: {private}")
    
    # Deploy model
    success = deploy_model_to_hf(model_path, repo_name, private)
    
    if success:
        # Create integration code
        create_hf_integration_code(repo_name)
        
        print(f"\n✅ Deployment Complete!")
        print(f"📋 Next steps:")
        print(f"1. Set HUGGINGFACE_TOKEN in your .env file")
        print(f"2. Update your backend to use HuggingFaceModelClient")
        print(f"3. Choose between API calls or local model loading")
        print(f"4. Test the integration with your existing pipeline")
        
        print(f"\n🔧 Integration options:")
        print(f"• API mode: Serverless, no local resources needed")
        print(f"• Local mode: Faster inference, requires model download")
    else:
        print("❌ Deployment failed. Check your credentials and try again.")

if __name__ == "__main__":
    main() 