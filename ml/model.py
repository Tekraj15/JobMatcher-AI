from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F

MODEL_NAME = "intfloat/e5-base-v2"

# Load model and tokenizer globally
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)

def embed_text(text: str) -> torch.Tensor:
    """
    Encodes a single string and returns a normalized embedding tensor.
    """
    input_text = f"passage: {text.strip()}"
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
        embeddings = outputs.last_hidden_state[:, 0, :]  # CLS token
        return F.normalize(embeddings, p=2, dim=1).squeeze().numpy()

def embed_batch(texts: list[str]) -> list:
    """
    Encodes a list of strings and returns normalized vectors.
    """
    batch_inputs = [f"passage: {t.strip()}" for t in texts]
    inputs = tokenizer(batch_inputs, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
        embeddings = outputs.last_hidden_state[:, 0, :]
        return F.normalize(embeddings, p=2, dim=1).numpy()
