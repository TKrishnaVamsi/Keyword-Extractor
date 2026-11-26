
import os
from pathlib import Path
from sentence_transformers import SentenceTransformer

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
CACHE_DIR = Path(__file__).resolve().parent / "model_cache"

os.environ["SENTENCE_TRANSFORMERS_HOME"] = str(CACHE_DIR)

print("Downloading embedding model...", flush=True)

model = SentenceTransformer(
    MODEL_NAME,
    cache_folder=str(CACHE_DIR)
)

print("Embedding model downloaded successfully.", flush=True)

embedding = model.encode(
    ["Keyword extraction deployment test"],
    normalize_embeddings=True
)

print("Embedding shape:", embedding.shape)
print("DOWNLOAD AND VERIFICATION SUCCESSFUL")
