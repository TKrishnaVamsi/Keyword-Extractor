"""MLflow pyfunc wrapper so the extractor can be versioned in the MLflow Model Registry."""
import json

import mlflow.pyfunc

from src.extractor import extract_keywords, get_encoder

DEFAULTS = {
    "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
    "top_n": 5,
    "ngram_range": [1, 2],
    "diversity": 0.5,
}


class KeywordExtractorModel(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        with open(context.artifacts["config"]) as f:
            self.cfg = {**DEFAULTS, **json.load(f)}
        self.encoder = get_encoder(self.cfg["embedding_model"])

    def predict(self, context, model_input, params=None):
        results = []
        for row in model_input.to_dict("records"):
            results.append(
                extract_keywords(
                    row["text"],
                    self.encoder,
                    top_n=int(row.get("top_n", self.cfg["top_n"])),
                    ngram_range=tuple(self.cfg["ngram_range"]),
                    diversity=float(self.cfg["diversity"]),
                )
            )
        return results


def save_config(cfg, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(cfg, f, indent=2)
    return path
