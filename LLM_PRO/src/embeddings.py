import warnings
warnings.filterwarnings("ignore")

import numpy as np
from typing import List, Union
from sklearn.feature_extraction.text import TfidfVectorizer
from src import config

class GeminiEmbedder:
    """Handles text embeddings via Gemini API."""
    
    def __init__(self, api_key: str = None, model_name: str = None):
        self.api_key = api_key or config.GEMINI_API_KEY
        self.model_name = model_name or config.GEMINI_EMBEDDING_MODEL
        if self.api_key and self.api_key != "your_gemini_api_key_here":
            from google import genai
            self.client = genai.Client(api_key=self.api_key)
            self._has_valid_key = True
        else:
            self._has_valid_key = False

    def embed_text(self, text: str) -> np.ndarray:
        """Embed a single text string."""
        if not self._has_valid_key:
            raise ValueError("GEMINI_API_KEY is not configured in .env.")
        
        result = self.client.models.embed_content(
            model=self.model_name,
            contents=text
        )
        return np.array(result.embeddings[0].values, dtype=np.float32)

    def embed_batch(self, texts: List[str], batch_size: int = 50) -> np.ndarray:
        """Embed a batch of text strings with chunking."""
        if not self._has_valid_key:
            raise ValueError("GEMINI_API_KEY is not configured in .env.")
        
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            chunk = texts[i:i + batch_size]
            result = self.client.models.embed_content(
                model=self.model_name,
                contents=chunk
            )
            for emb in result.embeddings:
                all_embeddings.append(emb.values)
        return np.array(all_embeddings, dtype=np.float32)


class TFIDFEmbedder:
    """Local fallback embedder using TF-IDF for offline testing."""
    
    def __init__(self):
        self.vectorizer = TfidfVectorizer(stop_words='english', max_features=10000)
        self._fitted = False

    def fit(self, texts: List[str]):
        self.vectorizer.fit(texts)
        self._fitted = True

    def embed_batch(self, texts: List[str]) -> np.ndarray:
        if not self._fitted:
            self.fit(texts)
        return self.vectorizer.transform(texts).toarray().astype(np.float32)

    def embed_text(self, text: str) -> np.ndarray:
        return self.vectorizer.transform([text]).toarray()[0].astype(np.float32)
