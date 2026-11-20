import os
import pandas as pd
import numpy as np
from typing import List, Dict, Any
from sklearn.metrics.pairwise import cosine_similarity
from src import config
from src.embeddings import GeminiEmbedder, TFIDFEmbedder

class CaseStore:
    """
    Manages the pool of training doctor-patient dialogues and corresponding gold notes.
    Implements Step 3 of the PPT algorithm: Top-k similar-case retrieval using cosine similarity.
    """
    
    def __init__(self, data_path=None, use_gemini: bool = True):
        self.data_path = data_path or config.TRAIN_FILE
        self.df = pd.read_csv(self.data_path)
        self.use_gemini = use_gemini
        self.embeddings = None
        self.embedder = None
        
        self._initialize_store()

    def _initialize_store(self):
        """Loads or computes embeddings for the case store containing both dialogue and clinical note."""
        # Bundle both the doctor-patient dialogue and the corresponding clinical note
        combined_texts = [
            f"Doctor-Patient Dialogue:\n{row['dialogue']}\n\nClinical Note:\n{row['clinical_note']}"
            for _, row in self.df.iterrows()
        ]

        if self.use_gemini and config.GEMINI_API_KEY and config.GEMINI_API_KEY != "your_gemini_api_key_here":
            self.embedder = GeminiEmbedder()
            if os.path.exists(config.TRAIN_EMBEDDINGS_CACHE):
                print(f"[CaseStore] Loading cached Gemini embeddings from {config.TRAIN_EMBEDDINGS_CACHE.name}...")
                self.embeddings = np.load(config.TRAIN_EMBEDDINGS_CACHE)
            else:
                print(f"[CaseStore] Generating Gemini embeddings for {len(self.df)} training cases (dialogue + notes)...")
                self.embeddings = self.embedder.embed_batch(
                    combined_texts,
                    batch_size=50
                )
                np.save(config.TRAIN_EMBEDDINGS_CACHE, self.embeddings)
                print(f"[CaseStore] Saved embeddings to {config.TRAIN_EMBEDDINGS_CACHE}")
        else:
            print("[CaseStore] Indexing training cases (dialogue + clinical note pairs)...")
            self.embedder = TFIDFEmbedder()
            self.embedder.fit(combined_texts)
            self.embeddings = self.embedder.embed_batch(combined_texts)

    def retrieve_similar_notes(self, query_dialogue: str, top_k: int = 2) -> List[Dict[str, Any]]:
        """
        Retrieves top-k most similar cases based on cosine similarity to the query dialogue.
        As proven in WangLab at MEDIQA-Chat 2023 Table 2, providing 'Notes only' as
        in-context examples outperforms dialogue+note pairs.
        """
        query_vec = self.embedder.embed_text(query_dialogue)
        if query_vec.ndim == 1:
            query_vec = query_vec.reshape(1, -1)
            
        sims = cosine_similarity(query_vec, self.embeddings)[0]
        top_indices = np.argsort(sims)[::-1][:top_k]
        
        results = []
        for rank, idx in enumerate(top_indices, 1):
            row = self.df.iloc[idx]
            results.append({
                "rank": rank,
                "id": row.get("id", f"TRAIN_{idx}"),
                "topic": row.get("topic", "Diabetes Encounter"),
                "similarity": float(sims[idx]),
                "clinical_note": row["clinical_note"],
                "dialogue": row["dialogue"]
            })
        return results
