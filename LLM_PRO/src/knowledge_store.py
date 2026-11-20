import json
import numpy as np
from typing import List, Dict, Any
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from src import config

class KnowledgeStore:
    """
    Manages curated diabetes clinical practice guidelines (ADA & AACE standards).
    Implements Step 4 of the PPT algorithm: Domain knowledge retrieval.
    """
    
    def __init__(self, json_path=None):
        self.json_path = json_path or config.KNOWLEDGE_FILE
        with open(self.json_path, 'r', encoding='utf-8') as f:
            self.guidelines = json.load(f)
            
        # Build text search index
        self.texts = [f"{item['title']}: {item['text']}" for item in self.guidelines]
        self.vectorizer = TfidfVectorizer(stop_words='english')
        self.matrix = self.vectorizer.fit_transform(self.texts)

    def retrieve_guidelines(self, query_dialogue: str, top_k: int = 1) -> List[Dict[str, Any]]:
        """
        Retrieves the most clinically relevant guideline sections for the dialogue.
        Provides background domain knowledge without fabricating patient facts.
        """
        query_vec = self.vectorizer.transform([query_dialogue])
        sims = cosine_similarity(query_vec, self.matrix)[0]
        top_indices = np.argsort(sims)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            item = self.guidelines[idx]
            results.append({
                "id": item["id"],
                "category": item["category"],
                "title": item["title"],
                "text": item["text"],
                "similarity": float(sims[idx])
            })
        return results
