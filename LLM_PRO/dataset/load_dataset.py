#!/usr/bin/env python3
"""
Diabetes Clinical Note Generation Dataset Loader & Hybrid RAG Helper
Based on:
- WangLab at MEDIQA-Chat 2023 (ACL-ClinicalNLP 2023)
- Diabetes-Focused Clinical Note Generation with Hybrid RAG (Vivek Chowdary & Jyothesh Kumar)
"""

import os
import json
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

DATASET_DIR = os.path.dirname(os.path.abspath(__file__))

class DiabetesDatasetLoader:
    def __init__(self, base_dir=DATASET_DIR):
        self.base_dir = base_dir
        self.train_file = os.path.join(base_dir, 'diabetes_cases_train.csv')
        self.val_file = os.path.join(base_dir, 'diabetes_cases_val.csv')
        self.test_file = os.path.join(base_dir, 'diabetes_cases_test.csv')
        self.aci_bench_file = os.path.join(base_dir, 'aci_bench_diabetes_subset.csv')
        self.knowledge_file = os.path.join(base_dir, 'knowledge_store', 'diabetes_clinical_guidelines.json')

    def load_case_store(self, split='all'):
        """Load paired doctor-patient dialogue and clinical notes"""
        if split == 'train':
            return pd.read_csv(self.train_file)
        elif split == 'validation' or split == 'val':
            return pd.read_csv(self.val_file)
        elif split == 'test':
            return pd.read_csv(self.test_file)
        elif split == 'aci_bench':
            return pd.read_csv(self.aci_bench_file)
        else:
            return pd.read_csv(os.path.join(self.base_dir, 'diabetes_clinical_cases_large.csv'))

    def load_knowledge_store(self):
        """Load diabetes clinical guidelines for domain context"""
        with open(self.knowledge_file, 'r', encoding='utf-8') as f:
            return json.load(f)


class HybridDiabetesRAG:
    """Implements the Hybrid RAG pipeline proposed in the PLS Presentation"""
    def __init__(self, loader=None):
        if loader is None:
            loader = DiabetesDatasetLoader()
        self.loader = loader
        
        # Load case store (train split for ICL in-context examples)
        self.train_cases = self.loader.load_case_store('train')
        self.knowledge_store = self.loader.load_knowledge_store()
        
        # Vectorizer for case store retrieval
        self.case_vectorizer = TfidfVectorizer(stop_words='english')
        self.case_vectors = self.case_vectorizer.fit_transform(self.train_cases['dialogue'])
        
        # Vectorizer for knowledge store retrieval
        self.kb_vectorizer = TfidfVectorizer(stop_words='english')
        kb_texts = [f"{item['title']}: {item['text']}" for item in self.knowledge_store]
        self.kb_vectors = self.kb_vectorizer.fit_transform(kb_texts)

    def retrieve_similar_cases(self, input_dialogue, top_k=3):
        """Step 3 of Proposed Algorithm: Retrieve top-k most similar cases"""
        query_vec = self.case_vectorizer.transform([input_dialogue])
        similarities = cosine_similarity(query_vec, self.case_vectors)[0]
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            results.append({
                'id': self.train_cases.iloc[idx]['id'],
                'topic': self.train_cases.iloc[idx]['topic'],
                'similarity_score': float(similarities[idx]),
                'dialogue': self.train_cases.iloc[idx]['dialogue'],
                'clinical_note': self.train_cases.iloc[idx]['clinical_note']
            })
        return results

    def retrieve_knowledge(self, input_dialogue, top_k=2):
        """Step 4 of Proposed Algorithm: Retrieve relevant diabetes knowledge"""
        query_vec = self.kb_vectorizer.transform([input_dialogue])
        similarities = cosine_similarity(query_vec, self.kb_vectors)[0]
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            item = self.knowledge_store[idx]
            results.append({
                'category': item['category'],
                'title': item['title'],
                'similarity_score': float(similarities[idx]),
                'text': item['text']
            })
        return results

    def build_prompt(self, input_dialogue, k_cases=2, k_knowledge=2):
        """Step 5 of Proposed Algorithm: Construct hybrid prompt"""
        similar_cases = self.retrieve_similar_cases(input_dialogue, top_k=k_cases)
        relevant_kb = self.retrieve_knowledge(input_dialogue, top_k=k_knowledge)

        prompt_parts = []
        
        # Instructions
        prompt_parts.append("### INSTRUCTIONS:")
        prompt_parts.append(
            "You are an expert clinical documentation assistant specializing in endocrinology and diabetes care.\n"
            "Given the doctor-patient dialogue, generate a comprehensive, structured diabetes clinical note.\n"
            "Ground all patient-specific facts exclusively in the dialogue.\n"
            "Use the provided reference cases for formatting and style, and the diabetes guidelines for clinical context.\n"
        )

        # Knowledge Context (Extension)
        if relevant_kb:
            prompt_parts.append("### CLINICAL KNOWLEDGE CONTEXT:")
            for kb in relevant_kb:
                prompt_parts.append(f"[{kb['title']}]\n{kb['text']}\n")

        # In-Context Examples (WangLab approach)
        if similar_cases:
            prompt_parts.append("### IN-CONTEXT REFERENCE CASES (FOR FORMATTING & STYLE GUIDANCE):")
            for i, case in enumerate(similar_cases, 1):
                prompt_parts.append(f"--- EXAMPLE NOTE {i} (Topic: {case['topic']}) ---")
                prompt_parts.append(case['clinical_note'].strip())
                prompt_parts.append("\n")

        # Test Input
        prompt_parts.append("### PATIENT-DOCTOR DIALOGUE:")
        prompt_parts.append(input_dialogue.strip())
        prompt_parts.append("\n### STRUCTURED DIABETES CLINICAL NOTE:")

        return "\n".join(prompt_parts)


if __name__ == '__main__':
    print("Testing DiabetesDatasetLoader and HybridDiabetesRAG...")
    rag = HybridDiabetesRAG()
    test_cases = rag.loader.load_case_store('test')
    print(f"Loaded {len(test_cases)} test cases.")
    
    sample_dialogue = test_cases.iloc[0]['dialogue']
    sample_topic = test_cases.iloc[0]['topic']
    print(f"\nSample Dialogue Topic: {sample_topic}")
    print(f"Sample Dialogue Preview: {sample_dialogue[:200]}...\n")
    
    print("Retrieving top-2 similar cases...")
    sim_cases = rag.retrieve_similar_cases(sample_dialogue, top_k=2)
    for c in sim_cases:
        print(f" - Retreived case ID: {c['id']}, Topic: {c['topic']}, Similarity: {c['similarity_score']:.4f}")
        
    print("\nRetrieving relevant diabetes clinical knowledge...")
    kb_items = rag.retrieve_knowledge(sample_dialogue, top_k=2)
    for kb in kb_items:
        print(f" - Retreived Guideline: {kb['title']} (Score: {kb['similarity_score']:.4f})")
        
    prompt = rag.build_prompt(sample_dialogue, k_cases=1, k_knowledge=1)
    print(f"\nGenerated Full Hybrid RAG Prompt (Length: {len(prompt)} chars). Preview:")
    print("-" * 50)
    print(prompt[:600] + "\n... [TRUNCATED] ...")
    print("-" * 50)
    print("Verification complete: All datasets and loader pipeline functional!")
