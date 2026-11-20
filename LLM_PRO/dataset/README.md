# Diabetes-Focused Clinical Note Generation: Dataset Suite

This dataset package is specifically curated and preprocessed for the project:
**"Diabetes-Focused Clinical Note Generation Using Hybrid Retrieval-Augmented Generation with a Pretrained LLM"**
(Vivek Chowdary & Jyothesh Kumar, CISMR) based on the **WangLab at MEDIQA-Chat 2023** methodology.

---

## 📁 Directory Structure & File Summary

```
dataset/
├── diabetes_clinical_cases_large.csv     # Primary dataset: 865 full diabetes encounters
├── diabetes_clinical_cases_large.json    # JSON format of primary dataset
├── diabetes_cases_train.csv             # Train split (692 encounters) - used for ICL Case Store
├── diabetes_cases_val.csv               # Validation split (86 encounters) - tuning & prompt eval
├── diabetes_cases_test.csv              # Test split (87 encounters) - unseen test evaluation
│
├── aci_bench_diabetes_subset.csv        # ACI-BENCH / MEDIQA-Chat 2023 Task B diabetes encounters (39 cases)
├── aci_bench_diabetes_subset.json       # JSON format of ACI-BENCH diabetes subset
├── aci_bench_all_207_encounters.csv     # Full official MEDIQA-Chat 2023 Task B dataset (207 cases)
│
├── aci_bench_endocrinology_eval_diabetes.csv # ACI-BENCH formatted endocrine/diabetes evaluation set (50 cases)
├── aci_bench_endocrinology_eval_all140.csv   # Full endocrine ACI-BENCH evaluation set (140 cases)
│
├── knowledge_store/                     # Hybrid RAG Knowledge Store (Pillar B of Presentation)
│   ├── diabetes_clinical_guidelines.json # Curated clinical guidelines (ADA / AACE standards)
│   └── diabetes_clinical_guidelines.csv  # CSV format of clinical guidelines
│
└── load_dataset.py                      # Ready-to-run Python loader & Hybrid RAG prompt builder
```

---

## 🎯 Dataset Alignment with the Project Architecture

As defined in the project presentation (**Slide 9 & 10: Proposed Diabetes-Focused RAG Architecture**):

### 1. Similar Diabetes Case Store (Paper-Inspired ICL)
- **Source Files**: `diabetes_cases_train.csv` (692 cases) & `aci_bench_diabetes_subset.csv` (39 cases)
- **Content**: Full doctor-patient dialogues mapped to detailed, structured clinical notes (HPI, Physical Exam, Results, Assessment & Plan).
- **Usage**: When a new patient dialogue arrives, compute cosine similarity between the input dialogue and stored training dialogues to retrieve top-$k$ reference notes for in-context few-shot guidance.

### 2. Diabetes Knowledge Store (Project Extension)
- **Source Files**: `knowledge_store/diabetes_clinical_guidelines.json`
- **Content**: 8 core clinical topics including Diagnostic Criteria (FPG, OGTT, HbA1c), Glycemic Targets, First-line & Combination Pharmacotherapy (Metformin, SGLT2i, GLP-1 RA), Basal-Bolus Insulin Management, Acute Complications (Hypoglycemia Rule of 15, DKA, HHS), Chronic Complication Screenings (Nephropathy, Retinopathy, Neuropathy, Foot Exam), Medical Nutrition Therapy, and Documentation Standards.
- **Usage**: Provides domain context injected into the prompt without contaminating dialogue-specific patient facts.

### 3. Evaluation & Test Benchmarks
- **Unseen Test Set**: `diabetes_cases_test.csv` (87 unseen diabetes encounters)
- **Official Benchmark**: `aci_bench_diabetes_subset.csv` & `aci_bench_endocrinology_eval_diabetes.csv`
- **Metrics**: Ready for evaluation using ROUGE-1 F1, BERTScore F1, BLEURT-20 (exact metrics from WangLab MEDIQA-Chat 2023).

---

## 🚀 Quickstart Example

Run the turnkey helper script:
```bash
python3 dataset/load_dataset.py
```

Or in your own Python script / notebook:
```python
from dataset.load_dataset import DiabetesDatasetLoader, HybridDiabetesRAG

# Initialize the Hybrid RAG pipeline
rag = HybridDiabetesRAG()

# Load test dialogue
test_df = rag.loader.load_case_store('test')
query_dialogue = test_df.iloc[0]['dialogue']

# Retrieve similar cases and clinical knowledge
similar_cases = rag.retrieve_similar_cases(query_dialogue, top_k=2)
clinical_guidelines = rag.retrieve_knowledge(query_dialogue, top_k=2)

# Construct the full prompt ready for LLM inference (GPT-4, Gemini, Claude, LLaMA)
prompt = rag.build_prompt(query_dialogue, k_cases=2, k_knowledge=2)
print(prompt)
```
