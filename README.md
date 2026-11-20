# Diabetes Clinical Note Generation

A diabetes-focused clinical note generation pipeline using hybrid retrieval-augmented generation (RAG) and few-shot in-context learning. Given a patient-clinician dialogue, the pipeline retrieves similar diabetes encounters and relevant clinical guidelines before asking an LLM to produce a structured clinical note.

> **Research software:** This project is intended for research and evaluation only. It is not a medical device and must not be used to diagnose, treat, or make decisions about real patients. Do not send protected health information to third-party model APIs.

## Features

- Similar-case retrieval from a diabetes training set using local TF-IDF embeddings.
- Optional Gemini embeddings for semantic retrieval.
- Clinical guideline retrieval from the bundled diabetes knowledge store.
- LLM note generation through Gemini or Groq-compatible/xAI providers.
- Dry-run mode for inspecting retrieval and prompts without an API call.
- Batch processing of validation or test encounters.
- ROUGE-1, ROUGE-2, and ROUGE-L evaluation against reference notes.
- Markdown reports and JSON results written under `outputs/`.

## Repository layout

```text
LLM_PRO/
├── run_pipeline.py             # Main generation and evaluation entry point
├── evaluate_notes.py           # ROUGE-based evaluator
├── requirements.txt
├── src/
│   ├── case_store.py            # Similar-case retrieval
│   ├── knowledge_store.py      # Guideline retrieval
│   ├── prompt_builder.py       # RAG/few-shot prompt construction
│   ├── generator.py            # Gemini and Groq/xAI generators
│   └── config.py               # Paths and environment configuration
├── dataset/
│   ├── diabetes_cases_train.csv
│   ├── diabetes_cases_val.csv
│   ├── diabetes_cases_test.csv
│   └── knowledge_store/
└── outputs/                    # Generated reports and evaluation results
```

## Setup

```bash
cd LLM_PRO
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Create a `.env` file in `LLM_PRO/` with the provider and API key you want to use. For Gemini:

```dotenv
LLM_PROVIDER=gemini
GEMINI_API_KEY=your_gemini_api_key
GEMINI_MODEL=gemini-2.5-flash
```

For Groq, use a `gsk_` key:

```dotenv
LLM_PROVIDER=groq
GROQ_API_KEY=your_groq_api_key
GROQ_MODEL=qwen/qwen3.8-27b
```

The code also supports xAI-compatible keys through `XAI_API_KEY` or `GROK_API_KEY`; configure the corresponding model as needed. Keep `.env` out of version control and never commit API keys.

## Quick start

Run a small test batch (three encounters by default):

```bash
python run_pipeline.py
```

Preview retrieval and prompt construction without calling an LLM:

```bash
python run_pipeline.py --dry-run --batch-size 1
```

Choose a provider, split, and retrieval size explicitly:

```bash
python run_pipeline.py \
  --provider gemini \
  --split test \
  --batch-size 10 \
  --k 2
```

Process every encounter in a split:

```bash
python run_pipeline.py --split val --all
```

Disable guideline retrieval for an ablation run:

```bash
python run_pipeline.py --dry-run --no-knowledge
```

Use Gemini embeddings instead of the default local TF-IDF retrieval:

```bash
python run_pipeline.py --use-gemini-embeddings
```

Generated encounter reports are saved to `outputs/reports/`; non-dry runs also save batch JSON results in `outputs/`.

## Data

The bundled dataset includes diabetes-specific train, validation, and test splits, ACI-BENCH/MEDIQA-Chat subsets, and a curated clinical guideline store. See [`dataset/README.md`](LLM_PRO/dataset/README.md) for file details, counts, and the dataset loader example.

## Evaluation

The pipeline compares generated notes with reference notes using ROUGE-1, ROUGE-2, and ROUGE-L F1 scores. Evaluation is performed automatically for non-dry runs. Results include per-encounter metrics and batch summaries in the generated JSON output.

## Limitations

- Generated notes can contain omissions or unsupported statements; all output requires qualified human review.
- Retrieval quality depends on the bundled examples and embedding configuration.
- The included metrics measure text overlap and do not establish clinical correctness or safety.
- Provider models, pricing, availability, and API behavior may change independently of this repository.

## Citation and acknowledgements

This project follows the project methodology based on the WangLab MEDIQA-Chat 2023 approach and extends it with diabetes-focused hybrid retrieval of similar cases and clinical guidelines. Refer to the included research and presentation PDFs for project-specific details.
