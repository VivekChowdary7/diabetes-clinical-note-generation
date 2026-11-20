#!/usr/bin/env python3
"""
Diabetes-Focused Clinical Note Generation via Few-Shot In-Context Learning
Main Execution Script (Supports Grok and Gemini API)

Adheres directly to:
- Reference Research Paper: WangLab at MEDIQA-Chat 2023 (ACL-ClinicalNLP)
- Proposed System: Diabetes-Focused Hybrid RAG Architecture (Vivek Chowdary & Jyothesh Kumar)
"""

import os
import sys
import json
import argparse
import pandas as pd
from datetime import datetime
from src import config
from src.case_store import CaseStore
from src.knowledge_store import KnowledgeStore
from src.prompt_builder import PromptBuilder
from src.generator import get_note_generator
from evaluate_notes import ClinicalNoteEvaluator
from src.display import (
    print_banner,
    display_input_conversation,
    display_retrieved_cases,
    display_formatted_note,
    display_validation_comparison,
    display_batch_summary,
    save_clean_markdown_report,
    console
)

def parse_args():
    parser = argparse.ArgumentParser(description="Run Few-Shot ICL Clinical Note Generation on Diabetes Consultations")
    parser.add_argument("--provider", type=str, default=None, choices=["groq", "grok", "gemini", "xai"], help="LLM Provider: groq or gemini (default from .env)")
    parser.add_argument("--model", type=str, default=None, help="Model name override (e.g. grok-2-latest or gemini-2.5-flash)")
    parser.add_argument("--batch-size", type=int, default=3, help="Number of encounters to process (default: 3)")
    parser.add_argument("--all", action="store_true", help="Process all cases in the selected split")
    parser.add_argument("--split", type=str, default="test", choices=["test", "val", "validation"], help="Dataset split to evaluate on (test or val)")
    parser.add_argument("--k", type=int, default=2, help="Number of in-context example cases to retrieve (default: 2)")
    parser.add_argument("--dry-run", action="store_true", help="Build prompts and retrieval without calling the LLM API")
    parser.add_argument("--no-knowledge", action="store_true", help="Ablation: Disable clinical knowledge guideline retrieval")
    parser.add_argument("--use-gemini-embeddings", action="store_true", help="Use Gemini API for embeddings instead of fast local TF-IDF")
    return parser.parse_args()

def main():
    args = parse_args()
    provider = (args.provider or config.LLM_PROVIDER).lower()
    if provider in ["grok", "xai"]:
        # If user key is gsk_, route to Groq; if xai-, route to xAI
        provider = "groq" if config.GROQ_API_KEY.startswith("gsk_") else "xai"

    model_name = args.model or (config.GROQ_MODEL if provider in ["groq", "xai"] else config.GEMINI_MODEL)

    # 1. Print visual banner
    print_banner(
        provider=provider,
        model=model_name,
        split=args.split,
        batch_size=args.batch_size if not args.all else "ALL",
        k=args.k
    )

    # 2. Check API Key
    if provider == "grok":
        has_api_key = bool(config.GROK_API_KEY and config.GROK_API_KEY != "your_xai_api_key_here")
    else:
        has_api_key = bool(config.GEMINI_API_KEY and config.GEMINI_API_KEY != "your_gemini_api_key_here")

    if not has_api_key and not args.dry_run:
        console.print(f"\n[bold red][Notice][/bold red] No valid API key found for [cyan]{provider.upper()}[/cyan] in [bold].env[/bold].")
        if provider == "grok":
            console.print("Please open [bold].env[/bold] and paste your Grok key:")
            console.print("  [green]GROK_API_KEY=xai-...[/green]\n")
        else:
            console.print("Please open [bold].env[/bold] and paste your Gemini key:")
            console.print("  [green]GEMINI_API_KEY=AIzaSy...[/green]\n")
        console.print("Running in [yellow]--dry-run[/yellow] mode so you can preview the input chart and top-K retrieved conversations.\n")
        args.dry_run = True

    # 3. Initialize Case Store and Knowledge Store
    console.print(f"[dim]• Initializing Case Store from {config.TRAIN_FILE.name}...[/dim]")
    case_store = CaseStore(data_path=config.TRAIN_FILE, use_gemini=args.use_gemini_embeddings and has_api_key)
    console.print(f"[dim]• Loaded {len(case_store.df)} exemplar cases into Case Store.[/dim]")

    kb_store = KnowledgeStore(json_path=config.KNOWLEDGE_FILE)
    console.print(f"[dim]• Loaded {len(kb_store.guidelines)} clinical practice guideline topics.[/dim]")

    # 4. Load Evaluation Split (Test or Validation)
    eval_file = config.VAL_FILE if args.split in ["val", "validation"] else config.TEST_FILE
    df_eval = pd.read_csv(eval_file)
    total_eval = len(df_eval)
    num_to_run = total_eval if args.all else min(args.batch_size, total_eval)
    eval_subset = df_eval.iloc[:num_to_run]
    console.print(f"[dim]• Loaded {total_eval} encounters from {eval_file.name}. Running batch: {num_to_run}.[/dim]\n")

    # 5. Initialize Generator
    generator = None
    if not args.dry_run:
        try:
            generator = get_note_generator(provider=provider, model_name=model_name)
        except Exception as e:
            console.print(f"[bold red]Failed to initialize {provider.upper()} generator:[/bold red] {e}")
            sys.exit(1)

    evaluator = ClinicalNoteEvaluator()
    results = []
    reports_dir = config.ROOT_DIR / "outputs" / "reports"

    # 6. Process Encounters
    for i, (_, row) in enumerate(eval_subset.iterrows(), 1):
        case_id = row.get("id", f"ENCOUNTER_{i}")
        topic = row.get("topic", "Diabetes Visit")
        dialogue = row["dialogue"]
        gold_note = row["clinical_note"]

        console.rule(f"[bold yellow]ENCOUNTER {i} of {num_to_run} — {case_id}[/bold yellow]")

        # 6.1 Display Input Conversation (New Chart)
        display_input_conversation(case_id=case_id, topic=topic, dialogue=dialogue)

        # 6.2 Retrieve Top-K Similar Cases (Both past chats & past notes)
        similar_cases = case_store.retrieve_similar_notes(dialogue, top_k=args.k)
        display_retrieved_cases(similar_cases)

        # 6.3 Clinical Guideline Retrieval
        guidelines = None
        if not args.no_knowledge:
            guidelines = kb_store.retrieve_guidelines(dialogue, top_k=1)

        # 6.4 Build Few-Shot Prompt
        prompt = PromptBuilder.build_prompt(
            dialogue=dialogue,
            similar_cases=similar_cases,
            clinical_guidelines=guidelines,
            include_knowledge=not args.no_knowledge
        )

        # 6.5 Generate / Format Clinical Note
        generated_note = ""
        if not args.dry_run and generator:
            with console.status(f"[bold green]Formatting clinical note with {provider.upper()} ({model_name})...[/bold green]"):
                start_time = datetime.now()
                generated_note = generator.generate_note(prompt)
                duration = (datetime.now() - start_time).total_seconds()
            display_formatted_note(generated_note, provider=provider, model=model_name)
        else:
            generated_note = "[DRY RUN PREVIEW - Note generation skipped]"
            console.print(f"\n[yellow]⚠️ [Dry-run][/yellow] Assembled prompt contains {len(prompt)} characters. Live API call skipped.")

        # 6.6 Validation: Compare with Hidden Gold Note
        metrics = {}
        if not args.dry_run:
            metrics = evaluator.evaluate_pair(gold_note, generated_note)
            display_validation_comparison(gold_note, metrics)

        record = {
            "encounter_id": case_id,
            "topic": topic,
            "dialogue": dialogue,
            "gold_note": gold_note,
            "generated_note": generated_note,
            "retrieved_cases": similar_cases,
            "metrics": metrics
        }
        results.append(record)

        # Save formatted standalone markdown report for this encounter
        report_file = reports_dir / f"encounter_{case_id}.md"
        save_clean_markdown_report(str(report_file), record, metrics, provider=provider)
        console.print(f"[dim]📄 Saved full report to: {report_file}[/dim]\n")

    # 7. Summary
    if not args.dry_run and results:
        summary_metrics = evaluator.evaluate_batch(results)
        display_batch_summary(summary_metrics)

        # Save batch JSON
        output_json = config.ROOT_DIR / "outputs" / f"results_{provider}_batch_{num_to_run}.json"
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump({"summary_metrics": summary_metrics, "cases": results}, f, indent=2)
        console.print(f"\n[bold green]✓ Complete batch data saved to:[/bold green] {output_json}")

if __name__ == "__main__":
    main()
