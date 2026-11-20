import os
from typing import List, Dict, Any
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.markdown import Markdown

console = Console()

def print_banner(provider: str, model: str, split: str, batch_size: int, k: int):
    """Displays a clean header banner for the pipeline run."""
    title_text = Text()
    title_text.append("DIABETES CLINICAL NOTE GENERATION PIPELINE\n", style="bold cyan")
    title_text.append(f"Provider: {provider.upper()} ({model})  |  Split: {split.upper()}  |  Batch: {batch_size}  |  Top-K: {k}", style="dim")
    console.print()
    console.print(Panel(title_text, expand=True, border_style="cyan"))

def format_dialogue_text(dialogue: str) -> Text:
    """Formats dialogue text with distinct styling for doctor and patient turns."""
    formatted = Text()
    for line in dialogue.strip().split("\n"):
        line = line.strip()
        if not line:
            continue
        if line.lower().startswith("doctor:") or line.lower().startswith("[doctor]"):
            formatted.append("\n🩺 Doctor: ", style="bold green")
            parts = line.split(":", 1) if ":" in line else line.split("]", 1)
            formatted.append(parts[1].strip() if len(parts) > 1 else line)
        elif line.lower().startswith("patient:") or line.lower().startswith("[patient]"):
            formatted.append("\n👤 Patient: ", style="bold yellow")
            parts = line.split(":", 1) if ":" in line else line.split("]", 1)
            formatted.append(parts[1].strip() if len(parts) > 1 else line)
        else:
            formatted.append(f"\n{line}")
    return formatted

def display_input_conversation(case_id: str, topic: str, dialogue: str):
    """Displays the new input conversation neatly in a panel."""
    content = format_dialogue_text(dialogue)
    panel = Panel(
        content,
        title=f"📥 [bold]INPUT CONVERSATION[/bold] — Encounter: [cyan]{case_id}[/cyan] ({topic})",
        subtitle="Unseen Patient Chart",
        border_style="blue",
        padding=(1, 2)
    )
    console.print(panel)

def display_retrieved_cases(similar_cases: List[Dict[str, Any]]):
    """Displays the top-k retrieved conversations and their clinical notes neatly."""
    k = len(similar_cases)
    console.print(f"\n[bold magenta]🔍 TOP-{k} SIMILAR CASES RETRIEVED FROM DATABASE (CASE STORE)[/bold magenta]")
    
    for i, case in enumerate(similar_cases, 1):
        case_id = case.get("id", f"CASE_{i}")
        topic = case.get("topic", "Diabetes Case")
        sim_pct = case.get("similarity", 0.0) * 100.0
        dialogue = case.get("dialogue", "")
        note = case.get("clinical_note", "")

        table = Table(show_header=True, header_style="bold magenta", expand=True)
        table.add_column(f"1. Past Doctor-Patient Conversation", ratio=1)
        table.add_column(f"2. Doctor's Given Clinical Note", ratio=1)
        
        dlg_formatted = format_dialogue_text(dialogue)
        note_formatted = Text(note.strip())

        table.add_row(dlg_formatted, note_formatted)
        
        case_panel = Panel(
            table,
            title=f"Match #{i}: [bold cyan]{case_id}[/bold cyan] — '{topic}' [green](Similarity: {sim_pct:.1f}%)[/green]",
            border_style="magenta",
            padding=(1, 1)
        )
        console.print(case_panel)

def display_formatted_note(generated_note: str, provider: str, model: str):
    """Displays the AI-formatted clinical note in an elegant medical chart panel."""
    panel = Panel(
        Markdown(generated_note.strip()),
        title=f"📝 [bold]FORMATTED CLINICAL NOTE[/bold] (Generated via {provider.upper()} - {model})",
        subtitle="Synthesized into Standard SOAP Sections",
        border_style="green",
        padding=(1, 2)
    )
    console.print("\n", panel)

def display_validation_comparison(gold_note: str, metrics: Dict[str, float]):
    """Displays the hidden doctor note and the validation ROUGE metrics."""
    # 1. Metrics Table
    table = Table(title="Validation Metrics (Overlap with Doctor's Note)", expand=True)
    table.add_column("Metric", style="cyan", justify="left")
    table.add_column("Score (F1)", style="bold green", justify="center")
    table.add_column("Clinical Significance", style="dim", justify="left")

    table.add_row("ROUGE-1 F1", f"{metrics.get('rouge1_f1', 0):.2f}%", "Overlap of critical medical keywords (A1c, Metformin, etc.)")
    table.add_row("ROUGE-2 F1", f"{metrics.get('rouge2_f1', 0):.2f}%", "Overlap of clinical phrases ('blood glucose', 'foot exam')")
    table.add_row("ROUGE-L F1", f"{metrics.get('rougeL_f1', 0):.2f}%", "Structural and sentence sequence adherence")

    # 2. Hidden Gold Note Preview Panel
    gold_panel = Panel(
        Text(gold_note.strip()),
        title="🎯 [bold]DOCTOR'S ORIGINAL CLINICAL NOTE[/bold] (Hidden Ground Truth)",
        subtitle="Used Exclusively for Validation Comparison",
        border_style="yellow",
        padding=(1, 2)
    )
    
    console.print("\n", gold_panel)
    console.print("\n", table)

def display_batch_summary(summary_metrics: Dict[str, Any]):
    """Displays overall summary table across the batch."""
    table = Table(title="[bold green]FINAL BATCH VALIDATION SUMMARY[/bold green]", expand=True)
    table.add_column("Total Encounters Evaluated", justify="center")
    table.add_column("Mean ROUGE-1 F1", justify="center", style="bold green")
    table.add_column("Mean ROUGE-2 F1", justify="center", style="bold green")
    table.add_column("Mean ROUGE-L F1", justify="center", style="bold green")

    table.add_row(
        str(summary_metrics.get("num_evaluated", 0)),
        f"{summary_metrics.get('mean_rouge1_f1', 0):.2f}%",
        f"{summary_metrics.get('mean_rouge2_f1', 0):.2f}%",
        f"{summary_metrics.get('mean_rougeL_f1', 0):.2f}%"
    )
    console.print("\n", Panel(table, border_style="bold green"))

def save_clean_markdown_report(report_path: str, case: Dict[str, Any], metrics: Dict[str, float], provider: str):
    """Saves a standalone, human-readable markdown file of the run for easy review."""
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    
    md_content = []
    md_content.append(f"# Clinical Encounter Report: {case['encounter_id']}")
    md_content.append(f"**Topic:** {case['topic']}  |  **LLM:** {provider.upper()}\n")
    md_content.append("---\n")
    
    md_content.append("## 1. Input Patient Conversation (New Chart)")
    md_content.append("```text")
    md_content.append(case['dialogue'].strip())
    md_content.append("```\n")
    
    md_content.append("## 2. Top-K Similar Cases Retrieved from Database")
    for i, sc in enumerate(case.get('retrieved_cases', []), 1):
        sim = sc.get('similarity', 0.0) * 100.0
        md_content.append(f"### Match #{i}: {sc.get('id')} — {sc.get('topic')} (Similarity: {sim:.1f}%)")
        if 'dialogue' in sc:
            md_content.append("**Conversation:**")
            md_content.append("```text")
            md_content.append(sc['dialogue'].strip())
            md_content.append("```")
        if 'clinical_note' in sc:
            md_content.append("**Doctor's Clinical Note:**")
            md_content.append("```text")
            md_content.append(sc['clinical_note'].strip())
            md_content.append("```\n")
            
    md_content.append("## 3. Formatted Clinical Note (Synthesized Output)")
    md_content.append(case['generated_note'].strip())
    md_content.append("\n---\n")
    
    md_content.append("## 4. Validation Comparison (Doctor's Original Note)")
    md_content.append("```text")
    md_content.append(case['gold_note'].strip())
    md_content.append("```\n")
    
    md_content.append("## 5. Validation Scores")
    md_content.append(f"- **ROUGE-1 F1:** {metrics.get('rouge1_f1', 0):.2f}%")
    md_content.append(f"- **ROUGE-2 F1:** {metrics.get('rouge2_f1', 0):.2f}%")
    md_content.append(f"- **ROUGE-L F1:** {metrics.get('rougeL_f1', 0):.2f}%\n")
    
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_content))
