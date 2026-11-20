from typing import List, Dict, Any
from rouge_score import rouge_scorer
import numpy as np

class ClinicalNoteEvaluator:
    """Evaluates generated notes against gold reference notes using standard NLG metrics."""
    
    def __init__(self):
        self.scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

    def evaluate_pair(self, reference_note: str, generated_note: str) -> Dict[str, float]:
        """Calculates ROUGE-1, ROUGE-2, and ROUGE-L F1 scores for a note pair."""
        scores = self.scorer.score(reference_note, generated_note)
        return {
            "rouge1_f1": scores['rouge1'].fmeasure * 100.0,
            "rouge2_f1": scores['rouge2'].fmeasure * 100.0,
            "rougeL_f1": scores['rougeL'].fmeasure * 100.0,
        }

    def evaluate_batch(self, test_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculates average metrics across a batch of test results."""
        r1_list, r2_list, rL_list = [], [], []
        
        for item in test_results:
            ref = item.get("gold_note", "")
            gen = item.get("generated_note", "")
            if ref and gen:
                scores = self.evaluate_pair(ref, gen)
                item["metrics"] = scores
                r1_list.append(scores["rouge1_f1"])
                r2_list.append(scores["rouge2_f1"])
                rL_list.append(scores["rougeL_f1"])
                
        summary = {
            "num_evaluated": len(r1_list),
            "mean_rouge1_f1": float(np.mean(r1_list)) if r1_list else 0.0,
            "mean_rouge2_f1": float(np.mean(r2_list)) if r2_list else 0.0,
            "mean_rougeL_f1": float(np.mean(rL_list)) if rL_list else 0.0,
        }
        return summary
