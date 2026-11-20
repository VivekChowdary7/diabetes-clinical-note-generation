from typing import List, Dict, Any

class PromptBuilder:
    """
    Constructs the Few-Shot In-Context Learning prompt.
    Directly adheres to WangLab at MEDIQA-Chat 2023 Figure 4 and PPT Slides 12-14.
    """
    
    SYSTEM_INSTRUCTION = (
        "You are a Clinical Note Formatter and Synthesizer.\n"
        "Your task is strictly FORMATTING AND STRUCTURING: You are given a new doctor-patient chart/conversation and the top-K clinical notes retrieved from the most similar patient cases in the database.\n\n"
        "ROLE AND CONSTRAINTS:\n"
        "1. DO NOT GENERATE OUTSIDE MEDICAL FACTS: Do not invent diagnoses, medications, or lab values using external medical training. All clinical facts and structure must be directly synthesized from the provided top-K retrieved clinical notes and the patient's chart.\n"
        "2. FORMATTING ROLE: Your sole role is to take the clinical information from the retrieved notes, align it with the new patient's chart, and format it into the required standard medical note structure.\n"
        "3. REQUIRED OUTPUT STRUCTURE:\n"
        "   - SUBJECTIVE / HISTORY OF PRESENT ILLNESS (Chief Complaint, HPI, duration, symptoms, medications discussed)\n"
        "   - OBJECTIVE EXAM (Vitals, physical examination findings)\n"
        "   - OBJECTIVE RESULTS (Blood glucose logs, HbA1c, renal/lipid labs)\n"
        "   - ASSESSMENT AND PLAN (Assessment of glycemic control, medication modifications, follow-up, lifestyle guidance)\n"
        "4. Output strictly the formatted clinical note without conversational filler."
    )

    @classmethod
    def build_prompt(
        cls,
        dialogue: str,
        similar_cases: List[Dict[str, Any]] = None,
        clinical_guidelines: List[Dict[str, Any]] = None,
        include_knowledge: bool = True
    ) -> str:
        prompt_parts = []
        
        # 1. Base Instructions
        prompt_parts.append(cls.SYSTEM_INSTRUCTION)
        prompt_parts.append("\n" + "=" * 50)
        
        # 2. Domain Knowledge (Project Extension - PPT Slide 12)
        if include_knowledge and clinical_guidelines:
            prompt_parts.append("\n### CLINICAL PRACTICE GUIDELINE REFERENCE (DOMAIN CONTEXT):")
            for kb in clinical_guidelines:
                prompt_parts.append(f"[{kb['title']}]\n{kb['text']}\n")
            prompt_parts.append("=" * 50)

        # 3. Retrieved Top-K Clinical Notes from Case Store
        if similar_cases:
            prompt_parts.append("\n### RETRIEVED TOP-K CLINICAL NOTES FROM SIMILAR CASES IN DATASET:")
            for i, case in enumerate(similar_cases, 1):
                prompt_parts.append(f"\n--- RETRIEVED CLINICAL NOTE {i} (From Similar Encounter: '{case['topic']}', Similarity: {case['similarity']:.2f}) ---")
                prompt_parts.append(f"[DOCTOR'S CLINICAL NOTE CONTENT]:\n{case['clinical_note'].strip()}")
                prompt_parts.append("\n" + "-" * 40)
            prompt_parts.append("=" * 50)

        # 4. Input New Chart
        prompt_parts.append("\n### NEW DOCTOR-PATIENT CHART (CURRENT CONVERSATION):")
        prompt_parts.append(dialogue.strip())
        
        # 5. Output trigger
        prompt_parts.append("\n" + "=" * 50)
        prompt_parts.append("### FORMATTED CLINICAL NOTE (SYNTHESIZED FROM RETRIEVED NOTES & CHART):")
        
        return "\n".join(prompt_parts)
