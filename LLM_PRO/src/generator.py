import time
from typing import Optional
from openai import OpenAI
from src import config

class GrokNoteGenerator:
    """
    Generates structured clinical notes using Groq (GroqCloud gsk_...) or xAI Grok (xai-...).
    Uses OpenAI-compatible endpoint.
    """
    
    def __init__(self, api_key: Optional[str] = None, model_name: Optional[str] = None, base_url: Optional[str] = None):
        self.api_key = api_key or config.GROQ_API_KEY
        self.model_name = model_name or config.GROQ_MODEL
        self.base_url = base_url or config.GROQ_BASE_URL
        
        if not self.api_key or self.api_key.startswith("your_"):
            config.validate_api_key("groq")
            
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url
        )

    def generate_note(self, prompt: str, max_retries: int = 3) -> str:
        """Sends the assembled prompt to Grok and retrieves the formatted clinical note."""
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "You are a Clinical Note Formatter and Synthesizer. "
                                "Your role is to format and synthesize retrieved clinical notes "
                                "and patient chart dialogue into a standard, professional SOAP note."
                            )
                        },
                        {"role": "user", "content": prompt}
                    ],
                    temperature=config.TEMPERATURE,
                    top_p=config.TOP_P,
                    max_tokens=4096
                )
                return response.choices[0].message.content
            except Exception as e:
                if attempt < max_retries - 1:
                    wait_time = 2 ** (attempt + 1)
                    print(f"[Warning] Grok API call failed: {e}. Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    raise RuntimeError(f"Failed to generate clinical note with Grok after {max_retries} attempts: {e}")


class GeminiNoteGenerator:
    """
    Generates structured clinical notes using the Gemini API.
    Uses temperature=0.2 matching clinical precision.
    """
    
    def __init__(self, api_key: Optional[str] = None, model_name: Optional[str] = None):
        self.api_key = api_key or config.GEMINI_API_KEY
        self.model_name = model_name or config.GEMINI_MODEL
        self.client = None
        self._init_client()

    def _init_client(self):
        config.validate_api_key("gemini")
        try:
            from google import genai
            self.client = genai.Client(api_key=self.api_key)
            self._backend = "genai"
        except Exception:
            import google.generativeai as legacy_genai
            legacy_genai.configure(api_key=self.api_key)
            self.client = legacy_genai.GenerativeModel(self.model_name)
            self._backend = "legacy"

    def generate_note(self, prompt: str, max_retries: int = 3) -> str:
        """Sends the assembled prompt to Gemini and retrieves the formatted note."""
        for attempt in range(max_retries):
            try:
                if self._backend == "genai":
                    from google.genai import types
                    response = self.client.models.generate_content(
                        model=self.model_name,
                        contents=prompt,
                        config=types.GenerateContentConfig(
                            temperature=config.TEMPERATURE,
                            top_p=config.TOP_P,
                            max_output_tokens=config.MAX_OUTPUT_TOKENS,
                        )
                    )
                    return response.text
                else:
                    response = self.client.generate_content(
                        prompt,
                        generation_config={
                            "temperature": config.TEMPERATURE,
                            "top_p": config.TOP_P,
                            "max_output_tokens": config.MAX_OUTPUT_TOKENS,
                        }
                    )
                    return response.text
            except Exception as e:
                if attempt < max_retries - 1:
                    wait_time = 2 ** (attempt + 1)
                    print(f"[Warning] Gemini API call failed: {e}. Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    raise RuntimeError(f"Failed to generate clinical note after {max_retries} attempts: {e}")


def get_note_generator(provider: Optional[str] = None, model_name: Optional[str] = None):
    """Factory to instantiate the appropriate generator based on configuration or flag."""
    p = (provider or config.LLM_PROVIDER).lower()
    if p in ["groq", "grok", "xai"]:
        return GrokNoteGenerator(model_name=model_name or config.GROK_MODEL)
    elif p in ["gemini", "google"]:
        return GeminiNoteGenerator(model_name=model_name or config.GEMINI_MODEL)
    else:
        raise ValueError(f"Unknown LLM provider '{provider}'. Supported providers are: 'groq', 'gemini'.")
