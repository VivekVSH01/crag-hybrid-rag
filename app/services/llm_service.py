import google.generativeai as genai
from app.config import get_settings
from loguru import logger


class LLMService:
    def __init__(self):
        self.settings = get_settings()
        genai.configure(api_key=self.settings.gemini_api_key)
        self.model = genai.GenerativeModel(self.settings.llm_model)

    def generate(
        self,
        prompt: str,
        system_prompt: str = "You are a helpful AI assistant.",
        temperature: float = 0.0,
        max_tokens: int = 1000
    ) -> str:
        try:
            response = self.model.generate_content(f"{system_prompt}\n\n{prompt}")
            return response.text.strip()
        except Exception as e:
            logger.error(f"LLM generation error: {e}")
            raise

    def generate_with_json(
        self,
        prompt: str,
        system_prompt: str = "You are a helpful AI assistant.",
        temperature: float = 0.0
    ) -> str:
        try:
            response = self.model.generate_content(f"{system_prompt}\n\nRespond only in JSON.\n\n{prompt}")
            return response.text.strip()
        except Exception as e:
            logger.error(f"LLM JSON generation error: {e}")
            raise
