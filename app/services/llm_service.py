from groq import Groq
from app.config import get_settings
from loguru import logger


class LLMService:
    def __init__(self):
        self.settings = get_settings()
        self.client = Groq(api_key=self.settings.groq_api_key)
        self.model = self.settings.llm_model

    def generate(
        self,
        prompt: str,
        system_prompt: str = "You are a helpful AI assistant.",
        temperature: float = 0.0,
        max_tokens: int = 1000
    ) -> str:
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                temperature=temperature,
                max_tokens=max_tokens
            )
            return response.choices[0].message.content.strip()
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
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                temperature=temperature,
                response_format={"type": "json_object"}
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"LLM JSON generation error: {e}")
            raise
