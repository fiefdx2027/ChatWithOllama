# llm_client.py
from abc import ABC, abstractmethod

class LLMClient(ABC):
    """Abstract base for all LLM back‑ends."""
    @abstractmethod
    def chat(self, prompt: str, history: list[dict]) -> str:
        ...