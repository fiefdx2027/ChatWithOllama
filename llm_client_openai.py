# llm_client_openai.py
# Generic OpenAI‑compatible client.  Works with Ollama, Llama.cpp, and any OpenAI‑style API.

from openai import OpenAI
from .llm_client import LLMClient

class OpenAIClient(LLMClient):
    """
    Use the official OpenAI client, pointing at `base_url`.
    The wrapper keeps the same chat signature: `chat(prompt, history)`.
    """
    def __init__(self, model: str, base_url: str):
        self.client = OpenAI(base_url=base_url)
        self.model = model

    def chat(self, prompt: str, history: list[dict]) -> str:
        # Build the messages array expected by the OpenAI spec.
        # The history is a list of dicts with keys "role" and "content".
        messages = list(history)  # shallow copy
        messages.append({"role": "user", "content": prompt})
        try:
            resp = self.client.chat(model=self.model, messages=messages, temperature=0.7, max_tokens=1024)
            # The response object has `choices[0].message.content`.
            return resp.choices[0].message.content
        except Exception as e:
            raise RuntimeError(f"OpenAIClient error: {e}")
