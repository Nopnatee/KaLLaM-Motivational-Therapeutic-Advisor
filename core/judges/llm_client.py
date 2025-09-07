from typing import Dict, Any

class LLMClient:
    """
    Abstract LLM client. Implement complete_json and (optionally) complete_text for your provider.
    """
    def __init__(self, provider: str, model: str):
        self.provider = provider
        self.model = model

    def complete_json(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """
        Return parsed JSON from an LLM call.
        Implement with your provider's structured output / JSON mode.
        """
        raise NotImplementedError

    def complete_text(self, prompt: str, **kwargs) -> str:
        """
        Return raw text completion. Optional.
        """
        raise NotImplementedError


# Example OpenAI implementation (uncomment and fill your key to use)
# pip install openai
# from openai import OpenAI
# class OpenAILLMClient(LLMClient):
#     def __init__(self, model: str, api_key: str | None = None):
#         super().__init__("openai", model)
#         self.client = OpenAI(api_key=api_key)
#     def complete_json(self, prompt: str, **kwargs) -> Dict[str, Any]:
#         resp = self.client.chat.completions.create(
#             model=self.model,
#             messages=[{"role":"user","content":prompt}],
#             response_format={"type":"json_object"},
#             temperature=kwargs.get("temperature", 0)
#         )
#         import json
#         return json.loads(resp.choices[0].message.content)
#     def complete_text(self, prompt: str, **kwargs) -> str:
#         resp = self.client.chat.completions.create(
#             model=self.model,
#             messages=[{"role":"user","content":prompt}],
#             temperature=kwargs.get("temperature", 0.2)
#         )
#         return resp.choices[0].message.content
