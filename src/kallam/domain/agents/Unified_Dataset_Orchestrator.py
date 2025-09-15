import json
import logging
import os
import time
from functools import wraps
from typing import Optional, Dict, Any, List, Literal
from dotenv import load_dotenv
load_dotenv()

from openai import OpenAI
from google import genai

import requests

def _trace(level: int = logging.INFO):
    def deco(fn):
        @wraps(fn)
        def wrapper(self, *args, **kwargs):
            self.logger.log(level, f"→ {fn.__name__}")
            t0 = time.time()
            try:
                out = fn(self, *args, **kwargs)
                dt = int((time.time() - t0) * 1000)
                self.logger.log(level, f"← {fn.__name__} done in {dt} ms")
                return out
            except Exception:
                self.logger.exception(f"✗ {fn.__name__} failed")
                raise
        return wrapper
    return deco


class UnifiedDatasetOrchestrator:

    SUPPORTED_PROVIDERS = {"gpt", "gemini", "sealion"}

    def __init__(
        self,
        # which provider to use per task
        flags_model: Literal["gpt", "gemini", "sealion"] = "gpt",
        translation_model: Literal["gpt", "gemini", "sealion"] = "gpt",
        response_model: Literal["gpt", "gemini", "sealion"] = "gpt",
        summarization_model: Literal["gpt", "gemini", "sealion"] = "gpt",
        # model names
        gpt_model: str = "gpt-4o",
        gemini_model: str = "gemini-1.5-flash",
        sealion_model: str = "aisingapore/sea-lion-7b-instruct",
        # runtime
        timeout: int = 30,
        log_level: Optional[int] = None,
        logger_name: str = "kallam.dataset.unified",
    ):
        self._setup_logging(log_level, logger_name)

        self.task_models = {
            "flags": flags_model,
            "translation": translation_model,
            "response": response_model,
            "summarization": summarization_model,
        }

        self.model_configs = {
            "gpt": {"name": gpt_model, "temperature": 0.2, "max_tokens": 2000},
            "gemini": {"name": gemini_model, "temperature": 0.2, "max_tokens": 2000},
            "sealion": {"name": sealion_model, "temperature": 0.2, "max_tokens": 2000},
        }

        self.config = {"timeout": timeout}
        self.clients: Dict[str, Any] = {}

        # pick up keys from environment
        self._openai_key = os.getenv("OPENAI_API_KEY")
        self._gemini_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        self._sealion_key = os.getenv("SEALION_API_KEY")
        self._sealion_url = os.getenv("SEALION_API_URL", "https://api.aisingapore.org/v1/chat/completions")

        self._init_required_clients()

    def _setup_logging(self, log_level: Optional[int], logger_name: str):
        self.logger = logging.getLogger(logger_name)
        self.logger.propagate = True
        if log_level is not None:
            self.logger.setLevel(log_level)

    def _init_required_clients(self):
        """Initialize only clients required by current task_models."""
        required = set(self.task_models.values())
        for provider in required:
            if provider == "gpt" and "gpt" not in self.clients:
                if OpenAI is None:
                    raise RuntimeError("OpenAI client library not installed")
                if not self._openai_key:
                    raise ValueError("OPENAI_API_KEY required for GPT provider")
                self.clients["gpt"] = OpenAI(api_key=self._openai_key)
                self.logger.debug("Initialized OpenAI client")

            if provider == "gemini" and "gemini" not in self.clients:
                if genai is None:
                    raise RuntimeError("google.genai client not available")
                if not self._gemini_key:
                    raise ValueError("GOOGLE_API_KEY (or GEMINI_API_KEY) required for Gemini provider")
                genai.configure(api_key=self._gemini_key)
                # store a lightweight factory; keep one default client
                self.clients["gemini"] = genai.GenerativeModel(
                    model_name=self.model_configs["gemini"]["name"],
                    generation_config={
                        "temperature": self.model_configs["gemini"]["temperature"],
                        "max_output_tokens": self.model_configs["gemini"]["max_tokens"],
                    },
                )
                self.logger.debug("Initialized Gemini client")

            if provider == "sealion" and "sealion" not in self.clients:
                if not self._sealion_key:
                    raise ValueError("SEALION_API_KEY required for SeaLion provider")
                self.clients["sealion"] = {
                    "api_key": self._sealion_key,
                    "api_url": self._sealion_url,
                    "headers": {
                        "Authorization": f"Bearer {self._sealion_key}",
                        "Content-Type": "application/json",
                    },
                }
                self.logger.debug("Initialized SeaLion client")

    def set_task_model(self, task: str, model: str):
        if task not in self.task_models:
            raise ValueError(f"Invalid task {task}")
        if model not in self.SUPPORTED_PROVIDERS:
            raise ValueError(f"Invalid model {model}")
        self.task_models[task] = model
        self._init_required_clients()

    # -------------------------------------------------------------------------
    # Internal unified caller (single-call-per-high-level-method policy)
    # -------------------------------------------------------------------------
    def _call_model(self, provider: str, messages: List[Dict[str, str]]) -> str:
        cfg = self.model_configs[provider]

        if provider == "gpt":
            resp = self.clients["gpt"].chat.completions.create(
                model=cfg["name"],
                messages=messages,
                temperature=cfg["temperature"],
                max_tokens=cfg["max_tokens"],
            )
            return resp.choices[0].message.content.strip()

        if provider == "gemini":
            prompt = "\n".join(f"{m['role']}: {m['content']}" for m in messages)
            resp = self.clients["gemini"].generate_content(prompt)
            return resp.text.strip()

        if provider == "sealion":
            payload = {
                "model": cfg["name"],
                "messages": messages,
                "temperature": cfg["temperature"],
                "max_tokens": cfg["max_tokens"],
            }
            r = requests.post(
                self.clients["sealion"]["api_url"],
                headers=self.clients["sealion"]["headers"],
                json=payload,
                timeout=self.config["timeout"],
            )
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"].strip()

        raise ValueError(f"Unknown provider {provider}")

    # API-compatible public methods — pure forwarders
    @_trace()
    def get_flags_from_supervisor(self, messages: List[Dict[str, str]]) -> str:
        provider = self.task_models["flags"]
        return self._call_model(provider, messages)

    @_trace()
    def get_translation(self, messages: List[Dict[str, str]]) -> str:
        provider = self.task_models["translation"]
        return self._call_model(provider, messages)

    @_trace()
    def get_commented_response(self, messages: List[Dict[str, str]]) -> str:
        provider = self.task_models["response"]
        return self._call_model(provider, messages)

    @_trace()
    def summarize_history(self, messages: List[Dict[str, str]]) -> str:
        provider = self.task_models["summarization"]
        return self._call_model(provider, messages)

