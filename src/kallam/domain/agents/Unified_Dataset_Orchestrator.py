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
    def get_flags_from_supervisor(
        self,
        chat_history: Optional[List[Dict[str, str]]] = None,
        user_message: str = "",
        memory_context: Optional[str] = "",
        summarized_histories: Optional[List] = None
    ) -> Dict[str, Any]:
        self.logger.info("Getting flags from SupervisorAgent")
        chat_history = chat_history or []
        summarized_histories = summarized_histories or []
        memory_context = memory_context or ""

        string_flags = self.supervisor.generate_feedback(
            chat_history=chat_history,
            user_message=user_message,
            memory_context=memory_context,
            task="flag",
            summarized_histories=summarized_histories
        )
        dict_flags = json.loads(string_flags)
        self.logger.debug(f"Supervisor flags: {dict_flags}")
        return dict_flags

    @_trace()
    def get_translation(self, message: str, flags: dict, translation_type: str) -> str:
        """Translate message based on flags and translation type."""
        try:
            source_lang = flags.get("language")
            default_target = self.config["agents_language"]
            supported_langs = self.config["supported_languages"]

            if translation_type not in {"forward", "backward"}:
                raise ValueError(f"Invalid translation type: {translation_type}. Allowed: 'forward', 'backward'")

            if source_lang is None:
                self.logger.debug("No translation flag set, using original message")
                return message

            if source_lang not in supported_langs:
                supported = ", ".join(f"'{lang}'" for lang in supported_langs)
                raise ValueError(f"Invalid translate flag: '{source_lang}'. Supported: {supported}")

            if translation_type == "forward":
                target_lang = default_target if source_lang != default_target else source_lang
                needs_translation = source_lang != default_target
            else:
                target_lang = source_lang if source_lang != default_target else default_target
                needs_translation = source_lang != default_target

            if needs_translation:
                self.logger.debug(f"Translating {translation_type}: '{source_lang}' -> '{target_lang}'")
                return self.translator.get_translation(message=message, target=target_lang)
            else:
                self.logger.debug(f"Source '{source_lang}' same as target, using original message")
                return message

        except ValueError:
            raise
        except Exception as e:
            self.logger.error(f"Unexpected error in translation: {e}", exc_info=True)
            raise RuntimeError(
                "เกิดข้อผิดพลาดขณะแปล โปรดตรวจสอบให้แน่ใจว่าคุณใช้ภาษาที่รองรับ แล้วลองอีกครั้ง\n"
                "An error occurred while translating. Please make sure you are using a supported language and try again."
            )

    @_trace()
    def get_commented_response(self,
                     original_history: List[Dict[str, str]],
                     original_message: str,
                     eng_history: List[Dict[str, str]],
                     eng_message: str,
                     flags: Dict[str, Any],
                     chain_of_thoughts: List[Dict[str, str]],
                     memory_context: Optional[Dict[str, Any]],
                     summarized_histories: List[Dict[str, str]]) -> Dict[str, Any]:

        self.logger.info(f"Routing message: {eng_message} | Flags: {flags}")

        commentary = {}

        if flags.get("doctor"):  # Dummy for now
            self.logger.debug("Activating DoctorAgent")
            commentary["doctor"] = self.doctor.analyze(
                eng_message, eng_history, json.dumps(chain_of_thoughts), json.dumps(summarized_histories)
            )

        if flags.get("psychologist"):  # Dummy for now
            self.logger.debug("Activating PsychologistAgent")
            commentary["psychologist"] = self.psychologist.analyze(
                original_message, original_history, chain_of_thoughts, summarized_histories
            )

        commentary["final_output"] = self.supervisor.generate_feedback(
            chat_history=original_history,
            user_message=original_message,
            memory_context=json.dumps(memory_context) if memory_context else "",
            task="finalize",
            summarized_histories=summarized_histories,
            commentary=commentary,
        )
        self.logger.info("Routing complete. Returning results.")
        return commentary

    def _merge_outputs(self, outputs: dict) -> str:
        """
        Merge multiple agent responses into a single final string.
        For now: concatenate. Later: ranking or weighting logic.
        """
        final = []
        for agent, out in outputs.items():
            if agent != "final_output":
                final.append(f"[{agent.upper()}]: {out}")
        return "\n".join(final)

    @_trace()
    def summarize_history(self,
                          chat_history: List[Dict[str, str]],
                          eng_summaries: List[Dict[str, str]]) -> Optional[str]:
        try:
            summary = self.summarizer.summarize(chat_history, eng_summaries)
            self.logger.info("Summarization complete.")
        except Exception as e:
            self.logger.error(f"Error during summarization: {e}", exc_info=True)
            summary = "Error during summarization."
        return str(summary)

