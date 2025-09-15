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
        # API keys / endpoints (can be passed or picked from env)
        openai_api_key: Optional[str] = None,
        gemini_api_key: Optional[str] = None,
        sealion_api_key: Optional[str] = None,
        sealion_api_url: str = "https://api.aisingapore.org/v1/chat/completions",
        # which provider to use per task
        flags_model: Literal["gpt", "gemini", "sealion"] = "gpt",
        translation_model: Literal["gpt", "gemini", "sealion"] = "gpt",
        response_model: Literal["gpt", "gemini", "sealion"] = "gpt",
        summarization_model: Literal["gpt", "gemini", "sealion"] = "gpt",
        # concrete model names
        gpt_model: str = "gpt-4o",
        gemini_model: str = "gemini-1.5-flash",
        sealion_model: str = "aisingapore/sea-lion-7b-instruct",
        # runtime / logging
        timeout: int = 30,
        log_level: Optional[int] = None,
        logger_name: str = "kallam.dataset.unified",
    ):
        self._setup_logging(log_level, logger_name)
        self.logger.info("Initializing UnifiedDatasetOrchestrator")

        # task -> provider (gpt|gemini|sealion)
        self.task_models = {
            "flags": flags_model,
            "translation": translation_model,
            "response": response_model,
            "summarization": summarization_model,
        }

        # provider -> config
        self.model_configs = {
            "gpt": {"name": gpt_model, "temperature": 0.2, "max_tokens": 2000},
            "gemini": {"name": gemini_model, "temperature": 0.2, "max_tokens": 2000},
            "sealion": {"name": sealion_model, "temperature": 0.2, "max_tokens": 2000},
        }

        self.config = {
            "supported_languages": {"thai", "english"},
            "agents_language": "english",
            "timeout": timeout,
        }

        # store client handles here, initialize lazily
        self.clients: Dict[str, Any] = {}

        # keys / urls
        self._openai_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        self._gemini_key = gemini_api_key or os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        self._sealion_key = sealion_api_key or os.getenv("SEALION_API_KEY")
        self._sealion_url = sealion_api_url

        # initialize clients for currently selected providers only
        self._init_required_clients()

        self.logger.info(
            "UnifiedDatasetOrchestrator initialized. Task providers: %s",
            ", ".join(f"{k}:{v}" for k, v in self.task_models.items())
        )

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

    def set_task_model(self, task: str, model: Literal["gpt", "gemini", "sealion"]):
        """
        Switch provider for a specific task at runtime.

        Example:
            orch.set_task_model("response", "gemini")
        """
        task = task.lower()
        model = model.lower()
        if task not in self.task_models:
            raise ValueError(f"Invalid task {task}. Must be one of {list(self.task_models.keys())}")
        if model not in self.SUPPORTED_PROVIDERS:
            raise ValueError(f"Unsupported model {model}. Supported: {self.SUPPORTED_PROVIDERS}")
        self.task_models[task] = model
        self.logger.info("Set task '%s' to provider '%s'", task, model)
        # lazily initialize client for the new provider if needed
        try:
            self._init_required_clients()
        except Exception as e:
            # don't crash on set; caller should supply proper keys
            self.logger.debug("Client init warning after set_task_model: %s", e)

    # -------------------------------------------------------------------------
    # Internal unified caller (single-call-per-high-level-method policy)
    # -------------------------------------------------------------------------
    def _call_model(self, provider: str, *, messages: Optional[List[Dict[str, str]]] = None,
                    prompt: Optional[str] = None, temperature: Optional[float] = None,
                    max_tokens: Optional[int] = None) -> str:
        """
        Single entry point to call provider. Will return the main text content.
        - For chat-style APIs use `messages`.
        - For Gemini the code will use `prompt` if provided; otherwise convert messages to a prompt.
        """
        if provider not in self.SUPPORTED_PROVIDERS:
            raise ValueError(f"Provider {provider} not supported")

        cfg = self.model_configs.get(provider, {})
        temperature = temperature if temperature is not None else cfg.get("temperature", 0.2)
        max_tokens = max_tokens if max_tokens is not None else cfg.get("max_tokens", 2000)

        try:
            if provider == "gpt":
                client = self.clients.get("gpt")
                if client is None:
                    raise RuntimeError("GPT client not initialized")
                # Uses the OpenAI SDK 'OpenAI' client pattern
                # single call per method -> create one request here
                response = client.chat.completions.create(
                    model=cfg["name"],
                    messages=messages or [{"role": "system", "content": prompt or ""}],
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                # compatible with typical openai-like responses
                return response.choices[0].message.content.strip()

            elif provider == "gemini":
                client = self.clients.get("gemini")
                if client is None:
                    raise RuntimeError("Gemini client not initialized")
                if prompt is None and messages:
                    # convert messages list to a single prompt
                    prompt = "\n".join(f"{m.get('role','')}: {m.get('content','')}" for m in messages)
                if prompt is None:
                    prompt = ""

                # single generate_content call
                # some genai wrappers use .generate or .generate_content
                if hasattr(client, "generate_content"):
                    response = client.generate_content(prompt)
                    return getattr(response, "text", str(response)).strip()
                else:
                    # fallback: construct a temporary model object with config and call generate_content
                    temp = genai.GenerativeModel(
                        model_name=cfg["name"],
                        generation_config={"temperature": temperature, "max_output_tokens": max_tokens},
                    )
                    response = temp.generate_content(prompt)
                    return getattr(response, "text", str(response)).strip()

            elif provider == "sealion":
                info = self.clients.get("sealion")
                if info is None:
                    raise RuntimeError("SeaLion client not initialized")
                payload = {
                    "model": cfg["name"],
                    "messages": messages or [{"role": "system", "content": prompt or ""}],
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                }
                resp = requests.post(info["api_url"], headers=info["headers"], json=payload, timeout=self.config["timeout"])
                resp.raise_for_status()
                data = resp.json()
                # expect sea-lion response format similar to openai
                return data["choices"][0]["message"]["content"].strip()

        except Exception as e:
            self.logger.error("%s API call error: %s", provider.upper(), e, exc_info=True)
            raise

    # -------------------------------------------------------------------------
    # Public methods matching original orchestrator interface
    # -------------------------------------------------------------------------
    @_trace()
    def get_flags_from_supervisor(
        self,
        chat_history: Optional[List[Dict[str, str]]] = None,
        user_message: str = "",
        memory_context: Optional[str] = "",
        summarized_histories: Optional[List] = None,
    ) -> Dict[str, Any]:
        """
        Calls the configured provider once to get supervisor flags.
        Expects the model to return a JSON string the function can parse.
        """
        provider = self.task_models["flags"]
        self.logger.info("Getting flags using provider: %s", provider.upper())

        chat_history = chat_history or []
        summarized_histories = summarized_histories or []
        memory_context = memory_context or ""

        # Build a prompt that instructs the model to return JSON flags.
        # Keep the prompt concise and deterministic.
        prompt_parts = [
            "You are a supervisor agent. Analyze the user message and the recent chat history and return a JSON object with these fields:",
            "- language: 'thai' or 'english' (or null if unknown)",
            "- doctor: true/false",
            "- psychologist: true/false",
            "- intent: short string",
            "Return only valid JSON (no explanation).",
            "",
            f"User message: {user_message}",
            "",
        ]
        if chat_history:
            prompt_parts.append("Recent chat history (last items):")
            limit = min(len(chat_history), 6)
            for m in chat_history[-limit:]:
                prompt_parts.append(f"{m.get('role','user')}: {m.get('content','')}")
        if memory_context:
            prompt_parts.append(f"\nMemory context: {memory_context}")
        if summarized_histories:
            prompt_parts.append(f"\nSummaries: {json.dumps(summarized_histories)}")

        prompt = "\n".join(prompt_parts)

        # Exactly one provider call
        raw = self._call_model(provider, prompt=prompt)
        # Some LLMs may wrap JSON in backticks or text. Try to extract the first JSON blob.
        cleaned = raw.strip()
        # quick heuristic to extract {...}
        try:
            if cleaned.startswith("```"):
                # remove code fence
                cleaned = "\n".join(cleaned.splitlines()[1:-1]).strip()
            # find first `{` and last `}` to attempt JSON parsing
            start = cleaned.find("{")
            end = cleaned.rfind("}")
            if start != -1 and end != -1 and end > start:
                json_blob = cleaned[start:end+1]
            else:
                json_blob = cleaned
            flags = json.loads(json_blob)
        except Exception:
            # best-effort fallback: return safe defaults and log raw output
            self.logger.warning("Failed to parse flags JSON from provider output: %s", cleaned)
            return {"language": "invalid", "doctor": False, "psychologist": False, "intent": "unknown"}

        self.logger.debug("Supervisor flags parsed: %s", flags)
        return flags

    @_trace()
    def get_translation(self, message: str, flags: dict, translation_type: str) -> str:
        """
        Translate message using configured translation provider. Only one provider call.
        translation_type: 'forward' or 'backward' (keeps symmetry with your code).
        """
        provider = self.task_models["translation"]
        self.logger.info("Translating with provider: %s", provider.upper())

        try:
            source_lang = flags.get("language")
            if source_lang is None or source_lang not in self.config["supported_languages"]:
                # nothing to do
                return message

            default_target = self.config["agents_language"]
            if translation_type not in {"forward", "backward"}:
                raise ValueError("translation_type must be 'forward' or 'backward'")

            if translation_type == "forward":
                target_lang = default_target if source_lang != default_target else source_lang
                needs_translation = source_lang != default_target
            else:
                target_lang = source_lang if source_lang != default_target else default_target
                needs_translation = source_lang != default_target

            if not needs_translation:
                return message

            prompt = (
                f"Translate the following text from {source_lang} to {target_lang}.\n\n"
                f"Text: {message}\n\n"
                "Return only the translated text without extra commentary."
            )

            # One call
            translated = self._call_model(provider, prompt=prompt)
            return translated.strip()

        except ValueError:
            raise
        except Exception as e:
            self.logger.error("Translation failed: %s", e, exc_info=True)
            raise RuntimeError("Translation failed, please try again.")

    @_trace()
    def get_commented_response(
        self,
        original_history: List[Dict[str, str]],
        original_message: str,
        eng_history: List[Dict[str, str]],
        eng_message: str,
        flags: Dict[str, Any],
        chain_of_thoughts: List[Dict[str, str]],
        memory_context: Optional[Dict[str, Any]],
        summarized_histories: List[Dict[str, str]],
    ) -> Dict[str, Any]:
        """
        Build a single prompt representing the agent pipeline and get one model response.
        The output will be interpreted into a dictionary with keys:
            - final_output: string (assistant text)
            - commentary: optional structured commentary (best-effort parse)
        """
        provider = self.task_models["response"]
        self.logger.info("Generating response using provider: %s", provider.upper())

        # Build instruction header
        system_parts = [
            "You are an AI assistant that can act as a doctor, psychologist, or general helper depending on flags.",
            "Produce a JSON object with at least the field 'final_output' (the assistant reply as plain text).",
            "Optionally include a 'commentary' field with internal notes.",
            "Do not include extraneous explanation outside the JSON. Return only valid JSON.",
        ]
        if flags.get("doctor"):
            system_parts.append("User has medical/health-related intent. Provide safe, non-diagnostic general information and advise consulting a healthcare professional.")
        if flags.get("psychologist"):
            system_parts.append("User requires emotional support. Be empathetic and supportive; avoid giving clinical diagnoses.")

        # Build context
        parts = [*system_parts, "", f"User message: {original_message}", ""]
        if eng_history:
            parts.append("Recent English history (last few turns):")
            parts += [f"{m.get('role','user')}: {m.get('content','')}" for m in eng_history[-6:]]
        if chain_of_thoughts:
            parts.append("\nChain of thoughts (assistant internal traces):")
            parts.append(json.dumps(chain_of_thoughts))
        if memory_context:
            parts.append("\nMemory context:")
            parts.append(json.dumps(memory_context))
        if summarized_histories:
            parts.append("\nPrevious summaries:")
            parts.append(json.dumps(summarized_histories))

        # Instruction to produce final result
        parts.append("\nReturn a JSON object like: {\"final_output\": \"...\", \"commentary\": { ... }}")
        prompt = "\n".join(parts)

        # Single model call
        raw = self._call_model(provider, prompt=prompt)

        # Parse JSON out of the raw output robustly
        cleaned = raw.strip()
        try:
            if cleaned.startswith("```"):
                cleaned = "\n".join(cleaned.splitlines()[1:-1]).strip()
            start = cleaned.find("{")
            end = cleaned.rfind("}")
            if start != -1 and end != -1 and end > start:
                json_blob = cleaned[start:end+1]
            else:
                json_blob = cleaned
            parsed = json.loads(json_blob)
        except Exception:
            # If JSON parsing fails, fallback to packaging raw text under final_output
            self.logger.warning("Failed to parse model JSON response for commented_response. Using raw text as final_output.")
            parsed = {"final_output": cleaned, "commentary": {}}

        # Ensure final_output exists
        if "final_output" not in parsed:
            # If model returned pure text instead of JSON, map it
            if isinstance(parsed, str):
                parsed = {"final_output": parsed, "commentary": {}}
            else:
                # fallback to raw string under final_output
                parsed.setdefault("final_output", cleaned)
                parsed.setdefault("commentary", parsed.get("commentary", {}))

        self.logger.debug("Response parsed: keys=%s", list(parsed.keys()))
        return parsed

    @_trace()
    def summarize_history(self, chat_history: List[Dict[str, str]], eng_summaries: List[Dict[str, str]]) -> Optional[str]:
        """
        Make one provider call to summarize chat_history. Returns a single string summary.
        """
        provider = self.task_models["summarization"]
        self.logger.info("Summarizing using provider: %s", provider.upper())

        if not chat_history:
            return "No chat history to summarize."

        history_text = "\n".join(f"{m.get('role','user')}: {m.get('content','')}" for m in chat_history[-50:])
        previous = ""
        if eng_summaries:
            previous = "\n".join(s.get("summary", "") for s in eng_summaries[-3:])

        prompt = (
            "Summarize the conversation below concisely (3-5 sentences) and return only the summary text.\n\n"
            f"Conversation:\n{history_text}\n\n"
        )
        if previous:
            prompt += f"Previous summaries (for continuity):\n{previous}\n\n"

        # single call
        summary = self._call_model(provider, prompt=prompt)
        return summary.strip()

