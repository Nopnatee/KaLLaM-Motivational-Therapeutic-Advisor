# src/your_pkg/domain/agents/dataset_orchestrator.py
import json
import logging
import os
from typing import Optional, Dict, Any, List, Literal
from functools import wraps
import time
from enum import Enum

# API clients
import openai
from openai import OpenAI
from google import genai
import requests


def _trace(level: int = logging.INFO):
    """Lightweight enter/exit trace with timing; relies on parent handlers/format."""
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


class ModelProvider(Enum):
    """Supported model providers."""
    GPT = "gpt"
    GEMINI = "gemini" 
    SEALION = "sealion"


class UnifiedDatasetOrchestrator:
    """
    Unified dataset orchestrator that can use different models for different tasks.
    You can specify which model to use for each operation: flags, translation, response, summarization.
    """
    
    def __init__(self,
                 # API Keys
                 openai_api_key: Optional[str] = None,
                 gemini_api_key: Optional[str] = None,
                 sealion_api_key: Optional[str] = None,
                 sealion_api_url: str = "https://api.aisingapore.org/v1/chat/completions",
                 
                 # Model selection for each task
                 flags_model: Literal["gpt", "gemini", "sealion"] = "gpt",
                 translation_model: Literal["gpt", "gemini", "sealion"] = "gpt", 
                 response_model: Literal["gpt", "gemini", "sealion"] = "gpt",
                 summarization_model: Literal["gpt", "gemini", "sealion"] = "gpt",
                 
                 # Specific model names
                 gpt_model: str = "gpt-4o",
                 gemini_model: str = "gemini-1.5-pro",
                 sealion_model: str = "aisingapore/sea-lion-7b-instruct",
                 
                 # Logging
                 log_level: int | None = None,
                 logger_name: str = "kallam.dataset.unified"):
        
        self._setup_logging(log_level, logger_name)
        
        # Store model preferences for each task
        self.task_models = {
            "flags": flags_model,
            "translation": translation_model,
            "response": response_model,
            "summarization": summarization_model
        }
        
        # Model configurations
        self.model_configs = {
            "gpt": {"name": gpt_model, "temperature": 0.7, "max_tokens": 2000},
            "gemini": {"name": gemini_model, "temperature": 0.7, "max_tokens": 2000},
            "sealion": {"name": sealion_model, "temperature": 0.7, "max_tokens": 2000}
        }
        
        # General config
        self.config = {
            "supported_languages": {"thai", "english"},
            "agents_language": "english",
            "timeout": 30
        }
        
        # Initialize API clients
        self._init_clients(openai_api_key, gemini_api_key, sealion_api_key, sealion_api_url)
        
        self.logger.info(f"Unified Dataset Orchestrator initialized:")
        self.logger.info(f"  Flags: {flags_model}, Translation: {translation_model}")
        self.logger.info(f"  Response: {response_model}, Summarization: {summarization_model}")

    def _setup_logging(self, log_level: int | None, logger_name: str) -> None:
        """Setup logging to inherit from parent logger."""
        self.logger = logging.getLogger(logger_name)
        self.logger.propagate = True
        if log_level is not None:
            self.logger.setLevel(log_level)

    def _init_clients(self, openai_key, gemini_key, sealion_key, sealion_url):
        """Initialize API clients based on required models."""
        self.clients = {}
        
        # Initialize OpenAI if needed
        if "gpt" in self.task_models.values():
            api_key = openai_key or os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OpenAI API key required but not provided")
            self.clients["gpt"] = OpenAI(api_key=api_key)
            self.logger.debug("OpenAI client initialized")
        
        # Initialize Gemini if needed  
        if "gemini" in self.task_models.values():
            api_key = gemini_key or os.getenv("GOOGLE_API_KEY")
            if not api_key:
                raise ValueError("Google AI API key required but not provided")
            genai.configure(api_key=api_key)
            self.clients["gemini"] = genai.GenerativeModel(
                model_name=self.model_configs["gemini"]["name"],
                generation_config={
                    "temperature": self.model_configs["gemini"]["temperature"],
                    "max_output_tokens": self.model_configs["gemini"]["max_tokens"]
                }
            )
            self.logger.debug("Gemini client initialized")
        
        # Initialize SeaLion if needed
        if "sealion" in self.task_models.values():
            api_key = sealion_key or os.getenv("SEALION_API_KEY")
            if not api_key:
                raise ValueError("SeaLion API key required but not provided")
            self.clients["sealion"] = {
                "api_key": api_key,
                "api_url": sealion_url,
                "headers": {
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json"
                }
            }
            self.logger.debug("SeaLion client initialized")

    def _call_model(self, provider: str, messages: List[Dict[str, str]] = None, 
                   prompt: str = None, temperature: float = None) -> str:
        """
        Unified method to call any model provider.
        
        Args:
            provider: "gpt", "gemini", or "sealion"
            messages: List of messages for chat-based APIs (GPT, SeaLion)
            prompt: Direct prompt for Gemini
            temperature: Override default temperature
        """
        temp = temperature if temperature is not None else self.model_configs[provider]["temperature"]
        
        try:
            if provider == "gpt":
                response = self.clients["gpt"].chat.completions.create(
                    model=self.model_configs["gpt"]["name"],
                    messages=messages,
                    temperature=temp,
                    max_tokens=self.model_configs["gpt"]["max_tokens"]
                )
                return response.choices[0].message.content.strip()
                
            elif provider == "gemini":
                # For Gemini, use prompt directly or convert messages to prompt
                if prompt is None and messages:
                    # Convert messages to a single prompt
                    prompt = "\n".join([f"{msg['role']}: {msg['content']}" for msg in messages])
                
                if temp != self.model_configs["gemini"]["temperature"]:
                    # Create temporary model with different temperature
                    temp_model = genai.GenerativeModel(
                        model_name=self.model_configs["gemini"]["name"],
                        generation_config={
                            "temperature": temp,
                            "max_output_tokens": self.model_configs["gemini"]["max_tokens"]
                        }
                    )
                    response = temp_model.generate_content(prompt)
                else:
                    response = self.clients["gemini"].generate_content(prompt)
                return response.text.strip()
                
            elif provider == "sealion":
                payload = {
                    "model": self.model_configs["sealion"]["name"],
                    "messages": messages,
                    "temperature": temp,
                    "max_tokens": self.model_configs["sealion"]["max_tokens"]
                }
                
                response = requests.post(
                    self.clients["sealion"]["api_url"],
                    headers=self.clients["sealion"]["headers"],
                    json=payload,
                    timeout=self.config["timeout"]
                )
                response.raise_for_status()
                result = response.json()
                return result["choices"][0]["message"]["content"].strip()
                
        except Exception as e:
            self.logger.error(f"{provider.upper()} API call failed: {e}")
            raise

    def set_task_model(self, task: str, model: str) -> None:
        """
        Change which model to use for a specific task at runtime.
        
        Args:
            task: "flags", "translation", "response", or "summarization"
            model: "gpt", "gemini", or "sealion"
        """
        if task not in self.task_models:
            raise ValueError(f"Invalid task: {task}. Must be one of {list(self.task_models.keys())}")
        if model not in ["gpt", "gemini", "sealion"]:
            raise ValueError(f"Invalid model: {model}. Must be one of ['gpt', 'gemini', 'sealion']")
        
        self.task_models[task] = model
        self.logger.info(f"Updated {task} model to: {model}")
        
        # Initialize client if not already done
        if model not in self.clients:
            self.logger.warning(f"Client for {model} not initialized. You may need to provide API key.")

    # ----------------------------------------------------------------------------------------------
    # Core Methods (matching original orchestrator interface)

    @_trace()
    def get_flags_from_supervisor(
        self,
        chat_history: Optional[List[Dict[str, str]]] = None,
        user_message: str = "",
        memory_context: Optional[str] = "",
        summarized_histories: Optional[List] = None
    ) -> Dict[str, Any]:
        model = self.task_models["flags"]
        self.logger.info(f"Getting flags using {model.upper()}")
        chat_history = chat_history or []
        summarized_histories = summarized_histories or []
        memory_context = memory_context or ""

        # Build context
        context_text = ""
        if chat_history:
            context_text += "Chat History:\n" + "\n".join([
                f"{msg.get('role', 'unknown')}: {msg.get('content', '')}" 
                for msg in chat_history[-5:]
            ])
        if memory_context:
            context_text += f"\nMemory Context: {memory_context}"
        if summarized_histories:
            context_text += f"\nPrevious Summaries: {json.dumps(summarized_histories)}"
            
            flags = json.loads(cleaned)
            self.logger.debug(f"{model.upper()} supervisor flags: {flags}")
            return flags

    @_trace()
    def get_translation(self, message: str, flags: dict, translation_type: str) -> str:
        """Translate message using the configured model for translation task."""
        model = self.task_models["translation"]
        
        try:
            source_lang = flags.get("language")
            default_target = self.config["agents_language"]
            supported_langs = self.config["supported_languages"]

            if translation_type not in {"forward", "backward"}:
                raise ValueError(f"Invalid translation type: {translation_type}")

            if source_lang is None or source_lang not in supported_langs:
                return message

            if translation_type == "forward":
                target_lang = default_target if source_lang != default_target else source_lang
                needs_translation = source_lang != default_target
            else:
                target_lang = source_lang if source_lang != default_target else default_target
                needs_translation = source_lang != default_target

            if needs_translation:
                self.logger.debug(f"Translating with {model.upper()}: '{source_lang}' -> '{target_lang}'")
            else:
                return message

        except ValueError:
            raise
        except Exception as e:
            self.logger.error(f"Translation error: {e}", exc_info=True)
            raise RuntimeError("Translation error occurred. Please try again.")

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
        """Generate response using the configured model for response task."""
        
        model = self.task_models["response"]
        self.logger.info(f"Generating response with {model.upper()}: {eng_message} | Flags: {flags}")

        # Build context
        context_parts = []
        if original_history:
            history_text = "\n".join([
                f"{msg.get('role', 'user')}: {msg.get('content', '')}" 
                for msg in original_history[-10:]
            ])
            context_parts.append(f"Recent Chat History:\n{history_text}")
        
        if memory_context:
            context_parts.append(f"Memory Context: {json.dumps(memory_context)}")
        if summarized_histories:
            context_parts.append(f"Previous Session Summaries: {json.dumps(summarized_histories)}")
        if chain_of_thoughts:
            context_parts.append(f"Previous Reasoning: {json.dumps(chain_of_thoughts)}")

        # Build instructions based on flags
        system_msg = "You are a helpful AI assistant. Respond naturally and helpfully."
        
        if flags.get("doctor"):
            system_msg += " The user has medical/health questions. Provide helpful information but remind them to consult healthcare professionals."
        
        if flags.get("psychologist"):
            system_msg += " The user needs emotional/psychological support. Be empathetic and supportive."

    @_trace()
    def summarize_history(self,
                          chat_history: List[Dict[str, str]],
                          eng_summaries: List[Dict[str, str]]) -> Optional[str]:
        """Generate summary using the configured model for summarization task."""
        
        model = self.task_models["summarization"]
        
        try:
            if not chat_history:
                return "No chat history to summarize."

            history_text = "\n".join([
                f"{msg.get('role', 'user')}: {msg.get('content', '')}" 
                for msg in chat_history
            ])
            
            previous_summaries = ""
            if eng_summaries:
                previous_summaries = "\n".join([
                    summary.get('summary', '') 
                    for summary in eng_summaries[-3:]
                ])