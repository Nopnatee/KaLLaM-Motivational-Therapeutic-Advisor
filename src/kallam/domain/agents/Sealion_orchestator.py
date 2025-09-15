# src/your_pkg/domain/agents/dataset_orchestrator_sealion.py
import json
import logging
from typing import Optional, Dict, Any, List
from functools import wraps
import time
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


class SeaLionDatasetOrchestrator:
    # ----------------------------------------------------------------------------------------------
    # Initialization

    def __init__(self, 
                 api_key: Optional[str] = None,
                 api_url: str = "https://api.aisingapore.org/v1/chat/completions",
                 model_name: str = "aisingapore/sea-lion-7b-instruct",
                 log_level: int | None = None, 
                 logger_name: str = "kallam.dataset.sealion"):
        """
        SeaLion-based dataset orchestrator that handles all tasks with a single LLM API.
        
        Args:
            api_key: SeaLion API key. If None, will look for SEALION_API_KEY environment variable
            api_url: SeaLion API endpoint URL
            model_name: SeaLion model to use (default: aisingapore/sea-lion-7b-instruct)
            log_level: if provided, sets this logger's level; otherwise inherit from parent.
            logger_name: child logger under the manager's logger hierarchy.
        """
        self._setup_logging(log_level, logger_name)

        # API configuration
        import os
        self.api_key = api_key or os.getenv("SEALION_API_KEY")
        if not self.api_key:
            raise ValueError("SeaLion API key is required. Set SEALION_API_KEY environment variable or pass api_key parameter.")
        
        self.api_url = api_url
        
        # Configuration
        self.config = {
            "model": model_name,
            "temperature": 0.7,
            "max_tokens": 2000,
            "timeout": 30,
            "supported_languages": {"thai", "english"},
            "agents_language": "english"
        }

        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        self.logger.info(f"SeaLion Dataset Orchestrator initialized with model: {model_name}")

    def _setup_logging(self, log_level: int | None, logger_name: str) -> None:
        """
        Use a child logger so we inherit handlers/formatters/filters (incl. request_id)
        from the ChatbotManager root logger setup.
        """
        self.logger = logging.getLogger(logger_name)
        self.logger.propagate = True
        if log_level is not None:
            self.logger.setLevel(log_level)

    def _call_sealion(self, messages: List[Dict[str, str]], temperature: float = None) -> str:
        """Make a call to SeaLion API with error handling."""
        try:
            temp = temperature if temperature is not None else self.config["temperature"]
            
            payload = {
                "model": self.config["model"],
                "messages": messages,
                "temperature": temp,
                "max_tokens": self.config["max_tokens"]
            }

            response = requests.post(
                self.api_url,
                headers=self.headers,
                json=payload,
                timeout=self.config["timeout"]
            )
            
            response.raise_for_status()
            
            result = response.json()
            return result["choices"][0]["message"]["content"].strip()
            
        except requests.exceptions.RequestException as e:
            self.logger.error(f"SeaLion API request failed: {e}")
            raise
        except KeyError as e:
            self.logger.error(f"SeaLion API response format error: {e}")
            raise
        except Exception as e:
            self.logger.error(f"SeaLion API call failed: {e}")
            raise

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
        """Generate flags using SeaLion to analyze user message and context."""
        self.logger.info("Getting flags from SeaLion supervisor")
        chat_history = chat_history or []
        summarized_histories = summarized_histories or []
        memory_context = memory_context or ""

        # Build context for SeaLion
        context_text = ""
        if chat_history:
            context_text += "Chat History:\n" + "\n".join([f"{msg.get('role', 'unknown')}: {msg.get('content', '')}" for msg in chat_history[-5:]])
        if memory_context:
            context_text += f"\nMemory Context: {memory_context}"
        if summarized_histories:
            context_text += f"\nPrevious Summaries: {json.dumps(summarized_histories)}"

        system_prompt = """You are a supervisor agent that analyzes user messages and context to generate flags for routing.

Your task is to analyze the user message and return ONLY a valid JSON object with these flags:
- "language": detect the language ("thai", "english", or "invalid" if unclear)
- "doctor": boolean, true if medical/health related content
- "psychologist": boolean, true if mental health/psychology related content

Return ONLY the JSON object, no additional text or explanations."""

        user_prompt = f"""Context:
{context_text}

Current User Message: "{user_message}"

Analyze and return the JSON flags:"""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        response = self._call_sealion(messages, temperature=0.3)
        
        try:
            # Clean up response in case it has extra text
            cleaned_response = response.strip()
            # Try to extract JSON if it's wrapped in other text
            if '{' in cleaned_response and '}' in cleaned_response:
                start = cleaned_response.find('{')
                end = cleaned_response.rfind('}') + 1
                cleaned_response = cleaned_response[start:end]
            
            flags = json.loads(cleaned_response)
            self.logger.debug(f"SeaLion supervisor flags: {flags}")
            return flags
        except json.JSONDecodeError:
            self.logger.warning(f"Failed to parse SeaLion response as JSON: {response}")
            return {"language": "invalid", "doctor": False, "psychologist": False}

    @_trace()
    def get_translation(self, message: str, flags: dict, translation_type: str) -> str:
        """Translate message using SeaLion based on flags and translation type."""
        try:
            source_lang = flags.get("language")
            default_target = self.config["agents_language"]
            supported_langs = self.config["supported_languages"]

            if translation_type not in {"forward", "backward"}:
                raise ValueError(f"Invalid translation type: {translation_type}. Allowed: 'forward', 'backward'")

            if source_lang is None or source_lang not in supported_langs:
                self.logger.debug("No valid translation flag, using original message")
                return message

            if translation_type == "forward":
                target_lang = default_target if source_lang != default_target else source_lang
                needs_translation = source_lang != default_target
            else:
                target_lang = source_lang if source_lang != default_target else default_target
                needs_translation = source_lang != default_target

            if needs_translation:
                self.logger.debug(f"Translating {translation_type}: '{source_lang}' -> '{target_lang}'")
                
                system_prompt = f"You are a professional translator. Translate text from {source_lang} to {target_lang} while maintaining the original meaning and context. Return ONLY the translation."
                
                user_prompt = f'Translate this text: "{message}"'

                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ]
                
                translation = self._call_sealion(messages, temperature=0.3)
                return translation
            else:
                self.logger.debug(f"Source '{source_lang}' same as target, using original message")
                return message

        except ValueError:
            raise
        except Exception as e:
            self.logger.error(f"Unexpected error in translation: {e}", exc_info=True)
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
        """Generate a response using SeaLion based on all provided context."""
        
        self.logger.info(f"Generating response for: {eng_message} | Flags: {flags}")

        # Build comprehensive context for SeaLion
        context_parts = []
        
        if original_history:
            history_text = "\n".join([f"{msg.get('role', 'user')}: {msg.get('content', '')}" for msg in original_history[-10:]])
            context_parts.append(f"Recent Chat History:\n{history_text}")
        
        if memory_context:
            context_parts.append(f"Memory Context: {json.dumps(memory_context)}")
        
        if summarized_histories:
            context_parts.append(f"Previous Session Summaries: {json.dumps(summarized_histories)}")
        
        if chain_of_thoughts:
            context_parts.append(f"Previous Reasoning: {json.dumps(chain_of_thoughts)}")

        context_text = "\n\n".join(context_parts)

        # Determine response style based on flags
        system_instructions = "You are a helpful AI assistant. Respond naturally and helpfully to the user's message. Use the provided context to give more relevant and personalized responses."
        
        if flags.get("doctor"):
            system_instructions += " The user has medical/health questions. Provide helpful information but always remind them to consult qualified healthcare professionals for serious medical concerns."
        
        if flags.get("psychologist"):
            system_instructions += " The user needs emotional or psychological support. Be empathetic, supportive, and understanding in your response."

        user_prompt = f"""Context Information:
{context_text}

User's message: "{original_message}"

Please provide a helpful response in the same language as the user's message."""

        messages = [
            {"role": "system", "content": system_instructions},
            {"role": "user", "content": user_prompt}
        ]

        response = self._call_sealion(messages)

        commentary = {
            "final_output": response,
            "model_used": self.config["model"],
            "flags_processed": flags
        }

        if flags.get("doctor"):
            commentary["doctor_analysis"] = "Medical context detected and addressed"
        
        if flags.get("psychologist"):
            commentary["psychology_analysis"] = "Psychological support context detected and addressed"

        self.logger.info("SeaLion response generation complete")
        return commentary

    @_trace()
    def summarize_history(self,
                          chat_history: List[Dict[str, str]],
                          eng_summaries: List[Dict[str, str]]) -> Optional[str]:
        """Generate a summary of chat history using SeaLion."""
        try:
            if not chat_history:
                return "No chat history to summarize."

            history_text = "\n".join([f"{msg.get('role', 'user')}: {msg.get('content', '')}" for msg in chat_history])
            
            previous_summaries = ""
            if eng_summaries:
                previous_summaries = "\n".join([summary.get('summary', '') for summary in eng_summaries[-3:]])

            system_prompt = """You are an expert at creating concise conversation summaries. Focus on:
- Key topics and themes discussed
- Important decisions or conclusions reached
- Main questions asked and answered
- Overall context and progression of the conversation

Create a clear, informative summary that captures the essence of the conversation."""

            user_prompt = f"""Previous Summaries (for context):
{previous_summaries}

Current Conversation to Summarize:
{history_text}

Please create a concise summary:"""

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]

            summary = self._call_sealion(messages, temperature=0.3)
            
            self.logger.info("SeaLion summarization complete.")
            return summary
            
        except Exception as e:
            self.logger.error(f"Error during summarization: {e}", exc_info=True)
            return "Error during summarization."