# src/your_pkg/domain/agents/dataset_orchestrator_gpt.py
import json
import logging
from typing import Optional, Dict, Any, List
from functools import wraps
import time
import openai
from openai import OpenAI


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


class GPTDatasetOrchestrator:
    # ----------------------------------------------------------------------------------------------
    # Initialization

    def __init__(self, 
                 api_key: Optional[str] = None,
                 model_name: str = "gpt-4o",
                 log_level: int | None = None, 
                 logger_name: str = "kallam.dataset.gpt"):
        """
        GPT-based dataset orchestrator that handles all tasks with a single LLM API.
        
        Args:
            api_key: OpenAI API key. If None, will look for OPENAI_API_KEY environment variable
            model_name: GPT model to use (default: gpt-4o)
            log_level: if provided, sets this logger's level; otherwise inherit from parent.
            logger_name: child logger under the manager's logger hierarchy.
        """
        self._setup_logging(log_level, logger_name)

        # Initialize OpenAI client
        self.client = OpenAI(api_key=api_key)
        
        # Configuration
        self.config = {
            "model": model_name,
            "temperature": 0.7,
            "max_tokens": 2000,
            "supported_languages": {"thai", "english"},
            "agents_language": "english"
        }

        self.logger.info(f"GPT Dataset Orchestrator initialized with model: {model_name}")

    def _setup_logging(self, log_level: int | None, logger_name: str) -> None:
        """
        Use a child logger so we inherit handlers/formatters/filters (incl. request_id)
        from the ChatbotManager root logger setup.
        """
        self.logger = logging.getLogger(logger_name)
        self.logger.propagate = True
        if log_level is not None:
            self.logger.setLevel(log_level)

    def _call_gpt(self, messages: List[Dict[str, str]], temperature: float = None) -> str:
        """Make a call to GPT API with error handling."""
        try:
            temp = temperature if temperature is not None else self.config["temperature"]
            response = self.client.chat.completions.create(
                model=self.config["model"],
                messages=messages,
                temperature=temp,
                max_tokens=self.config["max_tokens"]
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            self.logger.error(f"GPT API call failed: {e}")
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
        """Generate flags using GPT to analyze user message and context."""
        self.logger.info("Getting flags from GPT supervisor")
        chat_history = chat_history or []
        summarized_histories = summarized_histories or []
        memory_context = memory_context or ""

        # Build context for GPT
        context_text = ""
        if chat_history:
            context_text += "Chat History:\n" + "\n".join([f"{msg.get('role', 'unknown')}: {msg.get('content', '')}" for msg in chat_history[-5:]])
        if memory_context:
            context_text += f"\nMemory Context: {memory_context}"
        if summarized_histories:
            context_text += f"\nPrevious Summaries: {json.dumps(summarized_histories)}"

        prompt = f"""
You are a supervisor agent that analyzes user messages and context to generate flags for routing.

Context:
{context_text}

Current User Message: "{user_message}"

Analyze the message and context, then return a JSON object with the following flags:
- "language": detect the language ("thai", "english", or "invalid" if unclear)
- "doctor": boolean, true if medical/health related content
- "psychologist": boolean, true if mental health/psychology related content

Only return valid JSON, no additional text.

Example output:
{{"language": "thai", "doctor": false, "psychologist": true}}
"""

        messages = [{"role": "user", "content": prompt}]
        response = self._call_gpt(messages, temperature=0.3)
        
        try:
            flags = json.loads(response)
            self.logger.debug(f"GPT supervisor flags: {flags}")
            return flags
        except json.JSONDecodeError:
            self.logger.warning(f"Failed to parse GPT response as JSON: {response}")
            return {"language": "invalid", "doctor": False, "psychologist": False}

    @_trace()
    def get_translation(self, message: str, flags: dict, translation_type: str) -> str:
        """Translate message using GPT based on flags and translation type."""
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
                
                prompt = f"""
Translate the following text from {source_lang} to {target_lang}. 
Maintain the original meaning and context. Only return the translation, no additional text.

Text to translate: "{message}"
"""
                messages = [{"role": "user", "content": prompt}]
                translation = self._call_gpt(messages, temperature=0.3)
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
        """Generate a response using GPT based on all provided context."""
        
        self.logger.info(f"Generating response for: {eng_message} | Flags: {flags}")

        # Build comprehensive context for GPT
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
        response_instructions = "You are a helpful AI assistant. Respond naturally and helpfully."
        
        if flags.get("doctor"):
            response_instructions += " The user seems to have medical/health questions. Provide helpful information but remind them to consult healthcare professionals for serious concerns."
        
        if flags.get("psychologist"):
            response_instructions += " The user seems to need emotional or psychological support. Be empathetic and supportive in your response."

        prompt = f"""
{response_instructions}

Context:
{context_text}

User's current message: "{original_message}"

Please provide a helpful, contextually appropriate response. Respond in the same language as the user's message.
"""

        messages = [{"role": "system", "content": response_instructions}]
        if context_text.strip():
            messages.append({"role": "user", "content": f"Context: {context_text}"})
        messages.append({"role": "user", "content": original_message})

        response = self._call_gpt(messages)

        commentary = {
            "final_output": response,
            "model_used": self.config["model"],
            "flags_processed": flags
        }

        if flags.get("doctor"):
            commentary["doctor_analysis"] = "Medical context detected and addressed"
        
        if flags.get("psychologist"):
            commentary["psychology_analysis"] = "Psychological support context detected and addressed"

        self.logger.info("GPT response generation complete")
        return commentary

    @_trace()
    def summarize_history(self,
                          chat_history: List[Dict[str, str]],
                          eng_summaries: List[Dict[str, str]]) -> Optional[str]:
        """Generate a summary of chat history using GPT."""
        try:
            if not chat_history:
                return "No chat history to summarize."

            history_text = "\n".join([f"{msg.get('role', 'user')}: {msg.get('content', '')}" for msg in chat_history])
            
            previous_summaries = ""
            if eng_summaries:
                previous_summaries = "\n".join([summary.get('summary', '') for summary in eng_summaries[-3:]])

            prompt = f"""
Create a concise summary of this conversation. Focus on key topics, decisions, and important information.

Previous Summaries (for context):
{previous_summaries}

Current Conversation to Summarize:
{history_text}

Provide a clear, concise summary that captures the main points and context:
"""

            messages = [{"role": "user", "content": prompt}]
            summary = self._call_gpt(messages, temperature=0.3)
            
            self.logger.info("GPT summarization complete.")
            return summary
            
        except Exception as e:
            self.logger.error(f"Error during summarization: {e}", exc_info=True)
            return "Error during summarization."