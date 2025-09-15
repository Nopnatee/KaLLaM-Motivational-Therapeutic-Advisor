# src/your_pkg/domain/agents/dataset_orchestrator_gemini.py
import json
import logging
from google import genai
from typing import Optional, Dict, Any, List
from functools import wraps
import time


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


class GeminiDatasetOrchestrator:
    # ----------------------------------------------------------------------------------------------
    # Initialization

    def __init__(self, 
                 api_key: Optional[str] = None,
                 model_name: str = "gemini-1.5-flash",
                 log_level: int | None = None, 
                 logger_name: str = "kallam.dataset.gemini"):
        
        self._setup_logging(log_level, logger_name)

        # Configure Gemini
        if api_key:
            genai.configure(api_key=api_key)
        
        # Configuration
        self.config = {
            "model": model_name,
            "temperature": 0.7,
            "max_output_tokens": 2000,
            "supported_languages": {"thai", "english"},
            "agents_language": "english"
        }

        # Initialize model
        generation_config = {
            "temperature": self.config["temperature"],
            "max_output_tokens": self.config["max_output_tokens"],
        }
        
        self.model = genai.GenerativeModel(
            model_name=model_name,
            generation_config=generation_config
        )

        self.logger.info(f"Gemini Dataset Orchestrator initialized with model: {model_name}")

    def _setup_logging(self, log_level: int | None, logger_name: str) -> None:
        """
        Use a child logger so we inherit handlers/formatters/filters (incl. request_id)
        from the ChatbotManager root logger setup.
        """
        self.logger = logging.getLogger(logger_name)
        self.logger.propagate = True
        if log_level is not None:
            self.logger.setLevel(log_level)

    def _call_gemini(self, prompt: str, temperature: float = None) -> str:
        """Make a call to Gemini API with error handling."""
        try:
            if temperature is not None:
                # Create a new model instance with different temperature
                generation_config = {
                    "temperature": temperature,
                    "max_output_tokens": self.config["max_output_tokens"],
                }
                temp_model = genai.GenerativeModel(
                    model_name=self.config["model"],
                    generation_config=generation_config
                )
                response = temp_model.generate_content(prompt)
            else:
                response = self.model.generate_content(prompt)
            
            return response.text.strip()
        except Exception as e:
            self.logger.error(f"Gemini API call failed: {e}")
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
        """Generate flags using Gemini to analyze user message and context."""
        self.logger.info("Getting flags from Gemini supervisor")
        chat_history = chat_history or []
        summarized_histories = summarized_histories or []
        memory_context = memory_context or ""

        # Build context for Gemini
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

Analyze the message and context, then return ONLY a JSON object with the following flags:
- "language": detect the language ("thai", "english", or "invalid" if unclear)
- "doctor": boolean, true if medical/health related content
- "psychologist": boolean, true if mental health/psychology related content

Return ONLY valid JSON format, no markdown, no additional text, no explanations.

Example: {{"language": "thai", "doctor": false, "psychologist": true}}
"""

        response = self._call_gemini(prompt, temperature=0.3)
        
        try:
            # Clean up response in case it has markdown formatting
            cleaned_response = response.replace('```json', '').replace('```', '').strip()
            flags = json.loads(cleaned_response)
            self.logger.debug(f"Gemini supervisor flags: {flags}")
            return flags
        except json.JSONDecodeError:
            self.logger.warning(f"Failed to parse Gemini response as JSON: {response}")
            return {"language": "invalid", "doctor": False, "psychologist": False}

    @_trace()
    def get_translation(self, message: str, flags: dict, translation_type: str) -> str:
        """Translate message using Gemini based on flags and translation type."""
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
Maintain the original meaning and context. 
Return ONLY the translation, no additional text or explanations.

Text to translate: "{message}"

Translation:
"""
                translation = self._call_gemini(prompt, temperature=0.3)
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
        """Generate a response using Gemini based on all provided context."""
        
        self.logger.info(f"Generating response for: {eng_message} | Flags: {flags}")

        # Build comprehensive context for Gemini
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
        response_instructions = "You are a helpful AI assistant. Respond naturally and helpfully to the user's message."
        
        if flags.get("doctor"):
            response_instructions += "\n\nThe user seems to have medical/health questions. Provide helpful information but always remind them to consult qualified healthcare professionals for serious medical concerns."
        
        if flags.get("psychologist"):
            response_instructions += "\n\nThe user seems to need emotional or psychological support. Be empathetic, supportive, and understanding in your response."

        prompt = f"""
{response_instructions}

Context Information:
{context_text}

User's current message: "{original_message}"

Instructions:
- Provide a helpful, contextually appropriate response
- Respond in the same language as the user's message
- Be natural and conversational
- Use the context to provide more relevant responses
- Keep your response focused and helpful

Response:
"""

        response = self._call_gemini(prompt)

        commentary = {
            "final_output": response,
            "model_used": self.config["model"],
            "flags_processed": flags
        }

        if flags.get("doctor"):
            commentary["doctor_analysis"] = "Medical context detected and addressed"
        
        if flags.get("psychologist"):
            commentary["psychology_analysis"] = "Psychological support context detected and addressed"

        self.logger.info("Gemini response generation complete")
        return commentary

    @_trace()
    def summarize_history(self,
                          chat_history: List[Dict[str, str]],
                          eng_summaries: List[Dict[str, str]]) -> Optional[str]:
        """Generate a summary of chat history using Gemini."""
        try:
            if not chat_history:
                return "No chat history to summarize."

            history_text = "\n".join([f"{msg.get('role', 'user')}: {msg.get('content', '')}" for msg in chat_history])
            
            previous_summaries = ""
            if eng_summaries:
                previous_summaries = "\n".join([summary.get('summary', '') for summary in eng_summaries[-3:]])

            prompt = f"""
Create a concise summary of the following conversation. Focus on:
- Key topics discussed
- Important decisions made
- Main questions asked and answered
- Overall context and themes

Previous Summaries (for context):
{previous_summaries}

Current Conversation to Summarize:
{history_text}

Instructions:
- Provide a clear, concise summary that captures the main points
- Keep it informative but brief
- Focus on the most important information
- Write in a neutral, descriptive tone

Summary:
"""

            summary = self._call_gemini(prompt, temperature=0.3)
            
            self.logger.info("Gemini summarization complete.")
            return summary
            
        except Exception as e:
            self.logger.error(f"Error during summarization: {e}", exc_info=True)
            return "Error during summarization."