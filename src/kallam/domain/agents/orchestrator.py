import json
import logging
from typing import Optional, Dict, Any, List
from functools import wraps
import time

# Load agents
from .supervisor import SupervisorAgent
from .summarizer import SummarizerAgent
from .translator import TranslatorAgent
from .doctor import DoctorAgent
from .psychologist import PsychologistAgent


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
                self.logger.exception(f"✖ {fn.__name__} failed")
                raise
        return wrapper
    return deco


class Orchestrator:
    # ----------------------------------------------------------------------------------------------
    # Initialization

    def __init__(self, log_level: int | None = None, logger_name: str = "kallam.chatbot.orchestrator"):
        """
        log_level: if provided, sets this logger's level; otherwise inherit from parent.
        logger_name: child logger under the manager's logger hierarchy.
        """
        self._setup_logging(log_level, logger_name)

        # Initialize available agents
        self.supervisor = SupervisorAgent()
        self.summarizer = SummarizerAgent()
        self.translator = TranslatorAgent()
        self.doctor = DoctorAgent()
        self.psychologist = PsychologistAgent()

        # Optional config (model names, thresholds, etc.)
        self.config = {
            "default_model": "gpt-4o",
            "translation_model": "gpt-4o",
            "summarization_model": "gpt-4o",
            "doctor_model": "gpt-4o",
            "psychologist_model": "gpt-4o",
            "similarity_threshold": 0.75,
            "supported_languages": {"thai", "english"},
            "agents_language": "english"
        }

        eff = logging.getLevelName(self.logger.getEffectiveLevel())
        self.logger.info(f"KaLLaM agents manager initialized. Effective log level: {eff}")

    def _setup_logging(self, log_level: int | None, logger_name: str) -> None:
        """
        Use a child logger so we inherit handlers/formatters/filters (incl. request_id)
        from the ChatbotManager root logger setup. We do NOT add handlers here.
        """
        self.logger = logging.getLogger(logger_name)
        # Inherit manager's handlers
        self.logger.propagate = True
        # If caller explicitly sets a level, respect it; otherwise leave unset so it inherits.
        if log_level is not None:
            self.logger.setLevel(log_level)

    # ----------------------------------------------------------------------------------------------
    # Supervisor & Routing
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
                eng_message, eng_history, json.dumps(chain_of_thoughts, ensure_ascii=False), json.dumps(summarized_histories, ensure_ascii=False)
            )

        if flags.get("psychologist"):  # Dummy for now
            self.logger.debug("Activating PsychologistAgent")
            commentary["psychologist"] = self.psychologist.analyze(
                original_message, original_history, json.dumps(chain_of_thoughts, ensure_ascii=False), json.dumps(summarized_histories, ensure_ascii=False)
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
