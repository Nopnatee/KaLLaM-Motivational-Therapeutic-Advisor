import os
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, List

# Load agents
from core.agents.supervisor import SupervisorAgent
from core.agents.summarizer import SummarizerAgent
from core.agents.translator import TranslatorAgent
from core.agents.doctor import DoctorAgent
from core.agents.psychologist import PsychologistAgent

class Orchestrator:

    # ----------------------------------------------------------------------------------------------
    # Initialization

    def __init__(self, log_level: int = logging.INFO):

        # Setup logging
        self._setup_logging(log_level)

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
        
        self.logger.info(f"KaLLaM agents manager initialized successfully. Log level: {logging.getLevelName(log_level)}")

    def _setup_logging(self, log_level: int) -> None:
        """Setup file + console logging for the orchestrator."""
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)

        self.logger = logging.getLogger(f"{__name__}.AgentsManager")
        self.logger.setLevel(log_level)

        # Remove existing handlers to avoid duplicate logs
        self.logger.handlers.clear()

        # File handler for detailed logs
        file_handler = logging.FileHandler(
            log_dir / f"kallam_{datetime.now().strftime('%Y%m%d')}.log",
            encoding="utf-8"
        )
        file_handler.setLevel(logging.DEBUG)

        # Console handler for immediate feedback
        console_handler = logging.StreamHandler()
        console_handler.setLevel(log_level)

        # Formatter
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        # Add handlers
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)

    # ----------------------------------------------------------------------------------------------
    # Supervisor & Routing
    def get_flags_from_supervisor(
            self, 
            chat_history: List[Dict[str, str]], 
            user_message: str, 
            memory_context: str, 
            task: str, 
            summarized_histories: Optional[List] = None
            ) -> Dict[str, Any]:

        self.logger.info("Getting flags from SupervisorAgent")
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
    
    # Translation handling
    def get_translation(self, message: str, flags: dict, translation_type: str) -> str:
        """Translate message based on flags and translation type."""
        try:
            source_lang = flags.get("language")
            default_target = self.config["agents_language"]
            supported_langs = self.config["supported_languages"]
            
            # Validate translation type
            if translation_type not in {"forward", "backward"}:
                raise ValueError(f"Invalid translation type: {translation_type}. Allowed: 'forward', 'backward'")
            
            # No translation needed if no source language
            if source_lang is None:
                self.logger.debug("No translation flag set, using original message")
                return message
            
            # Validate source language
            if source_lang not in supported_langs:
                supported = ", ".join(f"'{lang}'" for lang in supported_langs)
                raise ValueError(f"Invalid translate flag: {source_lang}. Supported: {supported}")
            
            # Determine target and if translation needed
            if translation_type == "forward":
                target_lang = default_target if source_lang != default_target else source_lang
                needs_translation = source_lang != default_target
            else:  # backward
                target_lang = source_lang if source_lang != default_target else default_target
                needs_translation = source_lang != default_target
            
            # Log and execute
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

    # Main response generation logic
    def get_response(self, 
                     chat_history: List[Dict[str, str]], 
                     user_message: str, 
                     flags: Dict[str, Any],
                     chain_of_thoughts: List[Dict[str, str]],
                     memory_context: Optional[Dict[str, Any]],
                     summarized_histories: List[Dict[str, str]]) -> Dict[str, Any]:
        
        self.logger.info(f"Routing message: {user_message} | Flags: {flags}")

        # Prepare current chain of thought container
        commentary = {}

        # Get specialized agents suggestions via flags with chain_of_thoughts
        if flags.get("doctor"): # Dummy for now
            self.logger.debug("Activating DoctorAgent")
            commentary["doctor"] = self.doctor.analyze(user_message, 
                                                        chat_history, 
                                                        chain_of_thoughts,
                                                        summarized_histories)

        if flags.get("psychologist"): # Dummy for now
            self.logger.debug("Activating PsychologistAgent")
            commentary["psychologist"] = self.psychologist.analyze(user_message, 
                                                                    chat_history, 
                                                                    chain_of_thoughts,
                                                                    summarized_histories)

        # Get final output from all agents' suggestions
        commentary["final_output"] = self.supervisor.generate_feedback(
            chat_history=chat_history, 
            user_message=user_message,
            memory_context=memory_context, 
            task="flag", 
            summarized_histories=summarized_histories,
            commentary=commentary,
            )
        self.logger.info("Routing complete. Returning results.")

        return commentary

    # Dummy for now
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
    
    # Summarization handling
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