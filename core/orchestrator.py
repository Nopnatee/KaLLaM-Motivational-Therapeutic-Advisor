import os
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, List

# Load agents
from agents.summarizer import SummarizerAgent
from agents.translator import TranslatorAgent
from agents.doctor import DoctorAgent
from agents.psychologist import PsychologistAgent
from agents.supervisor import SupervisorAgent
from agents.base_agent import BaseAgent

class Orchestrator:

    def __init__(self, log_level: int = logging.INFO):

        # Setup logging
        self._setup_logging(log_level)

        # Initialize available agents
        self.supervisor = SupervisorAgent()
        self.summarizer = SummarizerAgent()
        self.translator = TranslatorAgent()
        self.doctor = DoctorAgent()
        self.psychologist = PsychologistAgent()
        self.base_agent = BaseAgent()

        # Optional config (model names, thresholds, etc.)
        self.config = {
                        "default_model": "gpt-4o",
                        "translation_model": "gpt-4o",
                        "summarization_model": "gpt-4o",
                        "doctor_model": "gpt-4o",
                        "psychologist_model": "gpt-4o",
                        "similarity_threshold": 0.75,
                      }
        
        self.logger.info(f"KaLLaM agents manager initialized successfully. Log level: {logging.getLevelName(log_level)}")

    def _setup_logging(self, log_level: int) -> None:
        """Setup file + console logging for the orchestrator."""
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)

        self.logger = logging.getLogger(f"{__name__}.AgentsManager")
        self.logger.setLevel(log_level)

        # Remove existing handlers to avoid duplicate logs
        if self.logger.handlers:
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
        if not self.logger.handlers:
            self.logger.addHandler(file_handler)
            self.logger.addHandler(console_handler)

    def get_flags_from_supervisor(self, user_message: str) -> Dict[str, Any]:

        self.logger.info("Getting flags from SupervisorAgent")
        dict_flags = self.supervisor.evaluate(user_message)
        self.logger.debug(f"Supervisor flags: {dict_flags}")

        return dict_flags
    
    def get_translation(self, message: str, flags: dict, translation_type: str) -> str:
        
        translated_message = None

        try:
            translate_flag = flags.get("translate")

            # Forward: Other -> English
            if translation_type == "forward":  # Other -> English
                if translate_flag == "thai":
                    self.logger.debug("Translation flag 'thai' detected, translating to English")
                    translated_message = self.translator.get_translation(message=message, 
                                                                    target="english")
                elif translate_flag == "english":
                    self.logger.debug("Translation flag 'english' detected, using original message")
                    translated_message = message
                elif translate_flag is None:  # no flag set
                    self.logger.debug("No translation flag set, using original message")
                    translated_message = message
                else:
                    raise ValueError(f"Invalid translate flag: {translate_flag}. Allowed values: 'thai', 'english', or None.")
            
            # Backward: English -> Other
            elif translation_type == "backward":  # English -> Other
                if translate_flag == "thai":
                    self.logger.debug("Translation flag 'thai' detected, translating back to Thai")
                    translated_message = self.translator.get_translation(message=message, 
                                                                    target="thai")
                elif translate_flag == "english":
                    self.logger.debug("Translation flag 'english' detected, using original message")
                    translated_message = message
                elif translate_flag is None:  # no flag set
                    self.logger.debug("No translation flag set, using original message")
                    translated_message = message
                else:
                    raise ValueError(f"Invalid translate flag: {translate_flag}. Allowed values: 'thai', 'english', or None.")
            else:
                raise ValueError(f"Invalid translation type: {translation_type}. Allowed values: 'forward' or 'backward'.")
            
        except ValueError as e:
            raise
        except Exception as e:
            self.logger.error(f"Unexpected error in translation for session: {e}", exc_info=True)
            raise RuntimeError("""เกิดข้อผิดพลาดขณะแปล โปรดตรวจสอบให้แน่ใจว่าคุณใช้ภาษาไทยหรือภาษาอังกฤษ แล้วลองอีกครั้ง
                                An error occurred while translating. Please make sure you are using Thai or English and try again.""")
            
        return translated_message

    def get_response(self, 
                     chat_history: List[Dict[str, str]], 
                     user_message: str, 
                     flags: Dict[str, Any],
                     chain_of_thoughts: List[Dict[str, str]],
                     summarized_histories: List[Dict[str, str]]) -> Dict[str, Any]:
        
        self.logger.info(f"Routing message: {user_message} | Flags: {flags}")

        # Prepare current chain of thought container
        commentary = {}

        # Get specialized agents suggestions via flags with chain_of_thoughts
        if flags.get("doctor"):
            self.logger.debug("Activating DoctorAgent")
            commentary["doctor"] = self.doctor.analyze(user_message, 
                                                        chat_history, 
                                                        chain_of_thoughts,
                                                        summarized_histories)

        if flags.get("psychologist"):
            self.logger.debug("Activating PsychologistAgent")
            commentary["psychologist"] = self.psychologist.analyze(user_message, 
                                                                    chat_history, 
                                                                    chain_of_thoughts,
                                                                    summarized_histories)

        # Get final output from all agents' suggestions
        commentary["final_output"] = self.base_agent.conclude(user_message, chain_of_thoughts)
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
    
    def summarize_history(self, 
                          chat_history: List[Dict[str, str]],
                          summarized_histories: List[Dict[str, str]]) -> Optional[Dict[str, Any]]:
        
        try:
            summary = self.summarizer.summarize(chat_history, summarized_histories)
            self.logger.info("Summarization complete.")
        except Exception as e:
            self.logger.error(f"Error during summarization: {e}", exc_info=True)
            summary = None

        return summary