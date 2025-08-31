import os
import json
import logging
import requests
import re
from google import genai
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, List

# Configure logging
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load agents
from agents.summarizer import SummarizerAgent
from agents.translator import TranslatorAgent
from agents.doctor import DoctorAgent
from agents.psychologist import PsychologistAgent
from agents.supervisor import SupervisorAgent

config = {
    # Example config options
    "default_model": "gpt-4o",
    "translation_model": "gpt-4o",
    "summarization_model": "gpt-4o",
    "doctor_model": "gpt-4o",
    "psychologist_model": "gpt-4o",
    "similarity_threshold": 0.75,
    # Add more as needed
}

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

        # Optional config (model names, thresholds, etc.)
        self.config = config or {}
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
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)

    def get_flags_from_supervisor(self, user_message: str) -> dict:
        """
        Use the SupervisorAgent to determine which agents to activate.
        Args:
            user_message (str): The raw user input.
        Returns:
            dict: Flags indicating which agents to activate.
        """
        self.logger.info("Getting flags from SupervisorAgent")
        dict_flags = self.supervisor.evaluate(user_message)
        self.logger.debug(f"Supervisor flags: {dict_flags}")
        return dict_flags
    
    def get_translation(self, message: str, flags: dict, type: str) -> str:
        """
        Handle translation requests.
        Args:
            message (str): The text to translate.
            flags (dict): Flags indicating translation needs.
        Returns:
            str: Translated text.
        """
        translate_flag = flags.get("translate")

        try:
            if type == "forward":  # Other -> English
                if translate_flag == "thai":
                    logger.debug("Translation flag 'thai' detected, translating to English")
                    translated_message = self.translator.get_translation(message=message, 
                                                                    target="english")
                elif translate_flag == "english":
                    logger.debug("Translation flag 'english' detected, using original message")
                    return message
                elif translate_flag is None:  # no flag set
                    logger.debug("No translation flag set, using original message")
                    return message
                else:
                    # 🚨 Anything else = error
                    raise ValueError(f"Invalid translate flag: {translate_flag}. Allowed values: 'thai', 'english', or None.")
            
            elif type == "backward":  # English -> Other
                if translate_flag == "thai":
                    logger.debug("Translation flag 'thai' detected, translating back to Thai")
                    translated_message = self.translator.get_translation(message=message, 
                                                                    target="thai")
                elif translate_flag == "english":
                    logger.debug("Translation flag 'english' detected, using original message")
                    return message
                elif translate_flag is None:  # no flag set
                    logger.debug("No translation flag set, using original message")
                    return message
                else:
                    # 🚨 Anything else = error
                    raise ValueError(f"Invalid translate flag: {translate_flag}. Allowed values: 'thai', 'english', or None.")
            else:
                raise ValueError(f"Invalid translation type: {type}. Allowed values: 'forward' or 'backward'.")
            
        except Exception as e:
            logger.error(f"Error translating via translate flags for session: {e}", exc_info=True)
            raise ValueError("""เกิดข้อผิดพลาดขณะแปล โปรดตรวจสอบให้แน่ใจว่าคุณใช้ภาษาไทยหรือภาษาอังกฤษ แล้วลองอีกครั้ง
                                An error occurred while translating. Please make sure you are using Thai or English and try again.""")
    
        return translated_message

    def get_response(self, user_message: str, flags: dict) -> dict:
        """
        Main orchestration logic.
        Args:
            user_message (str): The raw user input.
            flags (dict): Activation signals, e.g.,
                {
                    "translate": "thai",   # force translation
                    "summarize": True,
                    "doctor": False,
                    "psychologist": True
                }
        Returns:
            dict: Structured response with agent outputs.
        """
        self.logger.info(f"Routing message: {user_message} | Flags: {flags}")

        response_bundle = {}
        working_text = user_message

        # 1. Handle translation first if needed
        if flags.get("translate") == "thai":
            self.logger.debug("Activating TranslatorAgent for Thai → English")
            translated_text = self.translator.translate(user_message, target_lang="en")
            response_bundle["translator"] = translated_text
            working_text = translated_text

        # 2. Summarizer
        if flags.get("summarize"):
            self.logger.debug("Activating SummarizerAgent")
            summary = self.summarizer.summarize(working_text)
            response_bundle["summarizer"] = summary
            working_text = summary  # optional

        # 3. Specialized agents
        if flags.get("doctor"):
            self.logger.debug("Activating DoctorAgent")
            response_bundle["doctor"] = self.doctor.analyze(working_text)

        if flags.get("psychologist"):
            self.logger.debug("Activating PsychologistAgent")
            response_bundle["psychologist"] = self.psychologist.listen(working_text)

        # 4. Merge and return results
        response_bundle["final_output"] = self._merge_outputs(response_bundle)
        self.logger.info("Routing complete. Returning results.")
        return response_bundle

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