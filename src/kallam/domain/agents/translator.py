import os
import logging
import requests
import re
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional

from dotenv import load_dotenv
load_dotenv()


class TranslatorAgent:
    """
    Handles Thai-English translation using SEA-Lion API.
    """
    
    def __init__(self, log_level: int = logging.INFO):
        self._setup_logging(log_level)
        self._setup_api_client()
        
        self.logger.info("Translator Agent initialized")
        
        # Core configuration
        self.supported_languages = {"thai", "english"}
        self.default_language = "english"

    def _setup_logging(self, log_level: int) -> None:
        """Setup logging configuration"""
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        
        self.logger = logging.getLogger(f"{__name__}.TranslatorAgent")
        self.logger.setLevel(log_level)
        
        if self.logger.handlers:
            self.logger.handlers.clear()
        
        file_handler = logging.FileHandler(
            log_dir / f"translator_{datetime.now().strftime('%Y%m%d')}.log",
            encoding='utf-8'
        )
        file_handler.setLevel(logging.DEBUG)
        
        console_handler = logging.StreamHandler()
        console_handler.setLevel(log_level)
        
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)

    def _setup_api_client(self) -> None:
        """Setup SEA-Lion API client; degrade gracefully if missing."""
        raw_api_key = os.getenv("SEA_LION_API_KEY", "")
        self.api_key = raw_api_key.strip()
        base_url = os.getenv("SEA_LION_BASE_URL") or "https://api.sea-lion.ai/v1"
        self.api_url = base_url.rstrip('/')
        # Enable/disable external API usage based on key presence
        self.enabled = bool(self.api_key)
        if raw_api_key and not self.api_key:
            self.logger.warning("SEA_LION_API_KEY contained only whitespace after stripping")
        if self.enabled:
            self.logger.info("SEA-Lion API client initialized")
        else:
            # Do NOT raise: allow app to start and operate in passthrough mode
            self.logger.warning(
                "SEA_LION_API_KEY not set. Translator will run in passthrough mode (no external calls)."
            )

    def _call_api(self, text: str, target_language: str) -> str:
        """
        Simple API call to SEA-Lion for translation
        
        Args:
            text: Text to translate
            target_language: Target language ("thai" or "english")
            
        Returns:
            Translated text or original on error
        """
        # Short-circuit if API disabled
        if not getattr(self, "enabled", False):
            return text

        try:
            # Build simple translation prompt
            system_prompt = f"""
**Your Role:**
You are a translator for a chatbot which is used for medical and psychological help. 

**Core Rules:**
- Translate the given text to {target_language}.
- Provide ONLY the translation without quotes or explanations.
- Maintain medical/psychological terminology accuracy.
- For Thai: use appropriate polite forms.
- For English: use clear, professional language."""

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Translate: {text}"}
            ]
            
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "model": "aisingapore/Gemma-SEA-LION-v4-27B-IT",
                "messages": messages,
                "chat_template_kwargs": {"thinking_mode": "off"},
                "max_tokens": 2000,
                "temperature": 0.1,  # Low for consistency
                "top_p": 0.9
            }
            
            response = requests.post(
                f"{self.api_url}/chat/completions",
                headers=headers,
                json=payload,
                timeout=30
            )
            
            response.raise_for_status()
            data = response.json()
            
            # Extract response
            content = data["choices"][0]["message"]["content"]
            
            # Remove thinking blocks if present
            if "```thinking" in content and "```answer" in content:
                match = re.search(r"```answer\s*(.*?)\s*```", content, re.DOTALL)
                if match:
                    content = match.group(1).strip()
            elif "</think>" in content:
                content = re.sub(r".*?</think>\s*", "", content, flags=re.DOTALL).strip()
            
            # Clean up quotes
            content = content.strip()
            if content.startswith('"') and content.endswith('"'):
                content = content[1:-1]
            
            return content if content else text
            
        except requests.exceptions.RequestException as e:
            status_code = getattr(getattr(e, 'response', None), 'status_code', None)
            body_preview = None
            if getattr(e, 'response', None) is not None:
                try:
                    body_preview = e.response.text[:500]
                except Exception:
                    body_preview = '<unavailable>'
            request_url = getattr(getattr(e, 'request', None), 'url', None)
            self.logger.error(
                "SEA-Lion translation request failed (%s) url=%s status=%s body=%s message=%s",
                e.__class__.__name__,
                request_url,
                status_code,
                body_preview,
                str(e),
            )
            return text
        except Exception as e:
            self.logger.error(f"Translation API error: {str(e)}")
            return text  # Return original on error

    # ===== PUBLIC INTERFACE (Used by Orchestrator) =====
    
    def get_translation(self, message: str, target: str) -> str:
        """
        Main translation method used by orchestrator
        """
        # Validate target
        if target not in self.supported_languages:
            self.logger.warning(f"Unsupported language '{target}', returning original")
            return message
        
        # Skip if empty
        if not message or not message.strip():
            return message
        
        self.logger.debug(f"Translating to {target}: {message[:50]}...")
        
        translated = self._call_api(message, target)
        
        self.logger.debug(f"Translation complete: {translated[:50]}...")
        
        return translated

    def detect_language(self, message: str) -> str:
        """
        Detect language of the message
        """
        # Check for Thai characters
        thai_chars = sum(1 for c in message if '\u0E00' <= c <= '\u0E7F')
        total_chars = len(message.strip())
        
        if total_chars == 0:
            return "english"
        
        thai_ratio = thai_chars / total_chars
        
        # If more than 10% Thai characters, consider it Thai
        if thai_ratio > 0.1:
            return "thai"
        else:
            return "english"

    def translate_forward(self, message: str, source_language: str) -> str:
        """
        Forward translation to English (for agent processing)
        
        Args:
            message: Text in source language
            source_language: Source language
            
        Returns:
            English text
        """
        if source_language == "english":
            return message
        
        return self.get_translation(message, "english")

    def translate_backward(self, message: str, target_language: str) -> str:
        """
        Backward translation from English to user's language
        
        Args:
            message: English text
            target_language: User's language
            
        Returns:
            Text in user's language
        """
        if target_language == "english":
            return message
            
        return self.get_translation(message, target_language)


if __name__ == "__main__":
    # Simple test
    try:
        translator = TranslatorAgent(log_level=logging.DEBUG)
        
        print("=== SIMPLIFIED TRANSLATOR TEST ===\n")
        
        # Test 1: Basic translations
        test_cases = [
            ("ฉันปวดหัวมาก", "english"),
            ("I have a headache", "thai"),
            ("รู้สึกเครียดและกังวล", "english"),
            ("Feeling anxious about my health", "thai")
        ]
        
        for text, target in test_cases:
            result = translator.get_translation(text, target)
            print(f"{text[:30]:30} -> {target:7} -> {result}")
        
        # Test 2: Language detection
        print("\n=== Language Detection ===")
        texts = [
            "สวัสดีครับ",
            "Hello there",
            "ผมชื่อ John",
            "I'm feeling ดีมาก"
        ]
        
        for text in texts:
            detected = translator.detect_language(text)
            print(f"{text:20} -> {detected}")
            
    except Exception as e:
        print(f"Test error: {e}")
