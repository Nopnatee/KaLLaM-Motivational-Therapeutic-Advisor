import os
import json
import logging
import requests
import re
from pathlib import Path
from datetime import datetime
from typing import List, Literal, Dict, Any, Optional

from dotenv import load_dotenv
load_dotenv()


class TranslatorAgent:
    SupportedLanguage = Literal["thai", "english"]
    TranslationType = Literal["backward"]

    def __init__(self, log_level: int = logging.INFO):
        self._setup_logging(log_level)
        self._setup_api_clients()
        
        self.logger.info("Translator Agent initialized successfully")

        # Supported languages configuration
        self.supported_languages = {"thai", "english"}
        self.default_thinking_language = "thai"

        self.system_prompt = """
You are a Professional Thai-English Translator AI specializing in accurate, contextually appropriate translations from English to Thai languages. 
You are specifically optimized for Thai language nuances and cultural contexts.

**Your Core Capabilities:**
- High-quality translation from English to Thai
- Preservation of medical and psychology terminology accuracy
- Cultural context and idiomatic expression translation

**Example Input:**
{"conv_id":"Dummy","conversation":"Bot: Dummy\nBot: Dummy\nBot: thank you so much!\nBot: No problem. What kind of work does he do?\nBot: he is an armed guard","context":"hopeful","prompt":"i really hope my husband finds a full time job soon"}


**Translation Principles:**
1. **Accuracy First**: Maintain the exact emotional nuance, meanings, and conjunctions of the original text
2. **Medical Precision**: Ensure medical terms are translated accurately
3. **Natural Flow**: Recheck for naturality of the response based on Thai cultural and wordings

**Your Response Restrictions:**
- The input have User: and Bot: 
- Use exact original special characters formatting and structure with replaced Thai texts without any comments
"""

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

    def _setup_api_clients(self) -> None:
        """Setup API clients"""
        try:
            self.sea_lion_api_key = os.getenv("SEA_LION_API_KEY")
            self.sea_lion_base_url = os.getenv("SEA_LION_BASE_URL", "https://api.sea-lion.ai/v1")
            
            if not self.sea_lion_api_key:
                raise ValueError("SEA_LION_API_KEY not provided and not found in environment variables")
                
            self.logger.info("SEA-Lion API client initialized for Translator Agent")
                
        except Exception as e:
            self.logger.error(f"Failed to initialize API clients: {str(e)}")
            raise

    def _format_messages(self, content: str) -> List[Dict[str, str]]:
        """Format messages for API call"""
        
        system_message = {"role": "system", "content": self.system_prompt}
        user_message = {"role": "user", "content": content}
        
        return [system_message, user_message]

    def _generate_response(self, messages: List[Dict[str, str]]) -> str:
        try:
            self.logger.debug(f"Sending {len(messages)} messages to SEA-Lion API")
            headers = {
                "Authorization": f"Bearer {self.sea_lion_api_key}",
                "Content-Type": "application/json"
            }
            payload = {
                "model": "aisingapore/Gemma-SEA-LION-v4-27B-IT",
                "messages": messages,
                "max_tokens": 1500,
                "temperature": 0.35,
                "top_p": 1.0,
                "frequency_penalty": 0.0,
                "presence_penalty": 0.0,
                "cache": {"no-cache": True}  # remove in prod
            }

            for attempt in range(3):
                resp = requests.post(
                    f"{self.sea_lion_base_url}/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=30
                )
                if resp.status_code in (429, 500, 502, 503, 504) and attempt < 2:
                    self.logger.warning(f"Retryable status {resp.status_code}; retry {attempt+1}/2")
                    continue
                resp.raise_for_status()
                data = resp.json()
                break

            choices = data.get("choices") or []
            if not choices:
                self.logger.error(f"Unexpected response structure: {data}")
                return "ไม่สามารถแปลได้ในขณะนี้"

            content = (choices[0].get("message") or {}).get("content") or ""
            content = content.strip()
            if not content:
                self.logger.error("SEA-Lion API returned empty content")
                return "ไม่สามารถแปลได้ในขณะนี้"

            self.logger.info(f"Generated translation (length: {len(content)} chars)")
            return content

        except requests.exceptions.RequestException as e:
            self.logger.error(f"Error generating response from SEA-Lion API: {str(e)}")
            return "เกิดปัญหาในการเชื่อมต่อ กรุณาลองใหม่อีกครั้งค่ะ"
        except Exception as e:
            self.logger.error(f"Error generating response: {str(e)}")
            return "เกิดข้อผิดพลาดในการแปล"

    def translate_backward(self, user_message: str) -> str:
        """Translate from English to Thai"""
        messages = self._format_messages(user_message)
        response = self._generate_response(messages)

        self.logger.debug(f"Translation from English to {self.default_thinking_language}")
        return response

if __name__ == "__main__":
    # Test the Translator Agent
    try:
        translator = TranslatorAgent(log_level=logging.DEBUG)
        print("\n=== TEST: TRANSLATION ===")

        english_text = """{"conv_id":"hit:0_conv:0","conversation":"User: Yeah about 10 years ago I had a horrifying experience. It was 100% their fault but they hit the water barrels and survived. They had no injuries but they almost ran me off the road.\nBot: Did you suffer any injuries?\nUser: No I wasn't hit. It turned out they were drunk. I felt guilty but realized it was his fault.\nBot: Why did you feel guilty? People really shouldn't drive drunk.\nUser: I don't know I was new to driving and hadn't experienced anything like that. I felt like my horn made him swerve into the water barrels.","context":"guilty","prompt":"I felt guilty when I was driving home one night and a person tried to fly into my lane, and didn't see me. I honked and they swerved back into their lane, slammed on their brakes, and hit the water cones."}"""
        
        # Test translation
        english_to_thai = translator.translate_backward(english_text)
        print(f"English to Thai: '{english_text}' -> '{english_to_thai}'")
        
    except Exception as e:
        print(f"Error testing Translator Agent: {e}")