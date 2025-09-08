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
    TranslationType = Literal["forward", "backward"]

    def __init__(self, log_level: int = logging.INFO):
        self._setup_logging(log_level)
        self._setup_api_clients()
        
        self.logger.info("Translator Agent initialized successfully")

        # Supported languages configuration
        self.supported_languages = {"thai", "english"}
        self.default_thinking_language = "english"

        self.system_prompt = """
You are a Professional Thai-English Translator AI specializing in accurate, contextually appropriate translations between Thai and English languages. You are specifically optimized for Southeast Asian language nuances and cultural contexts.

**Your Core Capabilities:**
- High-quality bidirectional translation between Thai and English
- Preservation of medical and healthcare terminology accuracy
- Cultural context and idiomatic expression translation
- Formal and informal register adaptation
- Technical terminology handling for medical/psychological content

**Translation Principles:**
1. **Accuracy First**: Maintain the exact meaning of the original text
2. **Cultural Sensitivity**: Adapt cultural references and idioms appropriately
3. **Register Matching**: Preserve the formality level of the original
4. **Medical Precision**: Ensure medical terms are translated accurately
5. **Natural Flow**: Make translations sound natural in the target language

**Supported Languages:**
- **Thai (thai)**: Native Thai language with proper tone markers and cultural context
- **English (english)**: Standard English with appropriate register and style

**Translation Types:**
- **Forward Translation**: From user's native language to English (thinking language)
- **Backward Translation**: From English back to user's native language

**Special Handling:**
- **Medical Content**: Preserve medical accuracy while ensuring patient understanding
- **Psychological Terms**: Maintain therapeutic terminology precision
- **Cultural Context**: Adapt cultural references and honorifics appropriately
- **Emotional Tone**: Preserve the emotional nuance of the original message

**Quality Assurance:**
- Always double-check medical and psychological terminology
- Ensure natural language flow in target language
- Maintain consistent terminology throughout conversations
- Preserve the speaker's intent and emotional tone

**Output Requirements:**
- Provide only the translated text without explanations
- Maintain original formatting and structure
- Preserve names, dates, and technical terms appropriately
- Ensure culturally appropriate language use

Remember: Your translations support healthcare communication, so accuracy and cultural sensitivity are paramount for effective patient care.
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

    def _format_messages(self, prompt: str, context: str = "") -> List[Dict[str, str]]:
        """Format messages for API call"""
        now = datetime.now()
        
        context_info = f"""
**Current Context:**
- Date/Time: {now.strftime("%Y-%m-%d %H:%M:%S")}
- Translation Context: {context}
"""
        
        system_message = {
            "role": "system",
            "content": f"{self.system_prompt}\n\n{context_info}"
        }
        
        user_message = {
            "role": "user",
            "content": prompt
        }
        
        return [system_message, user_message]

    def _generate_response(self, messages: List[Dict[str, str]]) -> str:
        """Generate response using SEA-Lion API"""
        try:
            self.logger.debug(f"Sending {len(messages)} messages to SEA-Lion API")
            
            headers = {
                "Authorization": f"Bearer {self.sea_lion_api_key}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "model": "aisingapore/Llama-SEA-LION-v3.5-8B-R",
                "messages": messages,
                "chat_template_kwargs": {
                    "thinking_mode": "on"
                },
                "max_tokens": 1500,
                "temperature": 0.1,  # Very low temperature for consistent, accurate translations
                "top_p": 0.9,
                "frequency_penalty": 0.1,
                "presence_penalty": 0.1
            }
            
            response = requests.post(
                f"{self.sea_lion_base_url}/chat/completions",
                headers=headers,
                json=payload,
                timeout=30
            )
            
            response.raise_for_status()
            response_data = response.json()
            
            if "choices" not in response_data or len(response_data["choices"]) == 0:
                self.logger.error(f"Unexpected response structure: {response_data}")
                return "ไม่สามารถแปลได้ในขณะนี้"
            
            choice = response_data["choices"][0]
            if "message" not in choice or "content" not in choice["message"]:
                self.logger.error(f"Unexpected message structure: {choice}")
                return "ไม่สามารถแปลได้ในขณะนี้"
                
            raw_content = choice["message"]["content"]
            
            if raw_content is None or (isinstance(raw_content, str) and raw_content.strip() == ""):
                self.logger.error("SEA-Lion API returned None or empty content")
                return "ไม่สามารถแปลได้ในขณะนี้"
            
            # Extract thinking and answer blocks
            thinking_match = re.search(r"```thinking\s*(.*?)\s*```", raw_content, re.DOTALL)
            answer_match = re.search(r"```answer\s*(.*?)\s*```", raw_content, re.DOTALL)
            
            reasoning = thinking_match.group(1).strip() if thinking_match else None
            final_answer = answer_match.group(1).strip() if answer_match else raw_content.strip()
            
            if reasoning:
                self.logger.debug(f"Translator reasoning:\n{reasoning}")
            
            self.logger.info(f"Generated translation (length: {len(final_answer)} chars)")
            return final_answer
            
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Error generating response from SEA-Lion API: {str(e)}")
            return "เกิดปัญหาในการเชื่อมต่อ กรุณาลองใหม่อีกครั้งค่ะ"
        except Exception as e:
            self.logger.error(f"Error generating response: {str(e)}")
            return "เกิดข้อผิดพลาดในการแปล"

    # Public methods
    def get_translation(self, message: str, target: str) -> str:
        """
        Main translation method that matches the orchestrator's expected interface
        
        Args:
            message: Text to translate
            target: Target language ("thai" or "english")
            
        Returns:
            Translated text
        """
        
        # Validate target language
        if target not in self.supported_languages:
            raise ValueError(f"Unsupported target language: {target}. Supported: {list(self.supported_languages)}")
        
        # Detect source language and perform translation
        return self._translate_message(message, target)
    
    def _translate_message(self, message: str, target_language: str) -> str:
        """
        Internal method to handle the actual translation
        
        Args:
            message: Text to translate
            target_language: Target language for translation
            
        Returns:
            Translated text
        """
        
        prompt = f"""
Please translate the following text to {target_language}:

Text to translate: "{message}"
Target language: {target_language}

Translation requirements:
1. Maintain exact meaning and intent of the original text
2. Use natural, fluent language in the target language
3. Preserve any medical or technical terminology accuracy
4. Maintain appropriate formality level and cultural context
5. If the text contains Thai cultural references, adapt them appropriately for English and vice versa
6. For medical/healthcare content, ensure terminology is accurate but understandable
7. Preserve emotional tone and nuance of the original message

Special instructions for Thai translations:
- Use appropriate honorifics and polite forms (ค่ะ/ครับ) when culturally expected
- Maintain proper Thai grammar and sentence structure
- Use appropriate formality level based on medical context

Special instructions for English translations:
- Use clear, professional medical English when appropriate
- Maintain warmth and empathy in healthcare communications
- Adapt Thai cultural concepts to English-speaking context when necessary

**Important**: Provide ONLY the translated text without any explanations, quotation marks, or additional commentary. Return the pure translation result.
"""
        
        context = f"Translation from auto-detected language to {target_language}"
        messages = self._format_messages(prompt, context)
        
        response = self._generate_response(messages)
        
        # Clean up the response to ensure we only return the translation
        # Remove any potential quotation marks or extra formatting
        cleaned_response = response.strip()
        if cleaned_response.startswith('"') and cleaned_response.endswith('"'):
            cleaned_response = cleaned_response[1:-1]
        
        return cleaned_response

    def detect_language(self, message: str) -> str:
        """
        Detect the language of the input message
        
        Args:
            message: Text to analyze for language detection
            
        Returns:
            Detected language ("thai" or "english")
        """
        
        prompt = f"""
Analyze the following text and determine its primary language.

Text to analyze: "{message}"

Instructions:
1. Identify if the text is primarily in Thai or English
2. Consider mixed language texts and determine the dominant language
3. Look for Thai script characters, tone markers, and Thai-specific words
4. Consider sentence structure and grammar patterns
5. Respond with only one word: either "thai" or "english"

Language detection criteria:
- Thai: Contains Thai script (อักษรไทย), Thai words, Thai grammar patterns
- English: Uses Latin alphabet, English words and grammar patterns
- Mixed: Determine which language is more dominant in the text

**Important**: Respond with ONLY the language name in lowercase: "thai" or "english". No explanations or additional text.
"""
        
        context = "Language detection analysis"
        messages = self._format_messages(prompt, context)
        
        response = self._generate_response(messages)
        detected = response.strip().lower()
        
        # Validate and return detected language
        if detected in self.supported_languages:
            return detected
        else:
            # Default to Thai if detection is unclear (common in SEA context)
            self.logger.warning(f"Language detection unclear: {detected}. Defaulting to Thai.")
            return "thai"

    def translate_forward(self, message: str, source_language: str) -> str:
        """
        Forward translation: from user's native language to English (thinking language)
        
        Args:
            message: Text in user's native language
            source_language: Source language of the message
            
        Returns:
            Text translated to English for processing
        """
        
        if source_language == self.default_thinking_language:
            return message  # No translation needed
        
        self.logger.debug(f"Forward translation from {source_language} to {self.default_thinking_language}")
        return self.get_translation(message, self.default_thinking_language)

    def translate_backward(self, message: str, target_language: str) -> str:
        """
        Backward translation: from English (thinking language) back to user's native language
        
        Args:
            message: Text in English (thinking language)
            target_language: User's native language
            
        Returns:
            Text translated back to user's native language
        """
        
        if target_language == self.default_thinking_language:
            return message  # No translation needed
            
        self.logger.debug(f"Backward translation from {self.default_thinking_language} to {target_language}")
        return self.get_translation(message, target_language)

    def translate_medical_terms(self, terms: List[str], target_language: str) -> Dict[str, str]:
        """
        Specialized method for translating medical terminology with high accuracy
        
        Args:
            terms: List of medical terms to translate
            target_language: Target language for translation
            
        Returns:
            Dictionary mapping original terms to translations
        """
        
        terms_text = ", ".join(terms)
        
        prompt = f"""
Translate the following medical terms to {target_language} with high precision:

Medical terms: {terms_text}
Target language: {target_language}

Requirements:
1. Use accurate medical terminology in the target language
2. Maintain professional medical accuracy
3. Ensure terms are understandable to patients when appropriate
4. Preserve clinical precision and meaning
5. Use standard medical translations used in {target_language} healthcare systems

Format your response as a simple list, one term per line:
original_term -> translated_term

For example:
headache -> ปวดหัว
blood pressure -> ความดันโลหิต

**Important**: Provide ONLY the term translations in the specified format. No explanations or additional text.
"""
        
        context = f"Medical terminology translation to {target_language}"
        messages = self._format_messages(prompt, context)
        
        response = self._generate_response(messages)
        
        # Parse response into dictionary
        translations = {}
        lines = response.strip().split('\n')
        
        for line in lines:
            if '->' in line:
                try:
                    original, translated = line.split('->', 1)
                    translations[original.strip()] = translated.strip()
                except ValueError:
                    self.logger.warning(f"Could not parse translation line: {line}")
                    continue
        
        return translations

    def validate_translation_quality(self, original: str, translated: str, target_language: str) -> Dict[str, Any]:
        """
        Validate the quality of a translation
        
        Args:
            original: Original text
            translated: Translated text
            target_language: Target language of translation
            
        Returns:
            Dictionary with quality assessment
        """
        
        prompt = f"""
Evaluate the quality of this translation:

Original text: "{original}"
Translated text: "{translated}"
Target language: {target_language}

Assessment criteria:
1. Accuracy: Does the translation preserve the original meaning?
2. Fluency: Does the translation sound natural in the target language?
3. Cultural appropriateness: Are cultural elements properly adapted?
4. Medical accuracy: If medical content, is terminology correct?
5. Completeness: Is all information from original preserved?

Provide assessment in the following format:
Accuracy: [High/Medium/Low]
Fluency: [High/Medium/Low]
Cultural: [Appropriate/Needs adjustment]
Medical: [Accurate/Needs review/Not applicable]
Overall: [Excellent/Good/Needs improvement]
Issues: [List any specific problems or "None"]

**Important**: Use exactly the format specified above. Be concise and specific.
"""
        
        context = f"Translation quality assessment for {target_language} translation"
        messages = self._format_messages(prompt, context)
        
        response = self._generate_response(messages)
        
        # Parse the response into a structured assessment
        assessment = {
            "accuracy": "Medium",
            "fluency": "Medium", 
            "cultural": "Appropriate",
            "medical": "Not applicable",
            "overall": "Good",
            "issues": []
        }
        
        # Simple parsing of the response
        lines = response.strip().split('\n')
        for line in lines:
            if ':' in line:
                try:
                    key, value = line.split(':', 1)
                    key = key.strip().lower()
                    value = value.strip()
                    
                    if key == "accuracy":
                        assessment["accuracy"] = value
                    elif key == "fluency":
                        assessment["fluency"] = value
                    elif key == "cultural":
                        assessment["cultural"] = value
                    elif key == "medical":
                        assessment["medical"] = value
                    elif key == "overall":
                        assessment["overall"] = value
                    elif key == "issues":
                        if value.lower() != "none":
                            assessment["issues"] = [issue.strip() for issue in value.split(',')]
                except ValueError:
                    self.logger.warning(f"Could not parse assessment line: {line}")
                    continue
        
        return assessment

    def batch_translate(self, messages: List[str], target_language: str) -> List[str]:
        """
        Translate multiple messages in batch for efficiency
        
        Args:
            messages: List of messages to translate
            target_language: Target language for all translations
            
        Returns:
            List of translated messages in the same order
        """
        
        if not messages:
            return []
        
        # For small batches, translate individually for better quality
        if len(messages) <= 3:
            return [self.get_translation(msg, target_language) for msg in messages]
        
        # For larger batches, use batch processing
        numbered_messages = "\n".join([f"{i+1}. {msg}" for i, msg in enumerate(messages)])
        
        prompt = f"""
Translate the following numbered messages to {target_language}:

{numbered_messages}

Requirements:
1. Translate each message accurately while preserving meaning
2. Maintain the numbered format in your response
3. Use natural, fluent language in the target language
4. Preserve medical/technical terminology accuracy
5. Maintain emotional tone and context

**Important**: Provide ONLY the numbered translated messages in the same format. No explanations or additional text.

Format your response as:
1. [translated message 1]
2. [translated message 2]
...and so on
"""
        
        context = f"Batch translation of {len(messages)} messages to {target_language}"
        messages_formatted = self._format_messages(prompt, context)
        
        response = self._generate_response(messages_formatted)
        
        # Parse the numbered response back into a list
        translations = []
        lines = response.strip().split('\n')
        
        for line in lines:
            if re.match(r'^\d+\.', line.strip()):
                # Remove the number prefix and add to translations
                translated = re.sub(r'^\d+\.\s*', '', line.strip())
                translations.append(translated)
        
        # Ensure we have the same number of translations as input messages
        while len(translations) < len(messages):
            translations.append("Translation unavailable")
        
        return translations[:len(messages)]


if __name__ == "__main__":
    # Test the Translator Agent
    try:
        translator = TranslatorAgent(log_level=logging.DEBUG)
        
        # Test language detection
        print("=== TEST: LANGUAGE DETECTION ===")
        thai_text = "ฉันปวดหัวมาก"
        english_text = "I have a severe headache"
        
        detected_thai = translator.detect_language(thai_text)
        detected_english = translator.detect_language(english_text)
        
        print(f"Thai text '{thai_text}' detected as: {detected_thai}")
        print(f"English text '{english_text}' detected as: {detected_english}")
        
        # Test translation
        print("\n=== TEST: TRANSLATION ===")
        thai_to_english = translator.get_translation(thai_text, "english")
        english_to_thai = translator.get_translation(english_text, "thai")
        
        print(f"Thai to English: '{thai_text}' -> '{thai_to_english}'")
        print(f"English to Thai: '{english_text}' -> '{english_to_thai}'")
        
        # Test medical terms translation
        print("\n=== TEST: MEDICAL TERMS ===")
        medical_terms = ["headache", "blood pressure", "anxiety", "depression"]
        translated_terms = translator.translate_medical_terms(medical_terms, "thai")
        
        print("Medical terms translation:")
        for original, translated in translated_terms.items():
            print(f"  {original} -> {translated}")
        
    except Exception as e:
        print(f"Error testing Translator Agent: {e}")