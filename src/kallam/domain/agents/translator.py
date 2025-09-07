# translator_agent.py
# pip install "strands-agents" "boto3" "pydantic" "python-dotenv"

from typing import List, Literal, Dict, Any, Optional
import json
import os
from dotenv import load_dotenv
load_dotenv()  # Load environment variables from a .env file if present

from strands import Agent, tool
from strands.models import BedrockModel
from botocore.config import Config as BotocoreConfig

# Now you can access the environment variables
access_key = os.getenv("AWS_ACCESS_KEY_ID")
secret_key = os.getenv("AWS_SECRET_ACCESS_KEY")
region = os.getenv("AWS_DEFAULT_REGION")

# Use SEA-LION v3.5 for best Thai-English translation performance
MODEL_ID = "aisingapore/Llama-SEA-LION-v3.5-8B-IT"
REGION   = "ap-southeast-2"

GUARDRAIL_ID      = None
GUARDRAIL_VERSION = None

# --------------------------
# Translator Agent
# --------------------------
class TranslatorAgent:
    SupportedLanguage = Literal["thai", "english"]
    TranslationType = Literal["forward", "backward"]

    def __init__(
        self,
        model_id: str = MODEL_ID,
        region: str = REGION,
        guardrail_id: str = None,
        guardrail_version: str = None,
    ):
        boto_cfg = BotocoreConfig(
            retries={"max_attempts": 3, "mode": "standard"},
            connect_timeout=5,
            read_timeout=60,
        )

        bedrock_model = BedrockModel(
            model_id=model_id,
            region_name=region,
            streaming=False,
            temperature=0.1,  # Very low temperature for consistent, accurate translations
            top_p=0.9,
            stop_sequences=["</END>"],

            guardrail_id=guardrail_id,
            guardrail_version=guardrail_version,
            guardrail_trace="enabled",
            guardrail_stream_processing_mode="sync",
            guardrail_redact_input=True,
            guardrail_redacted_input_message="[User input redacted due to privacy policy]",
            guardrail_redact_output=False,

            cache_prompt="default",
            cache_tools="default",
            boto_client_config=boto_cfg,
        )

        system_prompt = """\
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

        self.agent = Agent(
            model=bedrock_model,
            system_prompt=system_prompt,
            tools=[],
            callback_handler=None,
        )

        # Supported languages configuration
        self.supported_languages = {"thai", "english"}
        self.default_thinking_language = "english"

    # --------------------------
    # Public methods
    # --------------------------
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
        - Use appropriate honorifics and polite forms (คะ/ครับ) when culturally expected
        - Maintain proper Thai grammar and sentence structure
        - Use appropriate formality level based on medical context
        
        Special instructions for English translations:
        - Use clear, professional medical English when appropriate
        - Maintain warmth and empathy in healthcare communications
        - Adapt Thai cultural concepts to English-speaking context when necessary
        
        Provide only the translated text without any explanations or additional commentary.
        """
        
        response = self.agent(prompt)
        return response.strip()

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
        
        Respond with only the language name in lowercase: "thai" or "english"
        """
        
        response = self.agent(prompt)
        detected = response.strip().lower()
        
        # Validate and return detected language
        if detected in self.supported_languages:
            return detected
        else:
            # Default to Thai if detection is unclear (common in SEA context)
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
        """
        
        response = self.agent(prompt)
        
        # Parse response into dictionary
        translations = {}
        lines = response.strip().split('\n')
        
        for line in lines:
            if '->' in line:
                original, translated = line.split('->', 1)
                translations[original.strip()] = translated.strip()
        
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
        """
        
        response = self.agent(prompt)
        
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
                key, value = line.split(':', 1)
                key = key.strip().lower()
                value = value.strip()
                
                if key in assessment:
                    assessment[key] = value
                elif key == "issues":
                    if value.lower() != "none":
                        assessment["issues"] = [issue.strip() for issue in value.split(',')]
        
        return assessment