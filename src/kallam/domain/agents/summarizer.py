import os
import json
import logging
import requests
import re
from google import genai
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Literal, Any, Optional

from dotenv import load_dotenv
load_dotenv()


class SummarizerAgent:
    SummaryType = Literal["conversation", "medical_session", "health_insights", "progress_report"]
    SummaryLength = Literal["brief", "detailed", "comprehensive"]

    def __init__(self, log_level: int = logging.INFO):
        self._setup_logging(log_level)
        self._setup_api_clients()
        
        self.logger.info("Summarizer Agent initialized successfully")

        self.system_prompt = """
**Your Role:**
You are a Medical Conversation Summarizer AI specializing in healthcare and mental health conversation analysis. Your role is to create concise, medically-relevant summaries of patient-doctor interactions.

**Core Rules:**
- Summarize healthcare conversations while preserving critical medical information
- Track patient progress and health status changes over time
- Identify key symptoms, treatments, and patient responses
- Maintain patient confidentiality and medical privacy
- Extract actionable health insights from conversation histories
- Organize information chronologically for continuity of care

**Summarization Principles:**
- Preserve all medically relevant information
- Include patient's emotional state and psychological well-being
- Track symptom progression and treatment responses
- Note patient compliance and engagement levels
- Identify patterns in health behaviors and concerns
- Maintain professional medical terminology where appropriate

**Privacy and Ethics:**
- Protect patient confidentiality in all summaries
- Use appropriate medical privacy guidelines
- Avoid including personally identifiable information beyond medical relevance
- Maintain professional boundaries in summary content

**Response Guideline:**
- Create structured summaries with clear timelines
- Use both Thai and English as appropriate to conversation content
- Focus on medical and psychological relevance
- Provide actionable insights for healthcare continuity
- Avoid redundant information from previous summaries

**Language Guidelines:**
- Respond in Thai and English as contextually appropriate
- Use professional medical terminology accurately
- Maintain cultural sensitivity in health communication
- Ensure clarity for healthcare providers and patients

Remember: Your summaries support continuity of care and help healthcare providers understand patient history and progress patterns.
"""

    def _setup_logging(self, log_level: int) -> None:
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        self.logger = logging.getLogger(f"{__name__}.DoctorAgent")
        self.logger.setLevel(log_level)
        if self.logger.handlers:
            self.logger.handlers.clear()
        file_handler = logging.FileHandler(
            log_dir / f"doctor_{datetime.now().strftime('%Y%m%d')}.log",
            encoding='utf-8'
        )
        file_handler.setLevel(logging.DEBUG)
        console_handler = logging.StreamHandler()
        console_handler.setLevel(log_level)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)

    def _setup_api_clients(self) -> None:
        try:
            self.gemini_api_key = os.getenv("GEMINI_API_KEY")
            if not self.gemini_api_key:
                raise ValueError("GEMINI_API_KEY not found in environment variables")
            
            self.gemini_client = genai.Client(api_key=self.gemini_api_key)
            self.gemini_model_name = "gemini-1.5-flash"
            self.logger.info(f"Gemini API client initialized with model: {self.gemini_model_name}")
        except Exception as e:
            self.logger.error(f"Failed to initialize API clients: {str(e)}")
            raise

    def _format_prompt_gemini(self, user_message: str, medical_context: str = "") -> str:
        now = datetime.now()
        context_info = f"""
**Current Context:**
- Date/Time: {now.strftime("%Y-%m-%d %H:%M:%S")}
- Medical Context: {medical_context}
"""

    def _generate_response_gemini(self, prompt: str) -> str:
        try:
            self.logger.debug(f"Sending prompt to Gemini API (length: {len(prompt)} chars)")
            
            response = self.gemini_client.models.generate_content(
                model=self.gemini_model_name,
                contents=[prompt],
            )
            
            response_text = response.text
            
            if response_text is None or (isinstance(response_text, str) and response_text.strip() == ""):
                self.logger.error("Gemini API returned None or empty content")
                return "ขออภัยค่ะ ไม่สามารถสร้างคำตอบได้ในขณะนี้"
            
            # Extract answer block if present
            answer_match = re.search(r"```answer\s*(.*?)\s*```", response_text, re.DOTALL)
            commentary = answer_match.group(1).strip() if answer_match else response_text.strip()

            self.logger.info(f"Generated medical response - Commentary: {len(commentary)} chars")
            return commentary
            
        except Exception as e:
            self.logger.error(f"Error generating Gemini response: {str(e)}")
            return "ขออภัยค่ะ เกิดปัญหาในการเชื่อมต่อ กรุณาลองใหม่อีกครั้งค่ะ"

    def _format_history_for_summarization(self, history: List[Dict[str, str]]) -> str:
        """Format conversation history for summarization"""
        if not history:
            return "No conversation history provided"
        
        formatted_history = []
        for msg in history:
            role = msg.get('role', 'unknown')
            content = msg.get('content', '')
            formatted_history.append(f"{role}: {content}")
        
        return "\n".join(formatted_history)
    
    def summarize_medical_session(
        self,   
        session_history: List[Dict[str, str]], 
        session_type: str = "general",
        focus_areas: Optional[List[str]] = None,
        language: str = "thai"
    ) -> str:
        """
        Summarize a specific medical or psychological session
        
        Args:
            session_history: Session conversation history
            session_type: Type of session (general, emergency, therapy, follow-up)
            focus_areas: Specific areas to focus on in summary
            language: Language for summary output
            
        Returns:
            Session summary
        """

if __name__ == "__main__":
    # Test the Summarizer Agent
    try:
        summarizer = SummarizerAgent(log_level=logging.DEBUG)
        
        # Test conversation summary
        print("=== TEST: CONVERSATION SUMMARY ===")
        chat_history = [
            {"role": "user", "content": "I've been having headaches for the past week"},
            {"role": "assistant", "content": "Tell me more about these headaches - when do they occur and how severe are they?"},
            {"role": "user", "content": "They're usually worse in the afternoon, around 7/10 pain level"},
            {"role": "assistant", "content": "That sounds concerning. Have you noticed any triggers like stress, lack of sleep, or screen time?"}
        ]
        
        summary = summarizer.summarize_conversation_history(
            response_history=chat_history,
            language="english"
        )
        print(summary)
        
        # Test medical session summary
        print("\n=== TEST: MEDICAL SESSION SUMMARY ===")
        session_summary = summarizer.summarize_medical_session(
            session_history=chat_history,
            session_type="consultation",
            focus_areas=["headache assessment", "symptom analysis"],
            language="english"
        )
        print(session_summary)
        
    except Exception as e:
        print(f"Error testing Summarizer Agent: {e}")