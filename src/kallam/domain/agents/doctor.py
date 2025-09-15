import os
import logging
import requests
import re
from google import genai
from pathlib import Path
from datetime import datetime
from typing import List, Literal, Dict, Any, Optional

from dotenv import load_dotenv
load_dotenv()


class DoctorAgent:
    SeverityLevel = Literal["low", "moderate", "high", "emergency"]
    RecommendationType = Literal["self_care", "consult_gp", "urgent_care", "emergency"]

    def __init__(self, log_level: int = logging.INFO):
        self._setup_logging(log_level)
        self._setup_api_clients()
        
        self.logger.info("Doctor Agent initialized successfully with Gemini API")

        self.system_prompt = """
**Your Role:**  
You are an expert doctor assisting medical personnel. You provide helpful medical information and guidance while being extremely careful about medical advice.

**Core Rules:**    
- Recommend consulting a healthcare professional for serious cases
- Advice specific diagnosis based on the context with variable confidence on each if there is any
- Keep your advice very concise and use medical keywords as it will be use only for advice for a expert medical personnel
- You only response based on the provided JSON format

**Specific Task:**
- Assess symptom severity and provide appropriate recommendations
- Offer first aid guidance for emergency situations

**Response Guidelines:**
- Recommend clarifying questions when needed
- Use clear, actionable for guidance
- Include appropriate medical disclaimers
- Use structured assessment approach
- Respond in the user's preferred language when specified
- Do not provide any reasons to your "Diagnosis" confidence

**Output Format in JSON:**
{"Recommendations": [one or two most priority recommendation or note for medical personnel]
"Diagnosis 0-10 with Confidence": {[Disease | Symptom]: [Confidence 0-10]}
"Doctor Plan": [short plan for your future self and medical personnel]}
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
            self.gemini_model_name = "gemini-2.5-flash-lite"
            self.logger.info(f"Gemini API client initialized with model: {self.gemini_model_name}")
        except Exception as e:
            self.logger.error(f"Failed to initialize API clients: {str(e)}")
            raise

    def _format_prompt_gemini(self, message_context: str, medical_context: str = "") -> str:
        now = datetime.now()
        current_context = f"""
**Current Context:**
- Date/Time: {now.strftime("%Y-%m-%d %H:%M:%S")}
- Medical Context: {medical_context}
"""
        
        prompt = f"""{self.system_prompt}
{current_context}
**Patient Query:** {message_context}

Please provide your medical guidance following the guidelines above."""
        
        return prompt

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

    def analyze(self, user_message: str, chat_history: List[Dict], chain_of_thoughts: str = "", summarized_histories: str = "") -> str:
        """
        Main analyze method expected by orchestrator.
        Returns a single commentary string.
        """
        context_parts = []
        if summarized_histories:
            context_parts.append(f"Patient History Summary: {summarized_histories}")
        if chain_of_thoughts:
            context_parts.append(f"Your Previous Recommendations: {chain_of_thoughts}")

        recent_context = []
        for msg in chat_history[-3:]:
            if msg.get("role") == "user":
                recent_context.append(f"Patient: {msg.get('content', '')}")
            elif msg.get("role") == "assistant":
                recent_context.append(f"Medical Personnel: {msg.get('content', '')}")
        if recent_context:
            context_parts.append("Recent Conversation:\n" + "\n".join(recent_context))

        full_context = "\n\n".join(context_parts) if context_parts else ""

        message_context = f"""

Current Patient Message: {user_message}
Available Context:
{full_context if full_context else "No previous context available"}
"""

        prompt = self._format_prompt_gemini(message_context=message_context)
        print(prompt)
        return self._generate_response_gemini(prompt)


if __name__ == "__main__":
    # Minimal reproducible demo for DoctorAgent using existing analyze() method

    # 1) Create the agent
    try:
        doctor = DoctorAgent(log_level=logging.DEBUG)
    except Exception as e:
        print(f"[BOOT ERROR] Unable to start DoctorAgent: {e}")
        raise SystemExit(1)

    # 2) Dummy chat history (what the user and assistant said earlier)
    chat_history = [
        {"role": "user", "content": "Hi, I've been having some stomach issues lately."},
        {"role": "assistant", "content": "I'm sorry to hear about your stomach issues. Can you tell me more about the symptoms?"}
    ]

    # 3) Chain of thoughts from previous analysis
    chain_of_thoughts = "Patient reports digestive issues, need to assess severity and duration."

    # 4) Summarized patient history
    summarized_histories = "Previous sessions: Patient is 25 y/o, works in high-stress environment, irregular eating habits, drinks 3-4 cups of coffee daily."

    # 5) Current user message about medical concern
    user_message = "I've been having sharp stomach pains after eating, and I feel nauseous. It's been going on for about a week now."

    # ===== Test: Medical Analysis =====
    print("\n=== DOCTOR AGENT TEST ===")
    print(f"User Message: {user_message}")
    print(f"Chat History Length: {len(chat_history)}")
    print(f"Context: {summarized_histories}")
    print("\n=== MEDICAL ANALYSIS RESULT ===")
    
    medical_response = doctor.analyze(
        user_message=user_message,
        chat_history=chat_history,
        chain_of_thoughts=chain_of_thoughts,
        summarized_histories=summarized_histories
    )
    
    print(medical_response)
    print("\n=== TEST COMPLETED ===")