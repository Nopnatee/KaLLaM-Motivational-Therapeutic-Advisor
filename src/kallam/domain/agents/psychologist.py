import os
import json
import logging
import requests
import re
from pathlib import Path
from datetime import datetime
from typing import List, Literal, Dict, Any, Optional, Tuple

from dotenv import load_dotenv
load_dotenv()


class PsychologistAgent:
    TherapyApproach = Literal["cbt", "dbt", "act", "motivational", "solution_focused", "mindfulness"]
    CrisisLevel = Literal["none", "mild", "moderate", "severe", "emergency"]

    def __init__(self, log_level: int = logging.INFO):
        self._setup_logging(log_level)
        self._setup_api_clients()
        
        self.logger.info("Psychologist Agent initialized successfully")

        self.system_prompt = """
**Your Role:**  
You are a Professional Psychological Counselor AI specializing in mental health support and therapeutic guidance.  
Your goal is to provide evidence-based psychological interventions while maintaining professional boundaries.

**Core Rules:**  
- You are NOT a replacement for professional mental health care.  
- Always use a calm and reassuring tone
- Maintain warmth, empathy, and professional boundaries at all times.  
- Never provide diagnoses; only supportive guidance and coping strategies.  
- Always recommend consulting a licensed professional for serious concerns.  
- In emergencies (suicidal thoughts, self-harm, psychosis), advise immediate professional intervention.  

**Therapeutic Approaches:**  
1. **Active Listening:** Reflect, paraphrase, validate, and show empathy.  
2. **CBT (Cognitive Behavioral Therapy):** Identify distortions, reframe thoughts, assign behavioral tasks.  
3. **Motivational Interviewing:** Explore ambivalence, elicit change talk, support autonomy.  
4. **Solution-Focused Therapy:** Highlight strengths, scaling questions, set small goals, miracle question.  
5. **Mindfulness & Stress Management:** Grounding, breathing, meditation, relaxation, stress inoculation.  
6. **Crisis Intervention:** Risk assessment, de-escalation, safety planning, connect to resources.  

**Response Guidelines:**  
- Use open-ended questions, empathetic tone, and validation.  
- Provide psychoeducation and coping strategies.  
- Include evidence-based interventions tailored to the client.  
- Respond in clientâ€™s preferred language when specified.  
- Always include crisis safety steps when risk is detected.  

**Crisis Assessment Protocol:**  
- If signs of suicidal ideation, self-harm, psychosis, severe dissociation, substance emergencies, or abuse appear:  
â†' Immediately recommend emergency professional intervention **and** provide supportive guidance.  

**Output Format:**  
Always structure responses to include:  
- Emotional validation and reflection  
- Psychological assessment (thoughts, emotions, behaviors, risks)  
- Intervention strategies (evidence-based techniques)  
- Professional referral or safety planning if needed  

**Specific Task:**  
Provide supportive, structured, evidence-based therapeutic guidance while ensuring client safety.  
Your primary purpose is to help clients develop coping skills and encourage appropriate professional care.
"""

    def _setup_logging(self, log_level: int) -> None:
        """Setup logging configuration"""
        # Create logs directory if it doesn't exist
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        
        # Setup logger
        self.logger = logging.getLogger(f"{__name__}.PsychologistAgent")
        self.logger.setLevel(log_level)
        
        # Remove existing handlers to avoid duplicates
        if self.logger.handlers:
            self.logger.handlers.clear()
        
        # File handler for detailed logs
        file_handler = logging.FileHandler(
            log_dir / f"psychologist_{datetime.now().strftime('%Y%m%d')}.log",
            encoding='utf-8'
        )
        file_handler.setLevel(logging.DEBUG)
        
        # Console handler for immediate feedback
        console_handler = logging.StreamHandler()
        console_handler.setLevel(log_level)
        
        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        # Add handlers
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)

    def _setup_api_clients(self) -> None:

        try:
            self.sea_lion_api_key = os.getenv("SEA_LION_API_KEY")
            self.sea_lion_base_url = os.getenv("SEA_LION_BASE_URL", "https://api.sea-lion.ai/v1")
            
            if not self.sea_lion_api_key:
                raise ValueError("SEA_LION_API_KEY not provided and not found in environment variables")
            
            self.logger.info("SEA-Lion API client initialized for Psychologist Agent")
        
        except Exception as e:
            self.logger.error(f"Failed to initialize API clients: {str(e)}")
            raise

    def _format_messages(self, prompt: str, context: str = "") -> List[Dict[str, str]]:
        now = datetime.now()
        context_info = f"""
**Current Context:**
- Date/Time: {now.strftime("%Y-%m-%d %H:%M:%S")}
- Therapeutic Context: {context}
"""
        system_message = {"role": "system", "content": f"{self.system_prompt}\n\n{context_info}"}
        user_message = {"role": "user", "content": prompt}
        return [system_message, user_message]

    def _generate_response_with_thinking(self, messages: List[Dict[str, str]]) -> str:
        try:
            self.logger.debug(f"Sending {len(messages)} messages to SEA-Lion API")
            headers = {"Authorization": f"Bearer {self.sea_lion_api_key}", "Content-Type": "application/json"}
            payload = {
                "model": "aisingapore/Llama-SEA-LION-v3.5-8B-R",
                "messages": messages,
                "chat_template_kwargs": {"thinking_mode": "on"},
                "max_tokens": 2500,
                "temperature": 0.4,
                "top_p": 0.8,
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
                return "ขออภัยค่ะ ไม่สามารถประมวลผลคำตอบได้ในขณะนี้"
            
            choice = response_data["choices"][0]
            if "message" not in choice or "content" not in choice["message"]:
                self.logger.error(f"Unexpected message structure: {choice}")
                return "ขออภัยค่ะ ไม่สามารถประมวลผลคำตอบได้ในขณะนี้"
        
            raw_content = choice["message"]["content"]
            if raw_content is None or (isinstance(raw_content, str) and raw_content.strip() == ""):
                self.logger.error("SEA-Lion API returned None or empty content")
                return "ขออภัยค่ะ ไม่สามารถสร้างคำตอบได้ในขณะนี้"
            
            answer_match = re.search(r"```answer\s*(.*?)\s*```", raw_content, re.DOTALL)
            commentary = answer_match.group(1).strip() if answer_match else raw_content.strip()
            
            self.logger.info(f"Generated therapeutic response - Commentary: {len(commentary)} chars")
            return commentary
        
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Error generating response from SEA-Lion API: {str(e)}")
            return "ขออภัยค่ะ เกิดปัญหาในการเชื่อมต่อ กรุณาลองใหม่อีกครั้งค่ะ"
        
        except Exception as e:
            self.logger.error(f"Error generating response: {str(e)}")
            return "ขออภัยค่ะ เกิดข้อผิดพลาดในระบบ"

    def provide_therapeutic_guidance(self, user_message: str, therapeutic_context: str = "") -> str:
        """
        Main method to provide psychological guidance and support
        
        Args:
            user_message: The client's message or concern
            therapeutic_context: Additional context about the client's situation
            
        Returns:
            Therapeutic response with guidance and support
        """
        self.logger.info("Processing therapeutic guidance request")
        self.logger.debug(f"User message: {user_message}")
        self.logger.debug(f"Therapeutic context: {therapeutic_context}")
        
        try:
            messages = self._format_messages(user_message, therapeutic_context)
            response = self._generate_response_with_thinking(messages)
            
            if response is None:
                raise Exception("SEA-Lion API returned None response")
                
            return response
            
        except Exception as e:
            self.logger.error(f"Error in provide_therapeutic_guidance: {str(e)}")
            return "ขออภัยค่ะ ระบบมีปัญหาชั่วคราว กรุณาลองใหม่อีกครั้งค่ะ หากมีความคิดทำร้ายตัวเอง กรุณาติดต่อแพทย์หรือสายด่วนช่วยเหลือทันที"


if __name__ == "__main__":
    # Minimal reproducible demo for PsychologistAgent
    # Requires SEA_LION_API_KEY in your environment, otherwise the class will raise.

    # 1) Create the agent
    try:
        psychologist = PsychologistAgent(log_level=logging.DEBUG)
    except Exception as e:
        print(f"[BOOT ERROR] Unable to start PsychologistAgent: {e}")
        raise SystemExit(1)

    # 2) Test scenarios for psychological support
    test_cases = [
        {
            "name": "Exam Anxiety",
            "user_message": "I have a headache and feel anxious about my exams. I can't sleep and keep worrying about failing.",
            "therapeutic_context": "User: 21 y/o student, midterm week, low sleep (4-5h), high caffeine, history of anxiety during academic stress."
        },
        {
            "name": "General Stress Management", 
            "user_message": "I've been feeling overwhelmed lately with work and personal life. Everything feels like too much.",
            "therapeutic_context": "User: Working professional, recent job change, managing family responsibilities, seeking coping strategies."
        },
        {
            "name": "Relationship Issues",
            "user_message": "My relationship with my partner has been really difficult. We keep arguing and I don't know how to fix it.",
            "therapeutic_context": "User: In committed relationship, communication difficulties, seeking relationship guidance and conflict resolution strategies."
        }
    ]

    # Run tests
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n{'='*50}")
        print(f"TEST {i}: {test_case['name']}")
        print(f"{'='*50}")
        
        print(f"\n User Message: {test_case['user_message']}")
        print(f" Context: {test_case['therapeutic_context']}")
        
        print(f"\n PSYCHOLOGIST RESPONSE:")
        print("-" * 40)
        
        response = psychologist.provide_therapeutic_guidance(
            user_message=test_case['user_message'],
            therapeutic_context=test_case['therapeutic_context']
        )
        
        print(response)
        print("\n" + "="*50)

    print(f"\n All tests completed successfully!")
    print(" The PsychologistAgent is ready for integration with the supervisor system.")