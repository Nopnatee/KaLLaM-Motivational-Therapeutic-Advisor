import os
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Literal, Dict, Any, Optional

from dotenv import load_dotenv
from strands import Agent, tool
from strands.models.openai import OpenAIModel

load_dotenv()


class DoctorAgent:
    SeverityLevel = Literal["low", "moderate", "high", "emergency"]
    RecommendationType = Literal["self_care", "consult_gp", "urgent_care", "emergency"]

    def __init__(self, log_level: int = logging.INFO):
        self._setup_logging(log_level)
        self._setup_agent()
        
        self.logger.info("Doctor Agent initialized successfully")

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

    def _setup_agent(self) -> None:
        """Setup Strands Agent with GPT-4o"""
        try:
            # Verify OpenAI API key is available
            openai_api_key = os.getenv("OPENAI_API_KEY")
            if not openai_api_key:
                raise ValueError("OPENAI_API_KEY not found in environment variables")
            
            system_prompt = """
**Your Role:**  
You are a Medical Assistant AI Doctor. You provide helpful medical information and guidance while being extremely careful about medical advice.

**Core Rules:**  
- You are NOT a replacement for professional medical care
- Always use a calm and reassuring tone
- Maintain warmth, empathy, and professional boundaries at all times.  
- Always recommend consulting a healthcare professional for serious concerns
- In emergencies, always advise calling emergency services immediately
- Do not provide specific diagnoses - only general information and guidance

**Specific Task:**
- Assess symptom severity and provide appropriate recommendations
- Provide general health information and wellness advice
- Offer first aid guidance for emergency situations
- Recognize when immediate medical attention is needed
- Support users with health concerns while emphasizing professional care

**Response Guidelines:**
- Be empathetic and supportive
- Ask clarifying questions when needed
- Provide clear, actionable guidance
- Always include appropriate medical disclaimers
- Use structured assessment approach
- Respond in the user's preferred language when specified

**Emergency Protocol:**
If you detect emergency symptoms, immediately:
1. Advise calling emergency services complete with the local emergency number
2. Provide relevant first aid guidance
3. Emphasize urgency while keeping the user calm

**Output Format:**
Provide structured responses including:
- Symptom assessment (if applicable)
- Recommendations (self-care, consult GP, urgent care, emergency)
- Next steps
- Medical disclaimer
"""
            
            # Create Strands Agent with GPT-4o
            self.agent = Agent(
                name="DoctorAgent",
                instructions=system_prompt,
                model="gpt-4o",
                temperature=0.3,
                max_tokens=2000
            )
            
            self.logger.info("Strands Agent initialized with GPT-4o")
                
        except Exception as e:
            self.logger.error(f"Failed to initialize Strands Agent: {str(e)}")
            raise

    def analyze(self, user_message: str, chat_history: List[Dict], chain_of_thoughts: str = "", summarized_histories: str = "") -> str:
        """
        Main analyze method expected by orchestrator.
        Returns a single commentary string.
        """
        try:
            # Build context
            context_parts = []
            now = datetime.now()
            context_parts.append(f"Current Date/Time: {now.strftime('%Y-%m-%d %H:%M:%S')}")
            
            if summarized_histories:
                context_parts.append(f"Patient History Summary: {summarized_histories}")
            if chain_of_thoughts:
                context_parts.append(f"Previous Medical Considerations: {chain_of_thoughts}")

            # Add recent conversation context
            recent_context = []
            for msg in chat_history[-3:]:
                if msg.get("role") == "user":
                    recent_context.append(f"Patient: {msg.get('content', '')}")
                elif msg.get("role") == "assistant":
                    recent_context.append(f"Previous Response: {msg.get('content', '')}")
            if recent_context:
                context_parts.append("Recent Conversation:\n" + "\n".join(recent_context))

            full_context = "\n\n".join(context_parts) if context_parts else ""

            # Create comprehensive prompt
            prompt = f"""
Based on the current medical query and available context, provide comprehensive medical guidance:

**Current Query:** {user_message}

**Available Context:**
{full_context if full_context else "No previous context available"}

Please provide:

1. **Medical Assessment**
2. **Recommendations**
3. **Patient Education**
4. **Safety Considerations**

Please provide concise, patient-friendly medical guidance with clear recommendations, appropriate disclaimers, and actionable next steps. Keep professional yet empathetic tone.
"""

            # Use Strands Agent to generate response
            response = self.agent.run(prompt)
            
            # Extract the text content from the response
            if hasattr(response, 'messages') and response.messages:
                # Get the last message content
                last_message = response.messages[-1]
                if hasattr(last_message, 'text'):
                    medical_response = last_message.text
                elif hasattr(last_message, 'content'):
                    medical_response = last_message.content
                else:
                    medical_response = str(last_message)
            else:
                medical_response = str(response)
            
            self.logger.info(f"Generated medical response - Length: {len(medical_response)} chars")
            return medical_response

        except Exception as e:
            self.logger.error(f"Error in analyze method: {str(e)}")
            return "ขออภัยค่ะ เกิดข้อผิดพลาดในระบบ กรุณาลองใหม่อีกครั้งค่ะ"


if __name__ == "__main__":
    # Minimal reproducible demo for DoctorAgent using existing analyze() method
    # Requires OPENAI_API_KEY in your environment, otherwise the class will raise.

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