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


class DoctorAgent:
    SeverityLevel = Literal["low", "moderate", "high", "emergency"]
    RecommendationType = Literal["self_care", "consult_gp", "urgent_care", "emergency"]

    def __init__(self, log_level: int = logging.INFO):
        self._setup_logging(log_level)
        self._setup_api_clients()
        
        self.logger.info("Doctor Agent initialized successfully")

        self.system_prompt = """
You are a Medical Assistant AI Doctor. You provide helpful medical information and guidance while being extremely careful about medical advice.

**IMPORTANT MEDICAL DISCLAIMERS:**
- You are NOT a replacement for professional medical care
- Always use a calm and reassuring tone
- Always recommend consulting a healthcare professional for serious concerns
- In emergencies, always advise calling emergency services immediately
- Do not provide specific diagnoses - only general information and guidance

**Your capabilities:**
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
1. Advise calling emergency services
2. Provide relevant first aid guidance
3. Emphasize urgency while keeping the user calm

**Output Format:**
Provide structured responses including:
- Symptom assessment (if applicable)
- Recommendations (self-care, consult GP, urgent care, emergency)
- Next steps
- Medical disclaimer

Remember: Your primary goal is to be helpful while ensuring user safety through appropriate medical guidance and referrals.
"""

    def _setup_logging(self, log_level: int) -> None:
        """Setup logging configuration"""
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
                
            self.logger.info("SEA-Lion API client initialized for Doctor Agent")
                
        except Exception as e:
            self.logger.error(f"Failed to initialize API clients: {str(e)}")
            raise

    def _format_messages(self, prompt: str, context: str = "") -> List[Dict[str, str]]:
        """Format messages for API call"""
        now = datetime.now()
        
        context_info = f"""
**Current Context:**
- Date/Time: {now.strftime("%Y-%m-%d %H:%M:%S")}
- Medical Context: {context}
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

    def _generate_response_with_thinking(self, messages: List[Dict[str, str]]) -> Tuple[str, str]:
        """Generate response using SEA-Lion API and extract thinking + commentary"""
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
                "max_tokens": 2000,
                "temperature": 0.3,  # Lower temperature for consistent medical advice
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
                return "Error in medical analysis", "ขออภัยค่ะ ไม่สามารถประมวลผลคำตอบได้ในขณะนี้"
            
            choice = response_data["choices"][0]
            if "message" not in choice or "content" not in choice["message"]:
                self.logger.error(f"Unexpected message structure: {choice}")
                return "Error in medical analysis", "ขออภัยค่ะ ไม่สามารถประมวลผลคำตอบได้ในขณะนี้"
                
            raw_content = choice["message"]["content"]
            
            if raw_content is None or (isinstance(raw_content, str) and raw_content.strip() == ""):
                self.logger.error("SEA-Lion API returned None or empty content")
                return "No medical analysis available", "ขออภัยค่ะ ไม่สามารถสร้างคำตอบได้ในขณะนี้"
            
            # Extract thinking and answer blocks
            thinking_match = re.search(r"```thinking\s*(.*?)\s*```", raw_content, re.DOTALL)
            answer_match = re.search(r"```answer\s*(.*?)\s*```", raw_content, re.DOTALL)
            
            thinking = thinking_match.group(1).strip() if thinking_match else "Medical analysis in progress..."
            commentary = answer_match.group(1).strip() if answer_match else raw_content.strip()
            
            self.logger.info(f"Generated medical response - Thinking: {len(thinking)} chars, Commentary: {len(commentary)} chars")
            return thinking, commentary
            
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Error generating response from SEA-Lion API: {str(e)}")
            return "Connection error during medical analysis", "ขออภัยค่ะ เกิดปัญหาในการเชื่อมต่อ กรุณาลองใหม่อีกครั้งค่ะ"
        except Exception as e:
            self.logger.error(f"Error generating response: {str(e)}")
            return "Error in medical analysis", "ขออภัยค่ะ เกิดข้อผิดพลาดในระบบ"

    def analyze(self, user_message: str, chat_history: List[Dict], chain_of_thoughts: str = "", summarized_histories: str = "") -> Dict[str, str]:
        """
        Main analyze method expected by orchestrator
        
        Args:
            user_message: Current user input
            chat_history: Previous conversation history
            chain_of_thoughts: Past analysis chain of thoughts
            summarized_histories: Summarized conversation histories
            
        Returns:
            Dict with 'thinking' and 'commentary' keys
        """
        # Build comprehensive context for medical analysis
        context_parts = []
        
        if summarized_histories:
            context_parts.append(f"Patient History Summary: {summarized_histories}")
        
        if chain_of_thoughts:
            context_parts.append(f"Previous Medical Considerations: {chain_of_thoughts}")
        
        # Extract recent relevant context from chat history
        recent_context = []
        for msg in chat_history[-3:]:  # Last 3 messages for context
            if msg.get("role") == "user":
                recent_context.append(f"Patient: {msg.get('content', '')}")
            elif msg.get("role") == "assistant":
                recent_context.append(f"Previous Response: {msg.get('content', '')}")
        
        if recent_context:
            context_parts.append("Recent Conversation:\n" + "\n".join(recent_context))
        
        full_context = "\n\n".join(context_parts) if context_parts else ""
        
        # Create comprehensive medical analysis prompt
        prompt = f"""
Based on the current medical query and available context, provide comprehensive medical guidance:

**Current Query:** {user_message}

**Available Context:**
{full_context if full_context else "No previous context available"}

Please provide:

1. **Medical Assessment:**
   - Symptom analysis and potential severity assessment
   - Risk factors and concerning signs identification
   - Differential considerations (without providing diagnoses)

2. **Recommendations:**
   - Immediate care recommendations (self-care, GP consultation, urgent care, emergency)
   - Timeline for seeking professional medical attention
   - Warning signs that would escalate the situation

3. **Patient Education:**
   - Relevant health information for the condition
   - Prevention strategies and self-monitoring guidance
   - When to seek follow-up care

4. **Safety Considerations:**
   - Emergency protocols if applicable
   - First aid guidance when relevant
   - Clear indicators for immediate medical attention

**Response Structure:**

```answer
[Concise, patient-friendly medical guidance with clear recommendations, appropriate disclaimers, and actionable next steps. Keep professional yet empathetic tone.]
```

Always include appropriate medical disclaimers and emphasize professional medical care when needed.
"""

        messages = self._format_messages(prompt, full_context)
        thinking, commentary = self._generate_response_with_thinking(messages)
        
        return {
            "thinking": thinking,
            "commentary": commentary
        }

    # Keep existing methods for backward compatibility
    def diagnose_and_advise(self, symptoms: str, patient_history: Optional[str] = None, language: str = "english") -> str:
        """Legacy method - returns commentary only for backward compatibility"""
        fake_history = []
        if patient_history:
            fake_history.append({"role": "system", "content": f"Patient History: {patient_history}"})
        
        result = self.analyze(symptoms, fake_history, "", patient_history or "")
        return result["commentary"]

    def provide_health_information(self, health_topic: str, language: str = "english") -> str:
        """Legacy method - returns commentary only for backward compatibility"""
        result = self.analyze(f"Please provide information about {health_topic}", [], "", "")
        return result["commentary"]

    def emergency_guidance(self, emergency_situation: str, language: str = "english") -> str:
        """Legacy method - returns commentary only for backward compatibility"""
        result = self.analyze(f"EMERGENCY: {emergency_situation}", [], "", "")
        return result["commentary"]

    def follow_up_consultation(self, previous_symptoms: str, current_status: str, language: str = "english") -> str:
        """Legacy method - returns commentary only for backward compatibility"""
        history = [{"role": "system", "content": f"Previous symptoms: {previous_symptoms}"}]
        result = self.analyze(f"Follow-up consultation - Current status: {current_status}", history, "", "")
        return result["commentary"]

    def assess_symptom_severity(self, symptoms: str, additional_context: str = "") -> Dict[str, Any]:
        """Assess symptom severity and provide structured recommendation"""
        result = self.analyze(symptoms, [], "", additional_context)
        
        # Parse response to extract structured information
        commentary = result["commentary"]
        severity_level = "moderate"  # default
        recommendation = "consult_gp"  # default
        
        # Simple parsing logic to extract severity and recommendation
        commentary_lower = commentary.lower()
        if "emergency" in commentary_lower:
            severity_level = "emergency"
            recommendation = "emergency"
        elif "high" in commentary_lower or "urgent" in commentary_lower:
            severity_level = "high"
            recommendation = "urgent_care"
        elif "low" in commentary_lower or "mild" in commentary_lower:
            severity_level = "low"
            recommendation = "self_care"
        
        return {
            "severity_level": severity_level,
            "recommendation": recommendation,
            "full_response": commentary,
            "thinking": result["thinking"],
            "timestamp": datetime.now().isoformat()
        }

    def provide_first_aid_guidance(self, situation: str, language: str = "english") -> str:
        """Legacy method - returns commentary only for backward compatibility"""
        result = self.analyze(f"First aid guidance needed for: {situation}", [], "", "")
        return result["commentary"]


if __name__ == "__main__":
    # Test the modified Doctor Agent
    try:
        doctor = DoctorAgent(log_level=logging.DEBUG)
        
        # Test the new analyze method
        print("=== TEST: ANALYZE METHOD ===")
        result = doctor.analyze(
            user_message="I have severe headache, nausea, and sensitivity to light for the past 2 hours",
            chat_history=[
                {"role": "user", "content": "I've been having headaches lately"},
                {"role": "assistant", "content": "I understand you've been experiencing headaches. Can you tell me more about them?"}
            ],
            chain_of_thoughts="Previous consideration of tension headaches, but current presentation more concerning",
            summarized_histories="Patient has history of occasional mild headaches, usually stress-related"
        )
        
        print("THINKING:")
        print(result["thinking"])
        print("\nCOMMENTARY:")
        print(result["commentary"])
        
        # Test legacy method still works
        print("\n=== TEST: LEGACY METHOD ===")
        legacy_result = doctor.diagnose_and_advise(
            symptoms="Fever and cough",
            patient_history="Healthy adult, no chronic conditions"
        )
        print(legacy_result)
        
    except Exception as e:
        print(f"Error testing Doctor Agent: {e}")