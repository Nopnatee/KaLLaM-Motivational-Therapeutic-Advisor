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
                return "ขออภัยค่ะ ไม่สามารถประมวลผลคำตอบได้ในขณะนี้"
            
            choice = response_data["choices"][0]
            if "message" not in choice or "content" not in choice["message"]:
                self.logger.error(f"Unexpected message structure: {choice}")
                return "ขออภัยค่ะ ไม่สามารถประมวลผลคำตอบได้ในขณะนี้"
                
            raw_content = choice["message"]["content"]
            
            if raw_content is None or (isinstance(raw_content, str) and raw_content.strip() == ""):
                self.logger.error("SEA-Lion API returned None or empty content")
                return "ขออภัยค่ะ ไม่สามารถสร้างคำตอบได้ในขณะนี้"
            
            # Extract thinking and answer blocks
            thinking_match = re.search(r"```thinking\s*(.*?)\s*```", raw_content, re.DOTALL)
            answer_match = re.search(r"```answer\s*(.*?)\s*```", raw_content, re.DOTALL)
            
            reasoning = thinking_match.group(1).strip() if thinking_match else None
            final_answer = answer_match.group(1).strip() if answer_match else raw_content.strip()
            
            if reasoning:
                self.logger.debug(f"Doctor reasoning:\n{reasoning}")
            
            self.logger.info(f"Generated medical response (length: {len(final_answer)} chars)")
            return final_answer
            
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Error generating response from SEA-Lion API: {str(e)}")
            return "ขออภัยค่ะ เกิดปัญหาในการเชื่อมต่อ กรุณาลองใหม่อีกครั้งค่ะ"
        except Exception as e:
            self.logger.error(f"Error generating response: {str(e)}")
            return "ขออภัยค่ะ เกิดข้อผิดพลาดในระบบ"

    # Public methods
    def diagnose_and_advise(self, symptoms: str, patient_history: Optional[str] = None, language: str = "english") -> str:
        """Main method to analyze symptoms and provide medical advice"""
        
        context = f"Patient symptoms: {symptoms}"
        if patient_history:
            context += f"\nPatient history: {patient_history}"
        
        prompt = f"""
Please analyze the following medical query and provide appropriate guidance:

{context}

Please provide:
1. Symptom severity assessment (low/moderate/high/emergency)
2. Appropriate medical guidance based on the assessment
3. Relevant first aid information if needed
4. Response in {language} language
5. Always include appropriate medical disclaimers

Provide a comprehensive but clear response that helps the user understand their situation and next steps.

Assessment Framework:
- **Emergency**: Life-threatening symptoms requiring immediate medical attention
- **High**: Serious symptoms needing urgent medical care within 24 hours
- **Moderate**: Concerning symptoms requiring medical consultation within few days
- **Low**: Minor symptoms manageable with self-care and monitoring

Include specific recommendations for:
- Immediate actions to take
- When to seek professional medical care
- Self-care measures if appropriate
- Warning signs that would escalate the situation
"""
        
        messages = self._format_messages(prompt, context)
        return self._generate_response(messages)

    def provide_health_information(self, health_topic: str, language: str = "english") -> str:
        """Provide general health information on a specific topic"""
        
        prompt = f"""
Please provide general health information about: {health_topic}

Include:
- Overview of the topic
- Common symptoms or signs (if applicable)
- Prevention strategies
- When to seek medical care
- General management tips

Respond in {language} language and include appropriate medical disclaimers.
Keep the information accurate, helpful, and accessible to general audiences.

Structure your response with:
1. **Overview**: Brief explanation of the condition/topic
2. **Key Information**: Important facts people should know
3. **Prevention**: How to prevent or reduce risk
4. **Management**: General approaches to managing the condition
5. **When to Seek Help**: Clear guidelines on when professional care is needed
6. **Medical Disclaimer**: Appropriate disclaimers about professional care
"""
        
        context = f"Health information request about: {health_topic}"
        messages = self._format_messages(prompt, context)
        return self._generate_response(messages)

    def emergency_guidance(self, emergency_situation: str, language: str = "english") -> str:
        """Provide emergency medical guidance"""
        
        prompt = f"""
EMERGENCY SITUATION: {emergency_situation}

Please provide immediate emergency medical guidance:

1. **Immediate Assessment**: Determine if this requires emergency services (call 911/emergency number)
2. **First Aid Steps**: Provide clear, step-by-step emergency guidance
3. **Safety Priorities**: Ensure immediate safety of the person
4. **Professional Care**: Emphasize the need for immediate professional emergency care
5. **Stay Calm**: Provide reassurance while maintaining urgency

Respond in {language} language with:
- Clear, simple instructions that can be followed under stress
- Prioritized action steps (most important first)
- Specific guidance on when to call emergency services
- How to monitor the person's condition
- What information to provide to emergency responders

This is urgent - provide immediate, potentially life-saving guidance while emphasizing professional emergency care.

**CRITICAL**: Always prioritize calling emergency services for serious medical emergencies.
"""
        
        context = f"Emergency situation: {emergency_situation}"
        messages = self._format_messages(prompt, context)
        return self._generate_response(messages)

    def follow_up_consultation(self, previous_symptoms: str, current_status: str, language: str = "english") -> str:
        """Handle follow-up consultations for ongoing health concerns"""
        
        prompt = f"""
FOLLOW-UP CONSULTATION:

Previous symptoms: {previous_symptoms}
Current status: {current_status}

Please provide follow-up medical guidance:

1. **Progress Assessment**: Evaluate improvement, stability, or worsening of symptoms
2. **Current Status Analysis**: Assess the current situation compared to baseline
3. **Treatment Effectiveness**: Determine if current approach is working
4. **Next Steps**: Advise on whether to continue, modify, or escalate care
5. **Monitoring**: Provide guidance on ongoing symptom monitoring

Respond in {language} language with:
- Clear assessment of symptom progression
- Recommendations for continuing or changing current approach
- Specific criteria for when to seek additional medical attention
- Timeline for expected improvement or when to reassess
- Warning signs that would require immediate medical attention

Focus on:
- **Improvement**: What indicates the situation is getting better
- **Stability**: What suggests the condition is stable and manageable
- **Deterioration**: What signs indicate the need for escalated care
- **Timeline**: Reasonable expectations for recovery or improvement
"""
        
        context = f"Follow-up: Previous - {previous_symptoms}, Current - {current_status}"
        messages = self._format_messages(prompt, context)
        return self._generate_response(messages)

    def assess_symptom_severity(self, symptoms: str, additional_context: str = "") -> Dict[str, Any]:
        """Assess symptom severity and provide structured recommendation"""
        
        prompt = f"""
Perform a structured symptom severity assessment:

Symptoms: {symptoms}
Additional context: {additional_context}

Please provide a structured assessment in the following format:

**SEVERITY ASSESSMENT:**
Severity Level: [emergency/high/moderate/low]

**REASONING:**
[Explain why this severity level was chosen]

**RECOMMENDATIONS:**
Primary Action: [emergency/urgent_care/consult_gp/self_care]
Timeline: [immediate/within 24h/within few days/monitor]

**SPECIFIC GUIDANCE:**
- Immediate steps to take
- Warning signs to watch for
- When to escalate care

**MEDICAL DISCLAIMER:**
[Appropriate medical disclaimer]

Base your assessment on:
- **Emergency**: Life-threatening, requires immediate medical attention
- **High**: Serious symptoms, needs urgent care within 24 hours
- **Moderate**: Concerning symptoms, should see doctor within few days
- **Low**: Minor symptoms, can manage with self-care and monitoring
"""
        
        context = f"Symptom assessment: {symptoms}"
        if additional_context:
            context += f" | Additional context: {additional_context}"
        
        messages = self._format_messages(prompt, context)
        response = self._generate_response(messages)
        
        # Parse response to extract structured information
        severity_level = "moderate"  # default
        recommendation = "consult_gp"  # default
        
        # Simple parsing logic to extract severity and recommendation
        response_lower = response.lower()
        if "emergency" in response_lower:
            severity_level = "emergency"
            recommendation = "emergency"
        elif "high" in response_lower:
            severity_level = "high"
            recommendation = "urgent_care"
        elif "low" in response_lower:
            severity_level = "low"
            recommendation = "self_care"
        
        return {
            "severity_level": severity_level,
            "recommendation": recommendation,
            "full_response": response,
            "timestamp": datetime.now().isoformat()
        }

    def provide_first_aid_guidance(self, situation: str, language: str = "english") -> str:
        """Provide specific first aid guidance"""
        
        prompt = f"""
Provide first aid guidance for: {situation}

Please provide clear, step-by-step first aid instructions:

**FIRST AID PROTOCOL:**

1. **Safety First**: Ensure scene safety and universal precautions
2. **Initial Assessment**: Quick assessment of the person's condition
3. **Primary Care Steps**: Main first aid interventions
4. **Monitoring**: What to watch for during care
5. **Professional Help**: When and how to get professional medical assistance

Respond in {language} language with:
- Clear, numbered steps that are easy to follow
- Safety considerations and precautions
- What supplies or materials might be needed
- Warning signs that indicate the situation is worsening
- When to call for emergency medical services

**Important**: 
- Always prioritize calling emergency services for serious injuries
- Provide reassurance and comfort to the injured person
- Continue monitoring until professional help arrives
- Do not attempt procedures beyond basic first aid training

Structure as:
**IMMEDIATE ACTIONS:**
**STEP-BY-STEP CARE:**
**MONITORING:**
**WHEN TO CALL FOR HELP:**
**SAFETY PRECAUTIONS:**
"""
        
        context = f"First aid situation: {situation}"
        messages = self._format_messages(prompt, context)
        return self._generate_response(messages)


if __name__ == "__main__":
    # Test the Doctor Agent
    try:
        doctor = DoctorAgent(log_level=logging.DEBUG)
        
        # Test diagnosis
        print("=== TEST: SYMPTOM DIAGNOSIS ===")
        diagnosis = doctor.diagnose_and_advise(
            symptoms="Severe headache, nausea, and sensitivity to light for the past 2 hours",
            patient_history="History of migraines, but this feels different and more severe",
            language="english"
        )
        print(diagnosis)
        
        # Test health information
        print("\n=== TEST: HEALTH INFORMATION ===")
        info = doctor.provide_health_information("hypertension", language="english")
        print(info)
        
    except Exception as e:
        print(f"Error testing Doctor Agent: {e}")