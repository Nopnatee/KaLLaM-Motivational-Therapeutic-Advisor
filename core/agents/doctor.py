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

MODEL_ID = "Llama-SEA-LION-v3-8B-IT"
REGION   = "ap-southeast-2"

GUARDRAIL_ID      = None
GUARDRAIL_VERSION = None

# --------------------------
# Doctor Agent
# -is static method good here?
# --------------------------

class DoctorAgent:
    SeverityLevel = Literal["low", "moderate", "high", "emergency"]
    RecommendationType = Literal["self_care", "consult_gp", "urgent_care", "emergency"]


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
            temperature=0.3,  # Lower temperature for more consistent medical advice
            top_p=0.8,
            stop_sequences=["</END>"],

            guardrail_id=guardrail_id,
            guardrail_version=guardrail_version,
            guardrail_trace="enabled",
            guardrail_stream_processing_mode="sync",
            guardrail_redact_input=True,
            guardrail_redacted_input_message="[User input redacted due to medical privacy policy]",
            guardrail_redact_output=False,

            cache_prompt="default",
            cache_tools="default",
            boto_client_config=boto_cfg,
        )

        system_prompt = """\
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
- Use tools available to assess severity and provide guidance
- Respond in the user's preferred language when specified

**Emergency Protocol:**
If you detect emergency symptoms, immediately:
1. Advise calling emergency services
2. Provide relevant first aid guidance using available tools
3. Emphasize urgency while keeping the user calm

**Output Format:**
Provide structured responses including:
- Symptom assessment (if applicable)
- Recommendations (self-care, consult GP, urgent care, emergency)
- Next steps
- Medical disclaimer

Remember: Your primary goal is to be helpful while ensuring user safety through appropriate medical guidance and referrals.
"""

        self.agent = Agent(
            model=bedrock_model,
            system_prompt=system_prompt,
            tools=[
                self.update_model_id, 
                self.update_temperature,
                self.assess_symptom_severity,
                self.provide_first_aid_guidance
            ],
            callback_handler=None,
        )

    # --------------------------
    # Public methods
    # --------------------------
    def diagnose_and_advise(self, symptoms: str, patient_history: Optional[str] = None, language: str = "english") -> str:
        """Main method to analyze symptoms and provide medical advice"""
        
        context = f"Patient symptoms: {symptoms}"
        if patient_history:
            context += f"\nPatient history: {patient_history}"
        
        prompt = f"""
        Please analyze the following medical query and provide appropriate guidance:
        
        {context}
        
        Please:
        1. Use the assess_symptom_severity tool to evaluate the symptoms
        2. Provide appropriate medical guidance based on the assessment
        3. Include relevant first aid information if needed (use provide_first_aid_guidance tool)
        4. Respond in {language} language
        5. Always include appropriate medical disclaimers
        
        Provide a comprehensive but clear response that helps the user understand their situation and next steps.
        """
        
        response = self.agent(prompt)
        return response

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
        """
        
        response = self.agent(prompt)
        return response

    def emergency_guidance(self, emergency_situation: str, language: str = "english") -> str:
        """Provide emergency medical guidance"""
        
        prompt = f"""
        EMERGENCY SITUATION: {emergency_situation}
        
        Please:
        1. Immediately assess if this requires emergency services
        2. Use the provide_first_aid_guidance tool for relevant first aid steps
        3. Provide clear, step-by-step emergency guidance
        4. Respond in {language} language
        5. Keep instructions simple and actionable
        
        This is urgent - provide immediate, life-saving guidance while emphasizing professional emergency care.
        """
        
        response = self.agent(prompt)
        return response

    def follow_up_consultation(self, previous_symptoms: str, current_status: str, language: str = "english") -> str:
        """Handle follow-up consultations for ongoing health concerns"""
        
        prompt = f"""
        FOLLOW-UP:
        
        Previous symptoms: {previous_symptoms}
        Current status: {current_status}
        
        Please:
        1. Assess the progression or improvement of symptoms
        2. Determine if the current approach is appropriate
        3. Advise on next steps (continue current care, escalate, etc.)
        4. Respond in {language} language
        
        Provide guidance on whether the situation is improving, stable, or requires different intervention.
        """
        
        response = self.agent(prompt)
        return response