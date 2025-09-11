import os
import logging
import re
from pathlib import Path
from datetime import datetime
from typing import List, Literal, Dict, Any, Optional

from dotenv import load_dotenv
import sagemaker
from sagemaker.huggingface import HuggingFaceModel

load_dotenv()

# --------------------------
# AWS & Hugging Face Setup
# --------------------------
AWS_ROLE_ARN = os.getenv("AWS_ROLE_ARN")  
AWS_REGION = os.getenv("AWS_DEFAULT_REGION", "us-east-1")
HF_MODEL_ID = "google/medgemma-27b-it"
HF_TASK = "text-generation"

# --------------------------
# SageMaker Model Wrapper
# --------------------------
class SageMakerModelWrapper:
    def __init__(self, predictor):
        self.predictor = predictor
        self.max_tokens = 1024
        self.temperature = 0.3
        
    def generate(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """
        Generate response using SageMaker endpoint
        messages: List of message dictionaries with 'role' and 'content' keys
        """
        # Convert messages to a single prompt string for Gemma
        prompt = self._messages_to_gemma_prompt(messages)
        
        # Get parameters
        max_tokens = kwargs.get('max_tokens', self.max_tokens)
        temperature = kwargs.get('temperature', self.temperature)
        
        try:
            response = self.predictor.predict({
                "inputs": prompt,
                "parameters": {
                    "max_new_tokens": max_tokens,
                    "temperature": temperature,
                    "do_sample": True,
                    "top_p": 0.8,
                    "top_k": 50,
                    "repetition_penalty": 1.1,
                    "return_full_text": False  # Only return generated text
                }
            })
            
            # Handle different response formats
            if isinstance(response, list) and len(response) > 0:
                if "generated_text" in response[0]:
                    return response[0]["generated_text"]
                elif isinstance(response[0], str):
                    return response[0]
            elif isinstance(response, dict) and "generated_text" in response:
                return response["generated_text"]
            elif isinstance(response, str):
                return response
            
            return str(response)
            
        except Exception as e:
            return f"Error generating medical response: {str(e)}"
    
    def _messages_to_gemma_prompt(self, messages: List[Dict[str, str]]) -> str:
        """Convert messages format to Gemma chat format"""
        prompt_parts = []
        
        for message in messages:
            role = message.get('role', '')
            content = message.get('content', '')
            
            if role == 'system':
                prompt_parts.append(f"<start_of_turn>user\nSystem: {content}<end_of_turn>")
            elif role == 'user':
                prompt_parts.append(f"<start_of_turn>user\n{content}<end_of_turn>")
            elif role == 'assistant':
                prompt_parts.append(f"<start_of_turn>model\n{content}<end_of_turn>")
        
        prompt_parts.append("<start_of_turn>model\n")
        return "".join(prompt_parts)

class DoctorAgent:
    SeverityLevel = Literal["low", "moderate", "high", "emergency"]
    RecommendationType = Literal["self_care", "consult_gp", "urgent_care", "emergency"]

    def __init__(self, aws_role_arn: str = AWS_ROLE_ARN, region: str = AWS_REGION, log_level: int = logging.INFO):
        self._setup_logging(log_level)
        
        if not aws_role_arn:
            raise ValueError("AWS_ROLE_ARN is required. Please set it in your environment variables.")
        
        # Verify AWS credentials before proceeding
        try:
            import boto3
            sts_client = boto3.client('sts', region_name=region)
            identity = sts_client.get_caller_identity()
            self.logger.info(f"AWS Identity verified: {identity.get('Arn', 'Unknown')}")
        except Exception as e:
            raise ValueError(f"AWS credentials error: {e}. Please check your AWS configuration.")
        
        # Create SageMaker session
        sess = sagemaker.Session()

        # Deploy Hugging Face model - using larger instance for 27B model
        hf_model = HuggingFaceModel(
            model_data=None,
            role=aws_role_arn,
            transformers_version="4.37.0",
            pytorch_version="2.1.0",
            py_version="py310",
            env={
                "HF_MODEL_ID": HF_MODEL_ID,
                "HF_TASK": HF_TASK
            },
            sagemaker_session=sess
        )

        self.logger.info("Deploying MedGemma 27B model to SageMaker endpoint...")
        try:
            # Using larger instance for 27B parameter model
            # ml.p4d.xlarge has A100 40GB - might need ml.p4d.2xlarge for full performance
            predictor = hf_model.deploy(
                initial_instance_count=1,
                instance_type="ml.p4d.xlarge",  # A100 40GB instance
                # Alternative larger instances:
                # instance_type="ml.p4d.2xlarge",  # A100 80GB instance
                # instance_type="ml.g5.12xlarge",  # A10G 96GB total
            )
            self.logger.info("MedGemma 27B model deployed successfully!")
        except Exception as e:
            self.logger.error(f"Failed to deploy model: {e}")
            # Fallback to smaller instance if needed
            self.logger.info("Attempting deployment with smaller instance...")
            predictor = hf_model.deploy(
                initial_instance_count=1,
                instance_type="ml.g5.4xlarge"  # Fallback option
            )

        # Create the model wrapper
        self.model = SageMakerModelWrapper(predictor)
        
        self.logger.info("Doctor Agent initialized successfully with SageMaker")

        self.system_prompt = """
**Your Role:**  
You are a Medical Assistant AI Doctor using advanced medical knowledge. You provide helpful medical information and guidance while being extremely careful about medical advice.

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

    def _format_messages(self, prompt: str, context: str = "") -> List[Dict[str, str]]:
        now = datetime.now()
        context_info = f"""
**Current Context:**
- Date/Time: {now.strftime("%Y-%m-%d %H:%M:%S")}
- Medical Context: {context}
"""
        system_message = {"role": "system", "content": f"{self.system_prompt}\n\n{context_info}"}
        user_message = {"role": "user", "content": prompt}
        return [system_message, user_message]

    def _generate_response_with_thinking(self, messages: List[Dict[str, str]]) -> str:
        try:
            self.logger.debug(f"Sending {len(messages)} messages to SageMaker endpoint")
            
            # Generate response using SageMaker model
            raw_content = self.model.generate(
                messages=messages,
                max_tokens=2000,
                temperature=0.3
            )
            
            if not raw_content or raw_content.strip() == "":
                self.logger.error("SageMaker endpoint returned empty content")
                return "ขออภัยค่ะ ไม่สามารถสร้างคำตอบได้ในขณะนี้"
            
            # Extract only the answer block if present
            answer_match = re.search(r"```answer\s*(.*?)\s*```", raw_content, re.DOTALL)
            commentary = answer_match.group(1).strip() if answer_match else raw_content.strip()

            # Clean up any remaining Gemma formatting tokens
            commentary = re.sub(r'<start_of_turn>.*?<end_of_turn>', '', commentary, flags=re.DOTALL)
            commentary = commentary.strip()

            self.logger.info(f"Generated medical response - Commentary: {len(commentary)} chars")
            return commentary

        except Exception as e:
            self.logger.error(f"Error generating response from SageMaker: {str(e)}")
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
            context_parts.append(f"Previous Medical Considerations: {chain_of_thoughts}")

        recent_context = []
        for msg in chat_history[-3:]:
            if msg.get("role") == "user":
                recent_context.append(f"Patient: {msg.get('content', '')}")
            elif msg.get("role") == "assistant":
                recent_context.append(f"Previous Response: {msg.get('content', '')}")
        if recent_context:
            context_parts.append("Recent Conversation:\n" + "\n".join(recent_context))

        full_context = "\n\n".join(context_parts) if context_parts else ""

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

**Response Structure:**

```answer
[Concise, patient-friendly medical guidance with clear recommendations, appropriate disclaimers, and actionable next steps. Keep professional yet empathetic tone.]
```"""

        messages = self._format_messages(prompt, full_context)
        return self._generate_response_with_thinking(messages)

    def cleanup(self):
        """Delete SageMaker endpoint to avoid charges"""
        try:
            self.model.predictor.delete_endpoint()
            self.logger.info("SageMaker endpoint deleted successfully.")
        except Exception as e:
            self.logger.error(f"Error deleting endpoint: {e}")


if __name__ == "__main__":
    # Minimal reproducible demo for DoctorAgent using SageMaker
    # Requires AWS_ROLE_ARN in your environment, otherwise the class will raise.

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
    
    try:
        medical_response = doctor.analyze(
            user_message=user_message,
            chat_history=chat_history,
            chain_of_thoughts=chain_of_thoughts,
            summarized_histories=summarized_histories
        )
        
        print(medical_response)
        print("\n=== TEST COMPLETED ===")
    except Exception as e:
        print(f"Error during analysis: {e}")
    finally:
        # Clean up SageMaker endpoint
        print("\n=== CLEANING UP SAGEMAKER ENDPOINT ===")
        doctor.cleanup()