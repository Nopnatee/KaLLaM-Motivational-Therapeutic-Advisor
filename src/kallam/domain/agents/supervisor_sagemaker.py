# pip install "strands-agents" "boto3" "pydantic" "sagemaker" "python-dotenv"
# Don't forget to set up your AWS credentials in environment variables or AWS config file.
# You can skip supervisor.cleanup() if you want to keep the endpoint running.

import os
import json
from dotenv import load_dotenv
load_dotenv()  # Load environment variables from a .env file if present

from typing import Literal, List, Dict, Any
from strands import Agent, tool
import sagemaker
from sagemaker.huggingface import HuggingFaceModel

# --------------------------
# AWS & Hugging Face Setup
# --------------------------
AWS_ROLE_ARN = os.getenv("AWS_ROLE_ARN")  # SageMaker execution role
AWS_REGION   = os.getenv("AWS_DEFAULT_REGION", "ap-southeast-2")
HF_MODEL_ID  = "aisingapore/Llama-SEA-LION-v3-8B-IT"
HF_TASK      = "text-generation"

# --------------------------
# SageMaker Model Wrapper (Strands Compatible)
# --------------------------
class SageMakerModelWrapper:
    def __init__(self, predictor):
        self.predictor = predictor
        self.max_tokens = 512
        self.temperature = 0.7
        
    def generate(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """
        Strands Agent expects this interface for generate method
        messages: List of message dictionaries with 'role' and 'content' keys
        """
        # Convert messages to a single prompt string
        prompt = self._messages_to_prompt(messages)
        
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
            return f"Error generating response: {str(e)}"
    
    def _messages_to_prompt(self, messages: List[Dict[str, str]]) -> str:
        """Convert messages format to a single prompt string"""
        prompt_parts = []
        
        for message in messages:
            role = message.get('role', '')
            content = message.get('content', '')
            
            if role == 'system':
                prompt_parts.append(f"System: {content}")
            elif role == 'user':
                prompt_parts.append(f"User: {content}")
            elif role == 'assistant':
                prompt_parts.append(f"Assistant: {content}")
            else:
                prompt_parts.append(content)
        
        return "\n\n".join(prompt_parts) + "\n\nAssistant:"

# --------------------------
# Supervisor Agent
# --------------------------
class SupervisorAgent:
    SpecialistName = Literal["math_specialist", "code_specialist", "research_specialist"]

    @staticmethod
    @tool
    def update_max_tokens(max_tokens: int, agent: Agent) -> str:
        """Update the max tokens for the agent's model"""
        if hasattr(agent.model, 'max_tokens'):
            agent.model.max_tokens = max_tokens
            return f"Supervisor max_tokens updated to {max_tokens}"
        return "Could not update max_tokens"

    def __init__(self, aws_role_arn: str = AWS_ROLE_ARN, region: str = AWS_REGION):
        if not aws_role_arn:
            raise ValueError("AWS_ROLE_ARN is required. Please set it in your environment variables.")
        
        # Verify AWS credentials before proceeding
        try:
            import boto3
            sts_client = boto3.client('sts', region_name=region)
            identity = sts_client.get_caller_identity()
            print(f"AWS Identity verified: {identity.get('Arn', 'Unknown')}")
        except Exception as e:
            raise ValueError(f"AWS credentials error: {e}. Please check your AWS configuration.")
        
        # Create SageMaker session
        sess = sagemaker.Session()

        # Deploy Hugging Face model
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

        print("Deploying model to SageMaker endpoint...")
        predictor = hf_model.deploy(
            initial_instance_count=1,
            instance_type="ml.g4dn.xlarge" # Instance type can be adjusted
        )
        print("Model deployed successfully!")

        # Create the model wrapper
        self.model = SageMakerModelWrapper(predictor)

        # System prompt
        system_prompt = """\
You are the SupervisorAgent. Read the user's request and decide which specialists to activate according to the **Flags Schema:** or finalize the answer given the suggestion by other agents in structured string.

**Rules:**
- Prefer 1–2 specialists unless clearly multi-domain.
- You are an expert in routing requests to the right specialists.
- Always respond according to the **Output Schema**.

**Flags Schema:**
{
  "language": "[detected language in lowercase, e.g. 'thai', 'english']",
  "doctor": "[true/false]",
  "psychologist": "[true/false]"
}

Respond only with valid JSON matching the schema above.
"""
        
        # Create Strands Agent with the correct model
        self.agent = Agent(
            model=self.model,  # Pass the model wrapper, not self
            system_prompt=system_prompt,
            tools=[self.update_max_tokens],
            callback_handler=None,
        )

    # --------------------------
    # Public methods
    # --------------------------
    def decide(self, user_message: str) -> str:
        """Make a decision about which specialists to activate"""
        prompt = f"User request: {user_message}\nDecide which specialists to activate and respond only in structured format according to the given **Flags Schema:**."
        
        # Use the agent's run method
        response = self.agent.run(prompt)
        return response

    def conclude(self, context: str, language: str) -> str:
        """Synthesize a final response"""
        prompt = f"Synthesize a response given collected context: {context}\nRespond only in {language} language."
        
        # Use the agent's run method
        response = self.agent.run(prompt)
        return response

    def cleanup(self):
        """Delete SageMaker endpoint to avoid charges"""
        try:
            self.model.predictor.delete_endpoint()
            print("SageMaker endpoint deleted successfully.")
        except Exception as e:
            print(f"Error deleting endpoint: {e}")

# --------------------------
# Example usage
# --------------------------
if __name__ == "__main__":
    try:
        supervisor = SupervisorAgent()

        user_input = "I have a headache and feel anxious about my exams."
        print(f"User input: {user_input}")
        
        decision = supervisor.decide(user_input)
        print("Decision:", decision)

        context = "Based on the analysis, it seems you might be experiencing stress-related symptoms."
        final_response = supervisor.conclude(context, language="english")
        print("Final Response:", final_response)

    except Exception as e:
        print(f"Error: {e}")
    finally:
        # Clean up SageMaker endpoint
        if 'supervisor' in locals():
            supervisor.cleanup()