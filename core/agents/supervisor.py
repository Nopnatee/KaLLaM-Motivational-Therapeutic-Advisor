# supervisor_agent_sagemaker.py
# pip install "strands-agents" "boto3" "pydantic" "sagemaker" "python-dotenv"

import os
import json
from dotenv import load_dotenv
load_dotenv()  # Load environment variables from a .env file if present

from typing import Literal
from strands import Agent, tool
import sagemaker
from sagemaker.huggingface import HuggingFaceModel

# --------------------------
# AWS & Hugging Face Setup
# --------------------------
AWS_ROLE_ARN = os.getenv("AWS_ROLE_ARN")  # SageMaker execution role
AWS_REGION   = os.getenv("AWS_DEFAULT_REGION", "ap-southeast-2")
HF_MODEL_ID  = "aisingapore/Llama-SEA-LION-v3-8B-IT"
HF_TASK      = "text-generation"  # adjust if model requires text2text-generation

# --------------------------
# SageMaker Model Wrapper
# --------------------------
class SageMakerModelWrapper:
    def __init__(self, predictor):
        self.predictor = predictor

    def generate(self, prompt: str, max_length: int = 512, **kwargs):
        response = self.predictor.predict({
            "inputs": prompt,
            "parameters": {"max_new_tokens": max_length}
        })
        # HF models return a list of dicts
        if isinstance(response, list) and "generated_text" in response[0]:
            return response[0]["generated_text"]
        return response

# --------------------------
# Supervisor Agent
# --------------------------
class SupervisorAgent:
    SpecialistName = Literal["math_specialist", "code_specialist", "research_specialist"]

    @staticmethod
    @tool
    def update_max_tokens(max_tokens: int, agent: Agent) -> str:
        agent.model.max_tokens = max_tokens
        return f"Supervisor max_tokens updated to {max_tokens}"

    def __init__(self, aws_role_arn: str = AWS_ROLE_ARN, region: str = AWS_REGION):
        # Create SageMaker session
        sess = sagemaker.Session()

        # Deploy Hugging Face model
        hf_model = HuggingFaceModel(
            model_data=None,
            role=aws_role_arn,
            transformers_version="4.32.0",
            pytorch_version="2.1.0",
            py_version="py310",
            env={
                "HF_MODEL_ID": HF_MODEL_ID,
                "HF_TASK": HF_TASK
            },
            sagemaker_session=sess
        )

        predictor = hf_model.deploy(
            initial_instance_count=1,
            instance_type="ml.g5.2xlarge"
        )

        self.model = SageMakerModelWrapper(predictor)

        # System prompt
        system_prompt = """\
You are the SupervisorAgent. Read the user's request and decide which specialists to activate according to the **Flags Schema:** or finalize the answer given the suggestion by other agents in structured string.
**Rules:**
- Prefer 1–2 specialists unless clearly multi-domain.
- You are an expert in routing requests to the right specialists.
- Always respond **Output Schema:**.

**Flags Schema:**
{
  "language": [detected language in lowercase, e.g. "thai", "english"],
  "doctor": [true/false],
  "psychologist": [true/false]
}
"""
        # Create Strands Agent
        self.agent = Agent(
            model=self,
            system_prompt=system_prompt,
            tools=[self.update_max_tokens],
            callback_handler=None,
        )

    # --------------------------
    # Public methods
    # --------------------------
    def decide(self, user_message: str) -> str:
        prompt = f"User request: {user_message}\nDecide which specialists to activate and respond only in structured format according to the given **Flags Schema:**."
        return self.model.generate(prompt)

    def conclude(self, context: str, language: str) -> str:
        prompt = f"synthesize a response given collected context: {context} Respond only in {language} language."
        return self.model.generate(prompt)

    def cleanup(self):
        # Delete SageMaker endpoint to avoid charges
        self.model.predictor.delete_endpoint()

# --------------------------
# Example usage
# --------------------------
if __name__ == "__main__":
    supervisor = SupervisorAgent()

    user_input = "I have a headache and feel anxious about my exams."
    decision = supervisor.decide(user_input)
    print("Decision:", decision)

    context = "Based on the analysis, it seems you might be experiencing stress-related symptoms."
    final_response = supervisor.conclude(context, language="english")
    print("Final Response:", final_response)

    # Clean up SageMaker endpoint
    supervisor.cleanup()
