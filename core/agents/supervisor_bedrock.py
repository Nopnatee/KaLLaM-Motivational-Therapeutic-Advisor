# supervisor_agent.py
# pip install "strands-agents" "boto3" "pydantic"

from typing import List, Literal
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
# Supervisor Agent (Need fixing)
# --------------------------
class SupervisorAgent:
    SpecialistName = Literal["math_specialist", "code_specialist", "research_specialist"]

    @staticmethod
    @tool
    def update_model_id(model_id: str, agent: Agent) -> str:
        agent.model.update_config(model_id=model_id)
        return f"Supervisor model_id updated to {model_id}"

    @staticmethod
    @tool
    def update_temperature(temperature: float, agent: Agent) -> str:
        agent.model.update_config(temperature=temperature)
        return f"Supervisor temperature updated to {temperature}"

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
            temperature=0.2,
            top_p=0.8,
            stop_sequences=["</END>"],

            guardrail_id=guardrail_id,
            guardrail_version=guardrail_version,
            guardrail_trace="enabled",
            guardrail_stream_processing_mode="sync",
            guardrail_redact_input=True,
            guardrail_redact_input_message="[User input redacted due to guardrail policy]",
            guardrail_redact_output=False,

            cache_prompt="default",
            cache_tools="default",
            boto_client_config=boto_cfg,
        )

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

        self.agent = Agent(
            model=bedrock_model,
            system_prompt=system_prompt,
            tools=[self.update_model_id, self.update_temperature],
            callback_handler=None,
        )

    # --------------------------
    # Public methods
    # --------------------------
    def decide(self, user_message: str) -> str:
        decision_obj = self.agent(
            f"User request: {user_message}\nDecide which specialists to activate and respond only in structured format according to the given **Flags Schema:**."
        )
        return decision_obj

    def conclude(self, context: str, language: str) -> str:
        decision_obj = self.agent(
            f"synthesize a response given collected context: {context} Respond only in {language} language."
        )
        return decision_obj

# --------------------------
# Example usage
# --------------------------
if __name__ == "__main__":
    supervisor = SupervisorAgent(
        model_id=MODEL_ID,
        region=REGION,
        guardrail_id=GUARDRAIL_ID,
        guardrail_version=GUARDRAIL_VERSION
    )

    user_input = "I have a headache and feel anxious about my exams."
    decision = supervisor.decide(user_input)
    print("Decision:", decision)

    context = "Based on the analysis, it seems you might be experiencing stress-related symptoms."
    final_response = supervisor.conclude(context, language="english")
    print("Final Response:", final_response)
