# supervisor_agent.py
# pip install "strands-agents" "boto3"

from typing import List, Literal

from strands import Agent, tool
from strands.models import BedrockModel
from botocore.config import Config as BotocoreConfig


class SupervisorAgent:
    """Supervisor agent powered by Amazon Bedrock SEA-Lion.
    - Produces structured decisions on which specialists to activate.
    - Can synthesize a final response from collected context.
    """

    # ------------------------------------------------------------------
    # Structured Output Schema (plain Python)
    # ------------------------------------------------------------------
    SpecialistName = Literal["math_specialist", "code_specialist", "research_specialist"]

    class ActivationFlag:
        def __init__(self, specialist: "SupervisorAgent.SpecialistName", reason: str):
            self.specialist = specialist
            self.reason = reason

    class SupervisorDecision:
        def __init__(
            self,
            plan: str,
            activate: List["SupervisorAgent.ActivationFlag"],
            final_answer_allowed: bool
        ):
            self.plan = plan
            self.activate = activate
            self.final_answer_allowed = final_answer_allowed

    # ------------------------------------------------------------------
    # Tools for runtime configuration updates
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------
    def __init__(
        self,
        model_id: str = "us.YOUR-SEALION-MODEL-ID",
        region: str = "us-east-1",
        guardrail_id: str = None,
        guardrail_version: str = None,
    ):
        boto_cfg = BotocoreConfig(
            retries={"max_attempts": 3, "mode": "standard"},
            connect_timeout=5,
            read_timeout=60,
        )

        supervisor_llm = BedrockModel(
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
You are the SupervisorAgent. Read the user's request and decide which specialists to activate.
Rules:
- Prefer 1–2 specialists unless clearly multi-domain.
- If trivial/general, set final_answer_allowed=true and activate=[].
- When in doubt about math -> math_specialist. Code -> code_specialist. Factual info -> research_specialist.
- Always produce a succinct plan.
End messages with </END>.
"""

        self.agent = Agent(
            name="SupervisorAgent",
            system_prompt=system_prompt,
            model=supervisor_llm,
            tools=[self.update_model_id, self.update_temperature],
            callback_handler=None,
        )

    # ------------------------------------------------------------------
    # Public methods
    # ------------------------------------------------------------------
    def decide(self, user_message: str) -> "SupervisorDecision":
        """Ask the supervisor for a structured decision on which agents to activate."""
        return self.agent.structured_output(
            self.SupervisorDecision,
            f"User request: {user_message}\nDecide which specialists to activate."
        )

    def conclude(self, context: str) -> str:
        """Use the supervisor to synthesize a final answer given collected context."""
        return str(self.agent(context))
