# summarizer_agent.py
# pip install "strands-agents" "boto3" "pydantic" "python-dotenv" "google-genai"

from typing import List, Dict, Any, Optional
import json
import os
from datetime import datetime
from dotenv import load_dotenv
load_dotenv()  # Load environment variables from a .env file if present

from strands import Agent, tool
from strands.models import BedrockModel
from botocore.config import Config as BotocoreConfig

# Import Gemini for summarization
from google import genai

# AWS credentials from environment
access_key = os.getenv("AWS_ACCESS_KEY_ID")
secret_key = os.getenv("AWS_SECRET_ACCESS_KEY")
region = os.getenv("AWS_DEFAULT_REGION")

# Model configuration
MODEL_ID = "Llama-SEA-LION-v3-8B-IT"
REGION   = "ap-southeast-2"

GUARDRAIL_ID      = None
GUARDRAIL_VERSION = None

# Gemini configuration
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL_NAME = "gemini-2.5-flash-preview-05-20"

# --------------------------
# Summarizer Agent
# --------------------------
class SummarizerAgent:

    def __init__(
        self,
        model_id: str = MODEL_ID,
        region: str = REGION,
        guardrail_id: str = None,
        guardrail_version: str = None,
        gemini_api_key: str = GEMINI_API_KEY,
        gemini_model_name: str = GEMINI_MODEL_NAME,
    ):
        # Setup AWS Bedrock model (for potential future use or consistency)
        boto_cfg = BotocoreConfig(
            retries={"max_attempts": 3, "mode": "standard"},
            connect_timeout=5,
            read_timeout=60,
        )

        bedrock_model = BedrockModel(
            model_id=model_id,
            region_name=region,
            streaming=False,
            temperature=0.2,  # Lower temperature for consistent summarization
            top_p=0.8,
            stop_sequences=["</END>"],

            guardrail_id=guardrail_id,
            guardrail_version=guardrail_version,
            guardrail_trace="enabled",
            guardrail_stream_processing_mode="sync",
            guardrail_redact_input=True,
            guardrail_redacted_input_message="[User input redacted due to privacy policy]",
            guardrail_redact_output=False,

            cache_prompt="default",
            cache_tools="default",
            boto_client_config=boto_cfg,
        )

        # Setup Gemini client for actual summarization work
        if not gemini_api_key:
            raise ValueError("GEMINI_API_KEY is required for summarization functionality")
        
        self.gemini_client = genai.Client(api_key=gemini_api_key)
        self.gemini_model_name = gemini_model_name

        system_prompt = """\
You are a Medical Conversation Summarizer AI specializing in healthcare and mental health conversation analysis. Your role is to create concise, medically-relevant summaries of patient-doctor interactions.

**Your Capabilities:**
- Summarize healthcare conversations while preserving critical medical information
- Track patient progress and health status changes over time
- Identify key symptoms, treatments, and patient responses
- Maintain patient confidentiality and medical privacy
- Extract actionable health insights from conversation histories
- Organize information chronologically for continuity of care

**Summarization Principles:**
- Preserve all medically relevant information
- Include patient's emotional state and psychological well-being
- Track symptom progression and treatment responses
- Note patient compliance and engagement levels
- Identify patterns in health behaviors and concerns
- Maintain professional medical terminology where appropriate

**Privacy and Ethics:**
- Protect patient confidentiality in all summaries
- Use appropriate medical privacy guidelines
- Avoid including personally identifiable information beyond medical relevance
- Maintain professional boundaries in summary content

**Output Requirements:**
- Create structured summaries with clear timelines
- Use both Thai and English as appropriate to conversation content
- Focus on medical and psychological relevance
- Provide actionable insights for healthcare continuity
- Avoid redundant information from previous summaries

Remember: Your summaries support continuity of care and help healthcare providers understand patient history and progress patterns.
"""

        self.agent = Agent(
            model=bedrock_model,
            system_prompt=system_prompt,
            tools=[],
            callback_handler=None,
        )

    def _generate_summary_with_gemini(self, prompt: str) -> str:
        """
        Generate summary using Gemini API directly
        
        Args:
            prompt: The summarization prompt
            
        Returns:
            Generated summary text
        """
        try:
            response = self.gemini_client.models.generate_content(
                model=self.gemini_model_name,
                contents=[prompt],
            )
            
            response_text = response.text
            
            # Check if response is None or empty
            if response_text is None:
                return "ไม่สามารถสร้างสรุปได้ในขณะนี้"
            
            if isinstance(response_text, str) and response_text.strip() == "":
                return "ไม่สามารถสร้างสรุปได้ในขณะนี้"
            
            return str(response_text).strip()
            
        except Exception as e:
            return f"เกิดข้อผิดพลาดในการสร้างสรุป: {str(e)}"

    # --------------------------
    # Public methods
    # --------------------------
    def summarize_conversation_history(
        self, 
        response_history: List[Dict[str, str]], 
        summarized_histories: List[Dict[str, str]] = None,
        language: str = "thai"
    ) -> str:
        """
        Summarize conversation history following the chatbot_prompt.py structure
        
        Args:
            response_history: Full conversation history to summarize
            summarized_histories: Previous summarized histories for context
            language: Primary language for the summary
            
        Returns:
            Summarized conversation history
        """
        
        # Convert histories to text format similar to chatbot_prompt.py
        history_text = ""
        if response_history:
            history_text = "\n".join([
                f"{msg.get('role', 'unknown')}: {msg.get('content', '')}" 
                for msg in response_history
            ])
        
        summarized_text = ""
        if summarized_histories:
            summarized_text = "\n".join([
                f"{msg.get('role', 'unknown')}: {msg.get('content', '')}" 
                for msg in summarized_histories
            ])
        
        current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        # Create the summarization prompt following the exact structure from chatbot_prompt.py
        summary_prompt = f"""
Your Task:
Summarize the given chat history into a short paragraph including all key events.

Input Format:
-chat_history (For content): {history_text}
-summarized_history (For repetitive context): {summarized_text}
-current_time: {current_time}

Requirements:
- Keep summary concise with all key events and important details
- Include time/date references (group close dates/times together)
- Use timeline format if history is very long
- Summarize in Thai and English on separate paragraphs
- Return "None" if insufficient information
- Track patient's progress and health concerns
- Do not summarize the summarized_histories, only use it for repetitive context
- Do not include repetitive information according to summarized_histories.
- In case of the information is already similar to the summarized_histories, just say ไม่มีข้อมูลใหม่ที่จำเป็นต้องสรุปเพิ่มเติมจากวันที่... (No new information to summarize from date...) without providing any reasons.

Response Format:
[Summarized content]
"""
        
        # Use Gemini API for summarization
        result = self._generate_summary_with_gemini(summary_prompt)
        return result

    def summarize_medical_session(
        self, 
        session_history: List[Dict[str, str]], 
        session_type: str = "general",
        focus_areas: List[str] = None,
        language: str = "thai"
    ) -> str:
        """
        Summarize a specific medical or psychological session
        
        Args:
            session_history: Session conversation history
            session_type: Type of session (general, emergency, therapy, follow-up)
            focus_areas: Specific areas to focus on in summary
            language: Language for summary output
            
        Returns:
            Session summary
        """
        
        history_text = "\n".join([
            f"{msg.get('role', 'unknown')}: {msg.get('content', '')}" 
            for msg in session_history
        ])
        
        focus_text = ", ".join(focus_areas) if focus_areas else "general medical consultation"
        current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        prompt = f"""
Your Task:
Create a medical session summary for healthcare continuity.

Session Information:
- Session Type: {session_type}
- Focus Areas: {focus_text}
- Session History: {history_text}
- Session Time: {current_time}

Medical Summary Requirements:
- Extract key symptoms, complaints, and patient concerns
- Document treatment recommendations and patient responses
- Note patient's emotional state and psychological well-being
- Track any medication discussions or lifestyle recommendations
- Include patient compliance and engagement level
- Identify any red flags or urgent concerns
- Note follow-up plans or referrals mentioned

Healthcare Documentation Standards:
- Use clear, professional medical language
- Maintain patient confidentiality
- Focus on clinically relevant information
- Structure information chronologically
- Include both physical and mental health aspects
- Document patient's understanding and agreement with plans

Output Format:
**Session Summary ({session_type.title()}):**

**Chief Concerns:** [Main symptoms/issues presented]

**Clinical Assessment:** [Key findings and observations]

**Treatment Discussion:** [Recommendations and patient responses]

**Psychological Status:** [Mental health and emotional state]

**Follow-up Plan:** [Next steps and recommendations]

**Clinical Notes:** [Important observations for continuity]

Respond primarily in {language} with appropriate medical terminology.
"""
        
        result = self._generate_summary_with_gemini(prompt)
        return result

    def extract_health_insights(
        self, 
        conversation_history: List[Dict[str, str]], 
        time_period: str = "recent",
        language: str = "thai"
    ) -> str:
        """
        Extract health insights and patterns from conversation history
        
        Args:
            conversation_history: Patient conversation history
            time_period: Time period for analysis (recent, weekly, monthly)
            language: Language for output
            
        Returns:
            Health insights and patterns
        """
        
        history_text = "\n".join([
            f"{msg.get('role', 'unknown')}: {msg.get('content', '')}" 
            for msg in conversation_history
        ])
        
        current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        prompt = f"""
Your Task:
Analyze patient conversation history to extract meaningful health insights and patterns.

Analysis Parameters:
- Time Period: {time_period}
- Conversation History: {history_text}
- Analysis Date: {current_time}

Health Pattern Analysis:
- Identify recurring symptoms or health concerns
- Track symptom progression or improvement over time
- Note patterns in patient mood and mental health
- Analyze treatment compliance and effectiveness
- Identify lifestyle factors affecting health
- Extract patient's health goals and motivations
- Note any concerning trends or warning signs

Psychological Pattern Analysis:
- Track emotional states and mood patterns
- Identify stress triggers and coping mechanisms
- Note social support and relationship impacts
- Analyze motivation levels and engagement changes
- Identify behavioral patterns affecting health

Actionable Insights:
- Suggest areas needing focused attention
- Identify successful interventions to continue
- Note areas where patient needs additional support
- Recommend preventive measures or early interventions
- Highlight positive progress and improvements

Output Format:
**Health Insights Summary ({time_period.title()} Period):**

**Symptom Patterns:** [Recurring or evolving symptoms]

**Mental Health Trends:** [Emotional and psychological patterns]

**Treatment Response:** [Effectiveness of current approaches]

**Risk Factors:** [Areas of concern requiring attention]

**Positive Progress:** [Improvements and successes]

**Recommendations:** [Suggested focus areas for future care]

Respond primarily in {language} with actionable healthcare insights.
"""
        
        result = self._generate_summary_with_gemini(prompt)
        return result

    def create_progress_report(
        self, 
        baseline_summary: str, 
        recent_conversations: List[Dict[str, str]], 
        metrics_focus: List[str] = None,
        language: str = "thai"
    ) -> str:
        """
        Create a progress report comparing baseline to current status
        
        Args:
            baseline_summary: Initial patient assessment or previous summary
            recent_conversations: Recent conversation history
            metrics_focus: Specific metrics to focus on (symptoms, mood, compliance)
            language: Language for output
            
        Returns:
            Progress report comparing baseline to current status
        """
        
        recent_text = "\n".join([
            f"{msg.get('role', 'unknown')}: {msg.get('content', '')}" 
            for msg in recent_conversations
        ])
        
        focus_metrics = ", ".join(metrics_focus) if metrics_focus else "overall health and well-being"
        current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        prompt = f"""
Your Task:
Create a comprehensive progress report comparing patient's baseline status to current condition.

Progress Analysis Data:
- Baseline Summary: {baseline_summary}
- Recent Conversations: {recent_text}
- Focus Metrics: {focus_metrics}
- Report Date: {current_time}

Progress Evaluation Areas:
- Symptom changes (improvement, stability, or worsening)
- Mental health and emotional well-being changes
- Treatment compliance and engagement levels
- Lifestyle modifications and their impacts
- Goal achievement and progress toward targets
- New concerns or challenges that have emerged
- Social support and relationship impacts

Quantitative Assessment (where possible):
- Rate improvements on relevant scales
- Track frequency of symptom occurrences
- Monitor engagement and participation levels
- Measure goal achievement percentages
- Note any measurable lifestyle changes

Clinical Recommendations:
- Continue effective interventions
- Modify approaches that aren't working
- Address new or emerging concerns
- Set updated goals based on progress
- Plan next steps for continued improvement

Output Format:
**Patient Progress Report:**

**Baseline Status:** [Initial condition summary]

**Current Status:** [Present condition assessment]

**Key Improvements:** [Positive changes observed]

**Areas of Concern:** [Issues requiring attention]

**Treatment Effectiveness:** [What's working and what isn't]

**Updated Goals:** [Revised objectives based on progress]

**Next Steps:** [Recommended actions going forward]

**Overall Assessment:** [Summary of progress trajectory]

Respond primarily in {language} with clear progress indicators and actionable recommendations.
"""
        
        result = self._generate_summary_with_gemini(prompt)
        return result