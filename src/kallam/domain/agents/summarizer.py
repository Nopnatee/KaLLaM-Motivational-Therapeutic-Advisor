import os
import json
import logging
import requests
import re
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Literal, Any, Optional

from dotenv import load_dotenv
load_dotenv()


class SummarizerAgent:
    SummaryType = Literal["conversation", "medical_session", "health_insights", "progress_report"]
    SummaryLength = Literal["brief", "detailed", "comprehensive"]

    def __init__(self, log_level: int = logging.INFO):
        self._setup_logging(log_level)
        self._setup_api_clients()
        
        self.logger.info("Summarizer Agent initialized successfully")

        self.system_prompt = """
**Your Role:**
You are a Medical Conversation Summarizer AI specializing in healthcare and mental health conversation analysis. Your role is to create concise, medically-relevant summaries of patient-doctor interactions.

**Core Rules:**
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

**Response Guideline:**
- Create structured summaries with clear timelines
- Use both Thai and English as appropriate to conversation content
- Focus on medical and psychological relevance
- Provide actionable insights for healthcare continuity
- Avoid redundant information from previous summaries

**Language Guidelines:**
- Respond in Thai and English as contextually appropriate
- Use professional medical terminology accurately
- Maintain cultural sensitivity in health communication
- Ensure clarity for healthcare providers and patients

Remember: Your summaries support continuity of care and help healthcare providers understand patient history and progress patterns.
"""

    def _setup_logging(self, log_level: int) -> None:
        """Setup logging configuration"""
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        
        self.logger = logging.getLogger(f"{__name__}.SummarizerAgent")
        self.logger.setLevel(log_level)
        
        if self.logger.handlers:
            self.logger.handlers.clear()
        
        file_handler = logging.FileHandler(
            log_dir / f"summarizer_{datetime.now().strftime('%Y%m%d')}.log",
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
                
            self.logger.info("SEA-Lion API client initialized for Summarizer Agent")
                
        except Exception as e:
            self.logger.error(f"Failed to initialize API clients: {str(e)}")
            raise

    def _format_messages(self, prompt: str, context: str = "") -> List[Dict[str, str]]:
        """Format messages for API call"""
        now = datetime.now()
        
        context_info = f"""
**Current Context:**
- Date/Time: {now.strftime("%Y-%m-%d %H:%M:%S")}
- Summarization Context: {context}
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
                "temperature": 0.2,  # Lower temperature for consistent summarization
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
                return "ไม่สามารถสร้างสรุปได้ในขณะนี้"
            
            choice = response_data["choices"][0]
            if "message" not in choice or "content" not in choice["message"]:
                self.logger.error(f"Unexpected message structure: {choice}")
                return "ไม่สามารถสร้างสรุปได้ในขณะนี้"
                
            raw_content = choice["message"]["content"]
            
            if raw_content is None or (isinstance(raw_content, str) and raw_content.strip() == ""):
                self.logger.error("SEA-Lion API returned None or empty content")
                return "ไม่สามารถสร้างสรุปได้ในขณะนี้"
            
            # Extract thinking and answer blocks
            thinking_match = re.search(r"```thinking\s*(.*?)\s*```", raw_content, re.DOTALL)
            answer_match = re.search(r"```answer\s*(.*?)\s*```", raw_content, re.DOTALL)
            
            reasoning = thinking_match.group(1).strip() if thinking_match else None
            final_answer = answer_match.group(1).strip() if answer_match else raw_content.strip()
            
            if reasoning:
                self.logger.debug(f"Summarizer reasoning:\n{reasoning}")
            
            self.logger.info(f"Generated summary (length: {len(final_answer)} chars)")
            return final_answer
            
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Error generating response from SEA-Lion API: {str(e)}")
            return "เกิดปัญหาในการเชื่อมต่อ กรุณาลองใหม่อีกครั้งค่ะ"
        except Exception as e:
            self.logger.error(f"Error generating response: {str(e)}")
            return "เกิดข้อผิดพลาดในการสร้างสรุป"

    def _format_history_for_summarization(self, history: List[Dict[str, str]]) -> str:
        """Format conversation history for summarization"""
        if not history:
            return "No conversation history provided"
        
        formatted_history = []
        for msg in history:
            role = msg.get('role', 'unknown')
            content = msg.get('content', '')
            formatted_history.append(f"{role}: {content}")
        
        return "\n".join(formatted_history)

    # Public methods
    def summarize_conversation_history(
        self, 
        response_history: List[Dict[str, str]], 
        summarized_histories: Optional[List[Dict[str, str]]] = None,
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
        
        # Format histories
        history_text = self._format_history_for_summarization(response_history) if response_history else ""
        summarized_text = self._format_history_for_summarization(summarized_histories) if summarized_histories else ""
        
        current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        prompt = f"""
Your Task:
Summarize the given chat history into a short paragraph including all key events.

Input Format:
- chat_history (For content): {history_text}
- summarized_history (For repetitive context): {summarized_text}
- current_time: {current_time}

Requirements:
- Keep summary concise with all key events and important details
- Include time/date references (group close dates/times together)
- Use timeline format if history is very long
- Summarize in Thai and English on separate paragraphs
- Return "None" if insufficient information
- Track patient's progress and health concerns
- Do not summarize the summarized_histories, only use it for repetitive context
- Do not include repetitive information according to summarized_histories
- In case of the information is already similar to the summarized_histories, just say ไม่มีข้อมูลใหม่ที่จำเป็นต้องสรุปเพิ่มเติมจากวันที่... (No new information to summarize from date...) without providing any reasons

Response Format:
[Summarized content]

**Focus on:**
- Medical symptoms and health concerns discussed
- Treatment recommendations and patient responses
- Emotional state and psychological well-being
- Progress in health status or treatment compliance
- Any red flags or concerning developments
- Patient engagement and understanding levels
"""
        
        context = f"Conversation summarization from {current_time}"
        messages = self._format_messages(prompt, context)
        return self._generate_response(messages)

    def summarize_medical_session(
        self, 
        session_history: List[Dict[str, str]], 
        session_type: str = "general",
        focus_areas: Optional[List[str]] = None,
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
        
        history_text = self._format_history_for_summarization(session_history)
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

**Structure Requirements:**
- Use professional medical documentation format
- Include timestamps and session context
- Focus on actionable clinical information
- Maintain clear, concise language
- Ensure continuity of care focus
"""
        
        context = f"Medical session summary: {session_type} session on {current_time}"
        messages = self._format_messages(prompt, context)
        return self._generate_response(messages)

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
        
        history_text = self._format_history_for_summarization(conversation_history)
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

**Analysis Focus:**
- Pattern recognition across time periods
- Clinical significance of changes
- Behavioral health connections
- Risk stratification and prevention
- Patient engagement and motivation trends
"""
        
        context = f"Health insights analysis for {time_period} period"
        messages = self._format_messages(prompt, context)
        return self._generate_response(messages)

    def create_progress_report(
        self, 
        baseline_summary: str, 
        recent_conversations: List[Dict[str, str]], 
        metrics_focus: Optional[List[str]] = None,
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
        
        recent_text = self._format_history_for_summarization(recent_conversations)
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

**Report Standards:**
- Evidence-based progress assessment
- Clear comparison metrics
- Actionable clinical recommendations
- Timeline-based progress tracking
- Goal-oriented outcome measures
"""
        
        context = f"Progress report generation comparing baseline to current status"
        messages = self._format_messages(prompt, context)
        return self._generate_response(messages)

    def generate_clinical_notes(
        self,
        session_content: str,
        session_type: str = "consultation",
        clinical_focus: Optional[List[str]] = None,
        language: str = "english"
    ) -> str:
        """
        Generate structured clinical notes for medical documentation
        
        Args:
            session_content: Content of the clinical session
            session_type: Type of clinical session
            clinical_focus: Specific clinical areas to emphasize
            language: Language for clinical notes
            
        Returns:
            Structured clinical notes
        """
        
        focus_areas = ", ".join(clinical_focus) if clinical_focus else "general clinical assessment"
        current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        prompt = f"""
Your Task:
Generate structured clinical notes for medical documentation and continuity of care.

Clinical Session Data:
- Session Content: {session_content}
- Session Type: {session_type}
- Clinical Focus: {focus_areas}
- Documentation Date: {current_time}

Clinical Documentation Requirements:
- Follow standard medical documentation format
- Include subjective and objective findings
- Document assessment and clinical reasoning
- Provide clear treatment plan recommendations
- Note any changes from previous assessments
- Include risk factors and safety considerations

SOAP Note Structure:
**Subjective:** Patient's reported symptoms, concerns, and history
**Objective:** Observable findings, mental status, and clinical observations
**Assessment:** Clinical interpretation and diagnostic considerations
**Plan:** Treatment recommendations, follow-up, and next steps

Medical Documentation Standards:
- Use appropriate medical terminology
- Maintain professional clinical language
- Include relevant timeline information
- Document patient understanding and consent
- Note any contraindications or concerns
- Specify follow-up requirements

Output Format:
**CLINICAL NOTES - {session_type.upper()}**
**Date:** {current_time}

**SUBJECTIVE:**
[Patient-reported information, symptoms, concerns]

**OBJECTIVE:**
[Clinical observations, mental status, behavioral observations]

**ASSESSMENT:**
[Clinical interpretation, severity assessment, diagnostic considerations]

**PLAN:**
[Treatment recommendations, medications, lifestyle modifications, follow-up]

**ADDITIONAL NOTES:**
[Risk factors, safety considerations, patient education provided]

Respond in {language} using appropriate clinical documentation standards.

**Documentation Focus:**
- Clinical accuracy and completeness
- Legal and regulatory compliance
- Continuity of care support
- Professional medical standards
- Patient safety considerations
"""
        
        context = f"Clinical notes generation for {session_type} session"
        messages = self._format_messages(prompt, context)
        return self._generate_response(messages)


if __name__ == "__main__":
    # Test the Summarizer Agent
    try:
        summarizer = SummarizerAgent(log_level=logging.DEBUG)
        
        # Test conversation summary
        print("=== TEST: CONVERSATION SUMMARY ===")
        chat_history = [
            {"role": "user", "content": "I've been having headaches for the past week"},
            {"role": "assistant", "content": "Tell me more about these headaches - when do they occur and how severe are they?"},
            {"role": "user", "content": "They're usually worse in the afternoon, around 7/10 pain level"},
            {"role": "assistant", "content": "That sounds concerning. Have you noticed any triggers like stress, lack of sleep, or screen time?"}
        ]
        
        summary = summarizer.summarize_conversation_history(
            response_history=chat_history,
            language="english"
        )
        print(summary)
        
        # Test medical session summary
        print("\n=== TEST: MEDICAL SESSION SUMMARY ===")
        session_summary = summarizer.summarize_medical_session(
            session_history=chat_history,
            session_type="consultation",
            focus_areas=["headache assessment", "symptom analysis"],
            language="english"
        )
        print(session_summary)
        
    except Exception as e:
        print(f"Error testing Summarizer Agent: {e}")