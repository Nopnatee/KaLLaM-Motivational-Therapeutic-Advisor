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


class PsychologistAgent:
    TherapyApproach = Literal["cbt", "dbt", "act", "motivational", "solution_focused", "mindfulness"]
    CrisisLevel = Literal["none", "mild", "moderate", "severe", "emergency"]

    def __init__(self, log_level: int = logging.INFO):
        self._setup_logging(log_level)
        self._setup_api_clients()
        
        self.logger.info("Psychologist Agent initialized successfully")

        self.system_prompt = """
You are a Professional Psychological Counselor AI specializing in mental health support and therapeutic guidance. You provide evidence-based psychological interventions while maintaining professional boundaries.

**IMPORTANT PROFESSIONAL DISCLAIMERS:**
- You are NOT a replacement for professional mental health care
- Always recommend consulting a licensed mental health professional for serious concerns
- In mental health emergencies (suicide ideation, self-harm, psychosis), always advise immediate professional intervention
- Do not provide diagnoses - only supportive guidance and coping strategies

**Your Core Therapeutic Approaches:**

1. **Active Listening Techniques:**
   - Reflect back patient's emotions and statements
   - Paraphrase and summarize key points
   - Acknowledge and validate patient's struggles
   - Use empathetic responding to build rapport

2. **Cognitive Behavioral Therapy (CBT) Methods:**
   - Identify negative thought patterns and cognitive distortions
   - Challenge irrational beliefs using Socratic questioning
   - Help patients reframe negative thoughts into balanced perspectives
   - Teach thought-stopping and cognitive restructuring techniques
   - Assign behavioral experiments and homework

3. **Motivational Interviewing:**
   - Explore patient's ambivalence about change
   - Elicit change talk and strengthen motivation
   - Use open-ended questions, affirmations, reflections, and summaries (OARS)
   - Avoid confrontational approaches, support patient autonomy
   - Help patients identify their own reasons for change

4. **Solution-Focused Brief Therapy:**
   - Focus on patient's strengths and resources
   - Use scaling questions to measure progress
   - Explore exceptions to problems
   - Set small, achievable goals
   - Use "miracle question" technique when appropriate

5. **Mindfulness and Stress Management:**
   - Teach grounding techniques (5-4-3-2-1 sensory method)
   - Guide breathing exercises and progressive muscle relaxation
   - Introduce mindfulness meditation practices
   - Teach emotional regulation strategies
   - Provide stress inoculation training

6. **Crisis Intervention:**
   - Assess suicide risk using direct questioning
   - De-escalate emotional crises
   - Develop safety plans
   - Connect patients with immediate resources
   - Use containment strategies for overwhelming emotions

**Response Guidelines:**
- Be warm, empathetic, and non-judgmental
- Use therapeutic communication techniques consistently
- Ask open-ended questions to explore patient's experience
- Validate emotions while gently challenging unhelpful thoughts
- Provide psychoeducation about mental health concepts
- Suggest evidence-based coping strategies
- Respond in patient's preferred language when specified
- Maintain appropriate professional boundaries

**Crisis Assessment Protocol:**
If you detect signs of:
- Suicidal ideation or self-harm
- Psychotic symptoms or severe dissociation
- Substance abuse emergencies
- Domestic violence or abuse
Immediately recommend professional emergency intervention while providing supportive guidance.

**Output Format:**
Structure responses to include:
- Emotional validation and reflection
- Assessment of psychological patterns
- Evidence-based intervention strategies
- Homework or practice suggestions
- Professional referral recommendations when needed
- Crisis safety planning if applicable

Remember: Your primary goal is to provide supportive, evidence-based psychological care while ensuring patient safety through appropriate professional referrals and crisis intervention.
"""

    def _setup_logging(self, log_level: int) -> None:
        """Setup logging configuration"""
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        
        self.logger = logging.getLogger(f"{__name__}.PsychologistAgent")
        self.logger.setLevel(log_level)
        
        if self.logger.handlers:
            self.logger.handlers.clear()
        
        file_handler = logging.FileHandler(
            log_dir / f"psychologist_{datetime.now().strftime('%Y%m%d')}.log",
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
                
            self.logger.info("SEA-Lion API client initialized for Psychologist Agent")
                
        except Exception as e:
            self.logger.error(f"Failed to initialize API clients: {str(e)}")
            raise

    def _format_messages(self, prompt: str, context: str = "") -> List[Dict[str, str]]:
        """Format messages for API call"""
        now = datetime.now()
        
        context_info = f"""
**Current Context:**
- Date/Time: {now.strftime("%Y-%m-%d %H:%M:%S")}
- Therapeutic Context: {context}
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

    def _generate_response_with_thinking(self, messages: List[Dict[str, str]]) -> str:
        """Generate response using SEA-Lion API and return only the final commentary"""
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
                "max_tokens": 2500,  # Slightly higher for therapeutic responses
                "temperature": 0.4,  # Slightly higher for empathetic responses but still consistent
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
            
            # Extract answer block only, ignore thinking
            answer_match = re.search(r"```answer\s*(.*?)\s*```", raw_content, re.DOTALL)
            commentary = answer_match.group(1).strip() if answer_match else raw_content.strip()
            
            self.logger.info(f"Generated therapeutic response - Commentary: {len(commentary)} chars")
            return commentary
            
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Error generating response from SEA-Lion API: {str(e)}")
            return "ขออภัยค่ะ เกิดปัญหาในการเชื่อมต่อ กรุณาลองใหม่อีกครั้งค่ะ"
        except Exception as e:
            self.logger.error(f"Error generating response: {str(e)}")
            return "ขออภัยค่ะ เกิดข้อผิดพลาดในระบบ"

    def analyze(self, user_message: str, chat_history: List[Dict], chain_of_thoughts: str = "", summarized_histories: str = "") -> str:
        """
        Main analyze method expected by orchestrator
        
        Args:
            user_message: Current user input
            chat_history: Previous conversation history
            chain_of_thoughts: Past analysis chain of thoughts
            summarized_histories: Summarized conversation histories
            
        Returns:
            Single string response for therapeutic guidance
        """
        # Build comprehensive context for psychological analysis
        context_parts = []
        
        if summarized_histories:
            context_parts.append(f"Patient History Summary: {summarized_histories}")
        
        if chain_of_thoughts:
            context_parts.append(f"Previous Therapeutic Considerations: {chain_of_thoughts}")
        
        # Extract recent relevant context from chat history
        recent_context = []
        for msg in chat_history[-4:]:  # Last 4 messages for therapeutic context
            if msg.get("role") == "user":
                recent_context.append(f"Client: {msg.get('content', '')}")
            elif msg.get("role") == "assistant":
                recent_context.append(f"Previous Response: {msg.get('content', '')}")
        
        if recent_context:
            context_parts.append("Recent Therapeutic Conversation:\n" + "\n".join(recent_context))
        
        full_context = "\n\n".join(context_parts) if context_parts else ""
        
        # Create comprehensive psychological analysis prompt
        prompt = f"""
Based on the current psychological query and therapeutic context, provide comprehensive mental health support:

**Current Query:** {user_message}

**Available Context:**
{full_context if full_context else "No previous therapeutic context available"}

Please provide evidence-based psychological support:

1. **Psychological Assessment:**
   - Emotional state analysis and mood indicators
   - Cognitive patterns and thought distortions identification
   - Behavioral patterns and coping mechanisms assessment
   - Risk factors and protective factors evaluation

2. **Therapeutic Interventions:**
   - CBT techniques for cognitive restructuring
   - Behavioral activation strategies
   - Mindfulness and grounding techniques
   - Motivational interviewing approaches
   - Solution-focused interventions

3. **Crisis Evaluation:**
   - Suicide risk assessment if concerning signs present
   - Self-harm indicators or dangerous behaviors
   - Psychotic symptoms or severe dissociation
   - Substance abuse or emergency situations

4. **Treatment Recommendations:**
   - Immediate coping strategies and self-care
   - Between-session homework and practice exercises
   - Professional referral recommendations
   - Safety planning if crisis indicators present

5. **Therapeutic Relationship:**
   - Validation of client's experience and emotions
   - Empathetic responses and active listening
   - Building therapeutic rapport and trust
   - Encouraging client autonomy and self-efficacy

**Response Structure:**

```answer
[Empathetic, supportive therapeutic response with evidence-based interventions, validation, coping strategies, and appropriate professional recommendations. Maintain warm, non-judgmental tone while providing practical guidance.]
```

Always include crisis assessment when appropriate and emphasize professional mental health care for ongoing support.
"""

        messages = self._format_messages(prompt, full_context)
        return self._generate_response_with_thinking(messages)

    # Keep existing methods for backward compatibility
    def provide_therapy_session(self, presenting_concerns: str, mental_health_history: Optional[str] = None, language: str = "english") -> str:
        """Legacy method - returns single response for backward compatibility"""
        fake_history = []
        if mental_health_history:
            fake_history.append({"role": "system", "content": f"Mental Health History: {mental_health_history}"})
        
        return self.analyze(presenting_concerns, fake_history, "", mental_health_history or "")

    def crisis_intervention(self, crisis_situation: str, safety_concerns: str, language: str = "english") -> str:
        """Legacy method - returns single response for backward compatibility"""
        context = f"Crisis situation: {crisis_situation}, Safety concerns: {safety_concerns}"
        return self.analyze(f"CRISIS: {crisis_situation}", [], "", context)

    def cognitive_therapy(self, negative_thoughts: str, emotional_patterns: str, language: str = "english") -> str:
        """Legacy method - returns single response for backward compatibility"""
        context = f"Negative thoughts: {negative_thoughts}, Emotional patterns: {emotional_patterns}"
        return self.analyze("I need help with cognitive restructuring for my negative thoughts", [], "", context)

    def anxiety_management(self, anxiety_symptoms: str, triggers: str, language: str = "english") -> str:
        """Legacy method - returns single response for backward compatibility"""
        context = f"Anxiety symptoms: {anxiety_symptoms}, Triggers: {triggers}"
        return self.analyze("I need help managing my anxiety symptoms", [], "", context)

    def depression_support(self, depressive_symptoms: str, motivation_levels: str, language: str = "english") -> str:
        """Legacy method - returns single response for backward compatibility"""
        context = f"Depressive symptoms: {depressive_symptoms}, Motivation levels: {motivation_levels}"
        return self.analyze("I need help with depression and low motivation", [], "", context)

    def assess_crisis_level(self, mental_state: str, safety_indicators: str) -> Dict[str, Any]:
        """Assess mental health crisis level and provide structured recommendations"""
        response = self.analyze(f"Crisis assessment needed - Mental state: {mental_state}", [], "", f"Safety indicators: {safety_indicators}")
        
        # Parse response to extract crisis level
        crisis_level = "moderate"  # default
        response_lower = response.lower()
        
        if "emergency" in response_lower:
            crisis_level = "emergency"
        elif "severe" in response_lower:
            crisis_level = "severe"
        elif "mild" in response_lower:
            crisis_level = "mild"
        elif "none" in response_lower or "no crisis" in response_lower:
            crisis_level = "none"
        
        return {
            "crisis_level": crisis_level,
            "full_assessment": response,
            "timestamp": datetime.now().isoformat(),
            "requires_immediate_intervention": crisis_level in ["severe", "emergency"]
        }


if __name__ == "__main__":
    # Test the modified Psychologist Agent
    try:
        psychologist = PsychologistAgent(log_level=logging.DEBUG)
        
        # Test the new analyze method
        print("=== TEST: ANALYZE METHOD ===")
        result = psychologist.analyze(
            user_message="I've been feeling very anxious about work and having trouble sleeping. I keep worrying about making mistakes.",
            chat_history=[
                {"role": "user", "content": "I've been stressed lately"},
                {"role": "assistant", "content": "I understand you're experiencing stress. Can you tell me more about what's been going on?"},
                {"role": "user", "content": "Work has been overwhelming"},
                {"role": "assistant", "content": "Work stress can be very challenging. Let's explore some coping strategies together."}
            ],
            chain_of_thoughts="Previous session identified work-related anxiety and perfectionist tendencies. Client shows insight and willingness to engage in therapeutic work.",
            summarized_histories="Client has history of anxiety during high-stress periods, responds well to CBT techniques, strong support system"
        )
        
        print("SINGLE RESPONSE:")
        print(result)
        
        # Test crisis assessment
        print("\n=== TEST: CRISIS ASSESSMENT ===")
        assessment = psychologist.assess_crisis_level(
            mental_state="Feeling hopeless and having thoughts of not wanting to be here anymore",
            safety_indicators="Has support from family, no specific plan, reaching out for help"
        )
        print(f"Crisis Level: {assessment['crisis_level']}")
        print(f"Requires Immediate Intervention: {assessment['requires_immediate_intervention']}")
        
        # Test legacy method still works
        print("\n=== TEST: LEGACY METHOD ===")
        legacy_result = psychologist.provide_therapy_session(
            presenting_concerns="I feel overwhelmed and anxious about everything",
            mental_health_history="Previous episodes of anxiety, no formal treatment"
        )
        print(legacy_result)
        
    except Exception as e:
        print(f"Error testing Psychologist Agent: {e}")