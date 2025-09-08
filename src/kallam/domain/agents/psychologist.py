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
            
            # Extract thinking and answer blocks
            thinking_match = re.search(r"```thinking\s*(.*?)\s*```", raw_content, re.DOTALL)
            answer_match = re.search(r"```answer\s*(.*?)\s*```", raw_content, re.DOTALL)
            
            reasoning = thinking_match.group(1).strip() if thinking_match else None
            final_answer = answer_match.group(1).strip() if answer_match else raw_content.strip()
            
            if reasoning:
                self.logger.debug(f"Psychologist reasoning:\n{reasoning}")
            
            self.logger.info(f"Generated therapeutic response (length: {len(final_answer)} chars)")
            return final_answer
            
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Error generating response from SEA-Lion API: {str(e)}")
            return "ขออภัยค่ะ เกิดปัญหาในการเชื่อมต่อ กรุณาลองใหม่อีกครั้งค่ะ"
        except Exception as e:
            self.logger.error(f"Error generating response: {str(e)}")
            return "ขออภัยค่ะ เกิดข้อผิดพลาดในระบบ"

    # Public methods
    def provide_therapy_session(self, presenting_concerns: str, mental_health_history: Optional[str] = None, language: str = "english") -> str:
        """Main therapeutic session method for addressing mental health concerns"""
        
        context = f"Presenting concerns: {presenting_concerns}"
        if mental_health_history:
            context += f"\nMental health history: {mental_health_history}"
        
        prompt = f"""
Please provide therapeutic support for the following case:

{context}

Please apply evidence-based therapeutic approaches:

1. **Active Listening & Validation:**
   - Reflect back the patient's emotions and experiences
   - Validate their struggles without judgment
   - Use empathetic responding to build therapeutic rapport

2. **CBT Assessment:**
   - Identify any cognitive distortions (catastrophizing, all-or-nothing thinking, mind reading, etc.)
   - Help patient recognize unhelpful thought patterns
   - Use Socratic questioning to challenge negative beliefs
   - Suggest cognitive restructuring techniques

3. **Motivational Interviewing:**
   - Explore patient's ambivalence about change
   - Use OARS technique (Open questions, Affirmations, Reflections, Summaries)
   - Elicit patient's own motivations for change
   - Support patient autonomy and self-efficacy

4. **Solution-Focused Approach:**
   - Identify patient's existing strengths and coping resources
   - Explore times when the problem was less severe (exceptions)
   - Use scaling questions (1-10) to assess current state
   - Set small, achievable therapeutic goals

5. **Mindfulness & Coping Skills:**
   - Teach grounding techniques (5-4-3-2-1 method: 5 things you see, 4 you touch, etc.)
   - Guide through breathing exercises or progressive muscle relaxation
   - Suggest mindfulness practices for emotional regulation
   - Provide stress management strategies

6. **Crisis Assessment:**
   - Screen for suicidal ideation, self-harm, or safety concerns
   - If crisis indicators present, recommend immediate professional intervention
   - Develop safety planning strategies

Respond in {language} language and provide:
- Therapeutic reflection and validation
- Psychological assessment and insights
- Evidence-based intervention strategies
- Homework assignments or between-session practices
- Professional referral guidance if needed
- Crisis safety recommendations if applicable

Always include appropriate mental health disclaimers and emphasize the importance of professional care for ongoing support.

**Structure your response as:**
**THERAPEUTIC REFLECTION:** [Validation and empathy]
**ASSESSMENT:** [Key psychological patterns identified]
**INTERVENTIONS:** [Specific therapeutic strategies]
**HOMEWORK/PRACTICE:** [Between-session activities]
**PROFESSIONAL GUIDANCE:** [When to seek additional help]
**SUPPORT:** [Encouragement and hope]
"""
        
        messages = self._format_messages(prompt, context)
        return self._generate_response(messages)

    def crisis_intervention(self, crisis_situation: str, safety_concerns: str, language: str = "english") -> str:
        """Handle mental health crisis situations with immediate intervention"""
        
        prompt = f"""
MENTAL HEALTH CRISIS INTERVENTION:

Crisis situation: {crisis_situation}
Safety concerns: {safety_concerns}

Please provide immediate crisis intervention support:

1. **Immediate Safety Assessment:**
   - Evaluate suicide risk factors and protective factors
   - Assess for self-harm intentions or behaviors
   - Screen for psychotic symptoms or severe dissociation
   - Check for substance use or medical emergencies

2. **Crisis De-escalation:**
   - Use calm, supportive, non-judgmental communication
   - Validate the person's emotional pain while instilling hope
   - Help ground the person in the present moment
   - Use active listening to understand their immediate needs

3. **Safety Planning:**
   - Help identify immediate coping strategies
   - Develop a step-by-step safety plan
   - Identify support people who can be contacted
   - Remove or limit access to means of self-harm
   - Create a crisis card with emergency contacts

4. **Immediate Resources:**
   - Strongly recommend immediate professional intervention
   - Provide crisis hotline numbers and emergency services information
   - Suggest accompanying person to emergency services if needed
   - Recommend psychiatric emergency evaluation

5. **Containment Strategies:**
   - Teach grounding techniques for overwhelming emotions
   - Guide through box breathing (4-4-4-4 count)
   - Use progressive muscle relaxation for physical tension
   - Provide emotional regulation techniques

6. **Follow-up Planning:**
   - Schedule immediate follow-up with mental health professional
   - Arrange for 24-hour supervision if needed
   - Connect with community mental health resources
   - Ensure medication compliance if applicable

Respond in {language} language with:
- Immediate crisis intervention strategies
- Safety planning recommendations
- Professional emergency referral guidance
- Specific crisis hotline numbers and resources
- Step-by-step action plan for the next 24-48 hours

This is a mental health emergency - prioritize immediate safety while providing compassionate support.

**CRITICAL CRISIS RESOURCES:**
- National Suicide Prevention Lifeline: 988 (US)
- Crisis Text Line: Text HOME to 741741
- Emergency Services: Call 911 (US) or local emergency number
- Local psychiatric emergency services

**Structure your response as:**
**IMMEDIATE SAFETY:** [Emergency assessment and actions]
**CRISIS SUPPORT:** [De-escalation and validation]
**SAFETY PLAN:** [Specific steps to take]
**RESOURCES:** [Emergency contacts and services]
**FOLLOW-UP:** [Next 24-48 hour plan]
"""
        
        context = f"Crisis intervention: {crisis_situation} | Safety concerns: {safety_concerns}"
        messages = self._format_messages(prompt, context)
        return self._generate_response(messages)

    def cognitive_therapy(self, negative_thoughts: str, emotional_patterns: str, language: str = "english") -> str:
        """Provide CBT-focused intervention for negative thought patterns"""
        
        prompt = f"""
COGNITIVE BEHAVIORAL THERAPY SESSION:

Negative thoughts: {negative_thoughts}
Emotional patterns: {emotional_patterns}

Please provide CBT-focused therapeutic intervention:

1. **Cognitive Assessment:**
   - Identify specific cognitive distortions present:
     * All-or-nothing thinking (black and white thinking)
     * Catastrophizing (jumping to worst-case scenarios)
     * Mind reading (assuming you know what others think)
     * Fortune telling (predicting negative outcomes)
     * Emotional reasoning (feelings = facts)
     * Should statements (rigid expectations)
     * Labeling (calling yourself names)
     * Mental filtering (focusing only on negatives)
     * Discounting positives (minimizing good things)
     * Personalization (taking blame for things outside your control)

2. **Thought Record Analysis:**
   - Help identify the connection between thoughts, feelings, and behaviors
   - Examine evidence for and against negative thoughts
   - Look for patterns in when these thoughts occur
   - Assess the impact these thoughts have on daily functioning

3. **Cognitive Restructuring:**
   - Use Socratic questioning to challenge negative beliefs
   - Help develop balanced, realistic alternative thoughts
   - Practice thought-stopping techniques
   - Create coping statements for difficult situations

4. **Behavioral Experiments:**
   - Suggest activities to test negative predictions
   - Design behavioral activation strategies
   - Plan pleasant activity scheduling
   - Create exposure exercises for avoidance patterns

5. **Homework Assignments:**
   - Daily thought records to track patterns
   - Behavioral experiments to test beliefs
   - Gratitude journaling or positive event logging
   - Mindfulness exercises to observe thoughts without judgment

6. **Relapse Prevention:**
   - Identify early warning signs of negative thinking
   - Develop a toolkit of coping strategies
   - Create an action plan for setbacks
   - Build ongoing self-monitoring skills

Respond in {language} language with:
- Specific cognitive distortions identified
- Evidence-based thought challenging techniques
- Behavioral intervention strategies
- Specific homework assignments
- Progress monitoring recommendations
- Relapse prevention planning

Provide clear, actionable CBT techniques while maintaining therapeutic rapport and validation.

**Structure your response as:**
**COGNITIVE PATTERNS:** [Distortions identified]
**THOUGHT CHALLENGING:** [Restructuring techniques]
**BEHAVIORAL STRATEGIES:** [Activity and exposure plans]
**HOMEWORK:** [Specific between-session tasks]
**MONITORING:** [Progress tracking methods]
**PREVENTION:** [Relapse prevention strategies]
"""
        
        context = f"CBT session - Thoughts: {negative_thoughts} | Emotions: {emotional_patterns}"
        messages = self._format_messages(prompt, context)
        return self._generate_response(messages)

    def anxiety_management(self, anxiety_symptoms: str, triggers: str, language: str = "english") -> str:
        """Specialized intervention for anxiety disorders and panic management"""
        
        prompt = f"""
ANXIETY MANAGEMENT THERAPY SESSION:

Anxiety symptoms: {anxiety_symptoms}
Identified triggers: {triggers}

Please provide specialized anxiety intervention:

1. **Anxiety Psychoeducation:**
   - Explain the fight-flight-freeze response
   - Normalize anxiety as a natural survival mechanism
   - Describe how anxiety maintains itself through avoidance
   - Explain the anxiety cycle and how to break it

2. **Immediate Anxiety Management:**
   - Box breathing technique (4-4-4-4 count)
   - 5-4-3-2-1 grounding technique (senses)
   - Progressive muscle relaxation (PMR)
   - Visualization and safe place imagery
   - Cold water or ice cube technique for panic attacks

3. **Cognitive Techniques for Anxiety:**
   - Challenge catastrophic thinking patterns
   - Reality testing anxious predictions
   - Develop coping self-talk statements
   - Practice uncertainty tolerance
   - Use the "So what?" technique for worry thoughts

4. **Exposure Therapy Principles:**
   - Create anxiety hierarchy (least to most feared situations)
   - Plan gradual exposure exercises
   - Practice staying in anxiety-provoking situations
   - Learn that anxiety peaks and naturally decreases
   - Build confidence through successful exposures

5. **Lifestyle Modifications:**
   - Sleep hygiene recommendations
   - Caffeine and alcohol impact on anxiety
   - Regular exercise for anxiety reduction
   - Nutrition's role in mood stability
   - Stress management and time management skills

6. **Long-term Anxiety Management:**
   - Develop daily mindfulness practice
   - Build distress tolerance skills
   - Create anxiety action plan
   - Identify support network
   - Plan for setback management

Respond in {language} language with:
- Immediate anxiety relief techniques
- Cognitive restructuring for anxious thoughts
- Gradual exposure planning
- Lifestyle modification recommendations
- Long-term anxiety management strategies
- Emergency coping plan for panic attacks

Provide compassionate, evidence-based anxiety treatment while validating the patient's experience.

**Structure your response as:**
**UNDERSTANDING ANXIETY:** [Psychoeducation]
**IMMEDIATE RELIEF:** [Crisis techniques]
**THOUGHT WORK:** [Cognitive strategies]
**EXPOSURE PLAN:** [Gradual facing of fears]
**LIFESTYLE:** [Daily management strategies]
**LONG-TERM PLAN:** [Ongoing anxiety management]
"""
        
        context = f"Anxiety management - Symptoms: {anxiety_symptoms} | Triggers: {triggers}"
        messages = self._format_messages(prompt, context)
        return self._generate_response(messages)

    def depression_support(self, depressive_symptoms: str, motivation_levels: str, language: str = "english") -> str:
        """Specialized intervention for depression and mood disorders"""
        
        prompt = f"""
DEPRESSION SUPPORT THERAPY SESSION:

Depressive symptoms: {depressive_symptoms}
Current motivation levels: {motivation_levels}

Please provide specialized depression intervention:

1. **Depression Assessment:**
   - Evaluate severity of depressive symptoms
   - Assess impact on daily functioning
   - Screen for suicidal ideation (ask directly if concerning)
   - Identify triggers and contributing factors
   - Assess social support and isolation levels

2. **Behavioral Activation:**
   - Schedule pleasant activities daily
   - Break large tasks into smaller, manageable steps
   - Create activity monitoring charts
   - Focus on mastery and pleasure activities
   - Establish daily routine and structure

3. **Cognitive Techniques for Depression:**
   - Challenge all-or-nothing thinking
   - Address self-criticism and negative self-talk
   - Examine evidence for hopeless thoughts
   - Practice self-compassion exercises
   - Develop balanced perspective on situations

4. **Motivational Enhancement:**
   - Explore values and what matters most
   - Set small, achievable daily goals
   - Celebrate small wins and progress
   - Use motivational interviewing techniques
   - Address ambivalence about change

5. **Social Connection:**
   - Identify supportive relationships
   - Plan social activities despite low mood
   - Address social withdrawal patterns
   - Practice communication skills
   - Build new social connections gradually

6. **Self-Care and Routine:**
   - Establish consistent sleep schedule
   - Plan regular meals and nutrition
   - Incorporate gentle physical activity
   - Create morning and evening routines
   - Practice good personal hygiene habits

7. **Relapse Prevention:**
   - Identify early warning signs of depression
   - Create depression action plan
   - Build coping skills toolkit
   - Plan ongoing self-monitoring
   - Establish professional support network

Respond in {language} language with:
- Behavioral activation strategies
- Cognitive restructuring for depressive thoughts
- Motivation enhancement techniques
- Social connection recommendations
- Self-care and routine planning
- Relapse prevention strategies
- Professional referral guidance if needed

Provide hope-instilling, evidence-based depression treatment while acknowledging the patient's struggle.

**Structure your response as:**
**VALIDATION:** [Understanding the depression experience]
**BEHAVIORAL ACTIVATION:** [Activity and routine strategies]
**THOUGHT WORK:** [Cognitive restructuring]
**MOTIVATION:** [Building energy and purpose]
**CONNECTION:** [Social support strategies]
**SELF-CARE:** [Daily management routines]
**PREVENTION:** [Long-term wellness planning]
"""
        
        context = f"Depression support - Symptoms: {depressive_symptoms} | Motivation: {motivation_levels}"
        messages = self._format_messages(prompt, context)
        return self._generate_response(messages)

    def assess_crisis_level(self, mental_state: str, safety_indicators: str) -> Dict[str, Any]:
        """Assess mental health crisis level and provide structured recommendations"""
        
        prompt = f"""
Perform a mental health crisis assessment:

Mental state: {mental_state}
Safety indicators: {safety_indicators}

Please provide a structured crisis assessment in the following format:

**CRISIS LEVEL ASSESSMENT:**
Crisis Level: [none/mild/moderate/severe/emergency]

**RISK FACTORS:**
[List identified risk factors]

**PROTECTIVE FACTORS:**
[List identified protective factors]

**IMMEDIATE RECOMMENDATIONS:**
[Specific actions to take immediately]

**PROFESSIONAL INTERVENTION:**
[Level of professional care needed]

**SAFETY PLANNING:**
[Immediate safety measures]

**FOLLOW-UP:**
[Timeline and follow-up recommendations]

Base your assessment on:
- **Emergency**: Immediate danger to self or others, active suicidal plan
- **Severe**: High suicide risk, severe symptoms, impaired functioning
- **Moderate**: Concerning symptoms, some risk factors, needs intervention
- **Mild**: Low-level symptoms, good coping, stable support
- **None**: No significant crisis indicators present
"""
        
        context = f"Crisis assessment - Mental state: {mental_state} | Safety: {safety_indicators}"
        messages = self._format_messages(prompt, context)
        response = self._generate_response(messages)
        
        # Parse response to extract crisis level
        crisis_level = "moderate"  # default
        response_lower = response.lower()
        
        if "emergency" in response_lower:
            crisis_level = "emergency"
        elif "severe" in response_lower:
            crisis_level = "severe"
        elif "mild" in response_lower:
            crisis_level = "mild"
        elif "none" in response_lower:
            crisis_level = "none"
        
        return {
            "crisis_level": crisis_level,
            "full_assessment": response,
            "timestamp": datetime.now().isoformat(),
            "requires_immediate_intervention": crisis_level in ["severe", "emergency"]
        }


if __name__ == "__main__":
    # Test the Psychologist Agent
    try:
        psychologist = PsychologistAgent(log_level=logging.DEBUG)
        
        # Test therapy session
        print("=== TEST: THERAPY SESSION ===")
        session = psychologist.provide_therapy_session(
            presenting_concerns="I've been feeling very anxious about work and having trouble sleeping. I keep worrying about making mistakes.",
            mental_health_history="Previous episodes of anxiety during stressful periods, no formal treatment",
            language="english"
        )
        print(session)
        
        # Test crisis assessment
        print("\n=== TEST: CRISIS ASSESSMENT ===")
        assessment = psychologist.assess_crisis_level(
            mental_state="Feeling hopeless and having thoughts of not wanting to be here anymore",
            safety_indicators="Has support from family, no specific plan, reaching out for help"
        )
        print(f"Crisis Level: {assessment['crisis_level']}")
        print(f"Requires Immediate Intervention: {assessment['requires_immediate_intervention']}")
        
    except Exception as e:
        print(f"Error testing Psychologist Agent: {e}")