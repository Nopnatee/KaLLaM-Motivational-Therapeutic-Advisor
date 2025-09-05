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
# Psychologist Agent
# --------------------------
class PsychologistAgent:

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
            temperature=0.4,  # Slightly higher for empathetic responses but still consistent
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

        system_prompt = """\
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

        self.agent = Agent(
            model=bedrock_model,
            system_prompt=system_prompt,
            tools=[],
            callback_handler=None,
        )

    # --------------------------
    # Public methods
    # --------------------------
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
        """
        
        response = self.agent(prompt)
        return response

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
        """
        
        response = self.agent(prompt)
        return response

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
        """
        
        response = self.agent(prompt)
        return response

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
        """
        
        response = self.agent(prompt)
        return response

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
        """
        
        response = self.agent(prompt)
        return response