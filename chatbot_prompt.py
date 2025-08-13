import os
import json
import logging
from google import genai
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any


class SangJaiChatbot:
    """
    SangJai - Thai AI Doctor Chatbot for health guidance and patient care
    """
    
    def __init__(self, api_key: Optional[str] = None, log_level: int = logging.INFO):
        """
        Initialize SangJai chatbot
        
        Args:
            api_key: Gemini API key (if None, will try to get from environment)
            log_level: Logging level (default: INFO)
        """
        self._setup_logging(log_level)
        self._setup_api_client(api_key)
        self._setup_base_config()
        
        self.logger.info("SangJai chatbot initialized successfully")
    
    def _setup_logging(self, log_level: int) -> None:
        """Setup logging configuration"""
        # Create logs directory if it doesn't exist
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        
        # Setup logger
        self.logger = logging.getLogger(f"{__name__}.SangJaiChatbot")
        self.logger.setLevel(log_level)
        
        # Remove existing handlers to avoid duplicates
        if self.logger.handlers:
            self.logger.handlers.clear()
        
        # File handler for detailed logs
        file_handler = logging.FileHandler(
            log_dir / f"sangjai_{datetime.now().strftime('%Y%m%d')}.log",
            encoding='utf-8'
        )
        file_handler.setLevel(logging.DEBUG)
        
        # Console handler for immediate feedback
        console_handler = logging.StreamHandler()
        console_handler.setLevel(log_level)
        
        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        # Add handlers
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
    
    def _setup_api_client(self, api_key: Optional[str]) -> None:
        """Setup Gemini API client"""
        try:
            # Get API key from parameter or environment
            self.api_key = api_key or os.getenv("GEMINI_API_KEY")
            
            if not self.api_key:
                raise ValueError("GEMINI_API_KEY not provided and not found in environment variables")
            
            self.client = genai.Client(api_key=self.api_key)
            self.model_name = "gemini-2.5-flash-preview-05-20"
            
            self.logger.info(f"Gemini API client initialized with model: {self.model_name}")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Gemini API client: {str(e)}")
            raise
    
    def _setup_base_config(self) -> None:
        """Setup base configuration for SangJai"""
        self.base_config = {
            "character": """
**Your Name:** "SangJai" or "แสงใจ"

**Your Role:**
You are a Thai, warm, friendly, female, doctor, psychiatrist, chatbot specializing in analyzing and improving patient's physical and mental health. Your goal is to provide actionable guidance that motivates patients to take better care of themselves.
""",
            
            "core_rules": """
**Core Rules:**
1. Provide specific, actionable health improvement feedback
2. Focus on patient motivation for self-care
3. Keep responses concise, practical, and culturally appropriate
4. Adjust response length based on message complexity – keep replies short for simple questions
5. Use easy-to-understand Thai language exclusively
6. Use "ค่ะ"/"คะ" appropriately (not consecutively) for warmth
7. Address users as "คนไข้", "คุณ", or by name (never "ลูกค้า"/"ผู้ใช้")
8. Consider practical context from chat history
9. Avoid repetitive content from previous responses
10. End conversation if user is satisfied and session is concluded
11. Skip unnecessary introductions/conclusions - focus on concise feedback
12. Focus on using routine health assessment questions (sleep, eating, exercise, mood, stress, social)
13. Apply probing techniques to understand root causes of the patient's problems
14. Apply active listening techniques: reflect user's feelings, summarize key points, and acknowledge their struggles
15. Use gamification elements (points, levels, challenges, progress tracking)
16. For low motivation patients: Use persuasion techniques and don't back down easily
17. For fatty liver conditions: Recommend weight loss, exercise, diet modification
18. If the patient does not give enough information, ask for more details one by one in small sentences through probing questions
19. When responding to a greeting, only greet on the first interaction
20. Regularly ask for patient's thoughts, feelings, and opinions about their condition and treatment plans
21. If patient shows unwillingness to change, use medical facts about symptom progression and statistics to illustrate serious consequences
22. If patient gives a definitive answer, do not ask for opinion again on the same topic.
""",
     
            "suggestions": """
**Interaction Guidelines:**
- For low context, ask about patient's routine such as eating, exercises,
- Use emoticons for engagement and warmth
- Use respectful tone for elderly patients
- In solution use actionable suggestion according to the patient's health condition
- Balance questioning with supportive statements
- Apply threat appraisal and social proof when facing resistance
- Implement gamification to maintain long-term engagement
- Most of the patients have fatty liver disease
- Consider Thai's dietary habits (like eating lots of rice)
- **Periodically ask for patient's opinion** after explaining conditions or suggesting treatments
- **Use medical facts strategically** when encountering resistance to change
- **Incorporate rest guidance** into treatment plans with specific recommendations
- **Explain symptom progression** to motivate early intervention
- Use active listening: echo back user's concerns in your own words before giving advice
"""
        }
        
        self.logger.debug("Base configuration loaded successfully")
    
    def _generate_feedback(self, prompt: str) -> str:
        """
        Generate feedback using Gemini API
        
        Args:
            prompt: The prompt to send to the API
            
        Returns:
            Generated response text
        """
        try:
            self.logger.debug(f"Sending prompt to Gemini API (length: {len(prompt)} chars)")
            
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=[prompt],
            )
            
            response_text = response.text
            self.logger.info(f"Received response from Gemini API (length: {len(response_text)} chars)")
            self.logger.debug(f"API Response: {response_text[:200]}..." if len(response_text) > 200 else f"API Response: {response_text}")
            
            return response_text
            
        except Exception as e:
            self.logger.error(f"Error generating feedback from Gemini API: {str(e)}")
            raise
    
    def _build_prompt(self, specific_content: str, inputs: str) -> str:
        """
        Build a complete prompt with shared base config and specific content
        
        Args:
            specific_content: Task-specific content and instructions
            inputs: Context inputs (chat history, user message, etc.)
            
        Returns:
            Complete formatted prompt
        """
        now = datetime.now()
        
        base_prompt = f"""
{self.base_config['character']}

**Current Context:**
- Date/Time: {now.strftime("%Y-%m-%d %H:%M:%S")}
{inputs}

{self.base_config['core_rules']}

{specific_content}

{self.base_config['suggestions']}
"""
        
        self.logger.debug(f"Built prompt with {len(base_prompt)} characters")
        return base_prompt
    
    def get_chatbot_response(
        self, 
        chat_history: str, 
        user_message: str, 
        health_status: str, 
        summarized_histories: Optional[str] = None
    ) -> str:
        """
        Get main chatbot response from SangJai
        
        Args:
            chat_history: Previous conversation history
            user_message: Current user message
            health_status: User's health status information
            summarized_histories: Optional summarized conversation history
            
        Returns:
            SangJai's response
        """
        self.logger.info("Processing chatbot response request")
        self.logger.debug(f"User message: {user_message}")
        self.logger.debug(f"Health status: {health_status}")
        
        try:
            inputs = f"""- Chat history: {chat_history}
- Current user message: {user_message}
- User's health status: {health_status}
- Summarized history: {summarized_histories}"""
            
            specific_content = """
**Your Task:**
Analyze the chat history and current message to provide health guidance. Only greet on first interaction.

**Response Strategy:**
1. If this is early in conversation, ask 1-2 routine assessment questions
2. If problems are mentioned, use probing techniques to understand deeper causes
3. Use active listening: reflect back patient's emotions or statements before giving advice
4. Always include encouragement and validation
5. Provide specific, actionable health advice including detailed rest recommendations
6. Balance questioning with supportive statements
7. For low motivation: Use persuasion techniques, don't back down easily
8. **If patient resists change**: Present medical facts, statistics, and consequences to motivate action
9. Implement gamification elements (points, challenges, progress tracking)
10. For fatty liver conditions: Provide specific treatment guidelines with progression warnings
11. Keep your responses concise given Thai do not like to read
12. For food questions, gently probe asking details one by one until the patient provides enough information
13. Suggest substitutes for unhealthy food choices and dietary modifications
14. **Include specific rest guidance** with duration, timing, and types relevant to patient's condition
15. **Connect symptoms to future consequences** using clear timelines and medical evidence
16. If user input is short or simple, give a brief response (1-2 short sentences maximum)
17. SYMPTOM PROGRESSION: If non co-operation or ignorance is detected explain how current symptoms may worsen if left untreated, using clear timeline and consequences
"""
            
            prompt = self._build_prompt(specific_content, inputs)
            response = self._generate_feedback(prompt)
            
            self.logger.info("Successfully generated chatbot response")
            return response
            
        except Exception as e:
            self.logger.error(f"Error generating chatbot response: {str(e)}")
            raise
    
    def get_followup_response(
        self, 
        chat_history: str, 
        summarized_histories: Optional[str] = None
    ) -> str:
        """
        Get follow-up response to check patient progress
        
        Args:
            chat_history: Previous conversation history
            summarized_histories: Optional summarized conversation history
            
        Returns:
            SangJai's follow-up response
        """
        self.logger.info("Processing follow-up response request")
        
        try:
            inputs = f"""- Chat history: {chat_history} (for context)
- Summarized history: {summarized_histories}"""
            
            specific_content = """
**Your Task:**
Ask about patient's progress based on chat history. Do not answer questions from the input.

**Follow-up Strategy:**
1. Reference specific previous concerns or goals mentioned
2. Use routine assessment questions relevant to their situation
3. Include encouraging words about their journey
4. Ask about any new developments or challenges
5. **Check on rest implementation** - ask specifically about sleep quality and rest practices
6. **Ask for patient's opinion** on their progress and how they feel about changes
7. **Remind about symptom progression** if patient seems to be slipping back into old habits

**Response Format:**
[Follow-up Notification]
"""
            
            prompt = self._build_prompt(specific_content, inputs)
            response = self._generate_feedback(prompt)
            
            self.logger.info("Successfully generated follow-up response")
            return response
            
        except Exception as e:
            self.logger.error(f"Error generating follow-up response: {str(e)}")
            raise
    
    def summarize_history(self, chat_history: str) -> str:
        """
        Summarize chat history into concise paragraph
        
        Args:
            chat_history: Full conversation history to summarize
            
        Returns:
            Summarized conversation history
        """
        self.logger.info("Processing history summarization request")
        self.logger.debug(f"Chat history length: {len(chat_history)} characters")
        
        try:
            prompt = f"""
Summarize the given chat history into a short paragraph including key events.

**Input:** {chat_history}

**Requirements:**
- Keep summary concise with key events and important details
- Include time/date references (group close dates/times together)
- Use timeline format if history is very long
- Respond in Thai language only
- Return "None" if insufficient information
- **Include patient's resistance patterns** and what facts/consequences were effective
- **Note rest and sleep patterns** mentioned by patient
- **Track symptom progression** concerns discussed

**Response Format:**
[Summarized content]
"""
            
            result = self._generate_feedback(prompt)
            
            self.logger.info("Successfully generated history summary")
            self.logger.debug(f"Summary result: {result}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error summarizing history: {str(e)}")
            raise
    
    def get_health_status(self) -> Dict[str, Any]:
        """
        Get current chatbot health status and statistics
        
        Returns:
            Dictionary containing chatbot status information
        """
        status = {
            "status": "healthy",
            "model": self.model_name,
            "api_key_configured": bool(self.api_key),
            "timestamp": datetime.now().isoformat(),
            "logging_enabled": True,
            "log_level": self.logger.level
        }
        
        self.logger.debug(f"Health status check: {status}")
        return status


# Example usage and testing
if __name__ == "__main__":
    try:
        # Initialize chatbot
        chatbot = SangJaiChatbot()
        
        # Test basic functionality
        print("Testing SangJai chatbot...")
        
        # Example conversation
        response = chatbot.get_chatbot_response(
            chat_history="",
            user_message="สวัสดีค่ะ ฉันมีปัญหาเรื่องนอนไม่หลับ",
            health_status="มีปัญหานอนไม่หลับ, ความดันโลหิตสูง",
            summarized_histories=None
        )
        
        print(f"SangJai Response: {response}")
        
        # Check health status
        status = chatbot.get_health_status()
        print(f"Chatbot Status: {json.dumps(status, indent=2, ensure_ascii=False)}")
        
    except Exception as e:
        print(f"Error testing chatbot: {e}")