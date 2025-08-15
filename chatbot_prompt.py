import os
import json
import logging
import requests
from google import genai
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, List

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

class KaLLaMChatbot:
    """
    KaLLaM - Thai AI Doctor Chatbot for health guidance and patient care
    """

    def __init__(self, api_provider: Optional[str] = None, max_messages: Optional[int] = 10, log_level: int = logging.INFO):
        """
        Initialize KaLLaM chatbot
        
        Args:
            api_provider: Which API to use - "sea_lion" or "gemini" (default: "sea_lion")
            max_messages: Maximum number of messages to keep in history
            log_level: Logging level (default: INFO)
        """
        if api_provider not in ["sea_lion", "gemini"]:
            raise ValueError("api_provider must be either 'sea_lion' or 'gemini'")
            
        self.api_provider = api_provider
        self.max_messages = max_messages
        self._setup_logging(log_level)
        self._setup_api_clients()
        self._setup_base_config()
        
        self.logger.info(f"KaLLaM chatbot initialized successfully using {self.api_provider} API")

    def _setup_logging(self, log_level: int) -> None:
        """Setup logging configuration"""
        # Create logs directory if it doesn't exist
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        
        # Setup logger
        self.logger = logging.getLogger(f"{__name__}.KaLLaMChatbot")
        self.logger.setLevel(log_level)
        
        # Remove existing handlers to avoid duplicates
        if self.logger.handlers:
            self.logger.handlers.clear()
        
        # File handler for detailed logs
        file_handler = logging.FileHandler(
            log_dir / f"kallam_{datetime.now().strftime('%Y%m%d')}.log",
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

    def _setup_api_clients(self) -> None:
        """Setup the selected API client based on api_provider"""
        try:
            if self.api_provider == "sea_lion":
                # Setup SEA-Lion API only
                self.sea_lion_api_key = os.getenv("SEA_LION_API_KEY")
                self.sea_lion_base_url = os.getenv("SEA_LION_BASE_URL", "https://api.sea-lion.ai/v1")
                
                if not self.sea_lion_api_key:
                    raise ValueError("SEA_LION_API_KEY not provided and not found in environment variables")
                    
                self.logger.info("SEA-Lion API client initialized")
                
            elif self.api_provider == "gemini":
                # Setup Gemini API only
                self.gemini_api_key = os.getenv("GEMINI_API_KEY")
                
                if not self.gemini_api_key:
                    raise ValueError("GEMINI_API_KEY not provided and not found in environment variables")
                    
                self.gemini_client = genai.Client(api_key=self.gemini_api_key)
                self.gemini_model_name = "gemini-2.5-flash-preview-05-20"
                self.logger.info(f"Gemini API client initialized with model: {self.gemini_model_name}")
                
        except Exception as e:
            self.logger.error(f"Failed to initialize {self.api_provider} API client: {str(e)}")
            raise

    def _setup_base_config(self) -> None:
        """Setup base configuration for KaLLaM"""
        self.base_config = {
            "character": """
**Your Name:** "KaLLaM" or "กะหล่ำ"
**Your Role:** You are a Thai, warm, friendly, female, doctor, psychiatrist, chatbot specializing in analyzing and improving patient's physical and mental health. Your goal is to provide actionable guidance that motivates patients to take better care of themselves.
""",
            "core_rules": """
**Core Rules:**
- only greet on first interaction (with the user saying greetings or the "current_user_message" first message)
- Provide specific, actionable health improvement feedback
- Focus on patient motivation for self-care
- Keep responses concise, practical, and culturally appropriate
- Adjust response length based on message complexity – keep replies short for simple questions
- Use easy-to-understand Thai language exclusively
- Use "ค่ะ"/"คะ" appropriately (not consecutively) for warmth
- Address users as "คนไข้", "คุณ", or by name (never "ลูกค้า"/"ผู้ใช้")
- Consider practical context from chat history
- Avoid repetitive content from previous responses
- End conversation if user is satisfied and session is concluded
- Skip unnecessary introductions/conclusions - focus on concise feedback
- Focus on using routine health assessment questions (sleep, eating, exercise, mood, stress, social)
- Apply probing techniques to understand root causes of the patient's problems
- Apply active listening techniques: reflect user's feelings, summarize key points, and acknowledge their struggles
- For low motivation patients: Use persuasion techniques and don't back down easily
- If the patient does not give enough information, ask for more details one by one in small sentences through probing questions
- When responding to a greeting, only greet on the first interaction
- Regularly ask for patient's thoughts, feelings, and opinions about their condition and treatment plans
- If patient shows unwillingness to change, use medical facts about symptom progression and statistics to illustrate serious consequences
- If patient gives a definitive answer, do not ask for opinion again on the same topic.
""",
            "suggestions": """
**Interaction Guidelines:**
- For low context, ask about patient's routine such as eating, exercises,
- Use emoticons for engagement and warmth
- Use respectful tone for elderly patients
- In solution use actionable suggestion according to the patient's health condition
- Balance questioning with supportive statements
- Apply threat appraisal and social proof when facing resistance
- Most of the patients have emotional and mental health issues - try to understand the patient and remain calm when they are angry
- **Periodically ask for patient's opinion** after explaining conditions or suggesting treatments
- **Use medical facts strategically** when encountering resistance to change
- **Incorporate rest guidance** into treatment plans with specific recommendations
- **Explain symptom progression** to motivate early intervention
- Use active listening: echo back user's concerns in your own words before giving advice
"""
        }
        
        self.logger.debug("Base configuration loaded successfully")

    def _format_chat_history_for_sea_lion(self, chat_history: List[Dict[str, str]], user_message: str, health_status: str, summarized_histories: Optional[List] = None) -> List[Dict[str, str]]:
        """
        Format chat history properly for SEA-Lion API
        
        Args:
            chat_history: List of message dictionaries
            user_message: Current user message
            health_status: User's health status
            summarized_histories: Optional summarized history
            
        Returns:
            Properly formatted messages for SEA-Lion API
        """
        # Create system message with context
        now = datetime.now()
        
        # Build context information
        context_info = f"""
**Current Context:**
- Date/Time: {now.strftime("%Y-%m-%d %H:%M:%S")}
- User Health Status: {health_status}
"""
        
        if summarized_histories:
            summarized_text = "\n".join([f"{m.get('role', 'unknown')}: {m.get('content', '')}" for m in summarized_histories])
            context_info += f"- Previous Session Summary: {summarized_text}\n"
        
        system_message = {
            "role": "system",
            "content": f"{self.base_config['character']}\n{context_info}\n{self.base_config['core_rules']}\n{self.base_config['suggestions']}"
        }
        
        # Format messages
        messages = [system_message]
        
        # Add truncated chat history
        truncated_history = self._truncate_history(chat_history, max_messages=self.max_messages)
        
        for msg in truncated_history:
            # Ensure proper role mapping
            role = msg.get('role', 'user')
            if role not in ['user', 'assistant', 'system']:
                role = 'user' if role != 'assistant' else 'assistant'
            
            messages.append({
                "role": role,
                "content": msg.get('content', '')
            })
        
        # Add current user message
        messages.append({
            "role": "user", 
            "content": user_message
        })
        
        return messages

    def _generate_feedback_sea_lion(self, messages: List[Dict[str, str]]) -> str:
        """
        Generate feedback using SEA-Lion API with proper message formatting
        
        Args:
            messages: Properly formatted messages for the API
            
        Returns:
            Generated response text
        """
        try:
            self.logger.debug(f"Sending {len(messages)} messages to SEA-Lion API")
            
            headers = {
                "Authorization": f"Bearer {self.sea_lion_api_key}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "model": "aisingapore/Llama-SEA-LION-v3.5-8B-R",
                "messages": messages,
                "max_tokens": 1000,  # Reduced for more focused responses
                "temperature": 0.7,
                "top_p": 0.9,
                "frequency_penalty": 0.1,  # Help prevent repetition
                "presence_penalty": 0.1    # Encourage new topics
            }
            
            response = requests.post(
                f"{self.sea_lion_base_url}/chat/completions",
                headers=headers,
                json=payload,
                timeout=30
            )
            
            response.raise_for_status()
            response_data = response.json()
            
            # Check if response has expected structure
            if "choices" not in response_data or len(response_data["choices"]) == 0:
                self.logger.error(f"Unexpected response structure: {response_data}")
                return "ขออภัยค่ะ ไม่สามารถประมวลผลคำตอบได้ในขณะนี้"
            
            choice = response_data["choices"][0]
            if "message" not in choice or "content" not in choice["message"]:
                self.logger.error(f"Unexpected message structure: {choice}")
                return "ขออภัยค่ะ ไม่สามารถประมวลผลคำตอบได้ในขณะนี้"
                
            response_text = choice["message"]["content"]
            
            # Check if response is None or empty
            if response_text is None:
                self.logger.error("SEA-Lion API returned None content")
                return "ขออภัยค่ะ ไม่สามารถสร้างคำตอบได้ในขณะนี้"
            
            if isinstance(response_text, str) and response_text.strip() == "":
                self.logger.error("SEA-Lion API returned empty content")
                return "ขออภัยค่ะ ไม่สามารถสร้างคำตอบได้ในขณะนี้"
            
            self.logger.info(f"Received response from SEA-Lion API (length: {len(response_text)} chars)")
            self.logger.debug(f"SEA-Lion Response: {response_text[:200]}..." if len(response_text) > 200 else f"SEA-Lion Response: {response_text}")
            
            return str(response_text).strip()
            
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Error generating feedback from SEA-Lion API: {str(e)}")
            return "ขออภัยค่ะ เกิดปัญหาในการเชื่อมต่อ กรุณาลองใหม่อีกครั้งค่ะ"
        except KeyError as e:
            self.logger.error(f"Unexpected response format from SEA-Lion API: {str(e)}")
            return "ขออภัยค่ะ รูปแบบข้อมูลไม่ถูกต้อง"
        except Exception as e:
            self.logger.error(f"Error generating feedback from SEA-Lion API: {str(e)}")
            return "ขออภัยค่ะ เกิดข้อผิดพลาดในระบบ"

    def _generate_feedback_gemini(self, prompt: str) -> str:
        """
        Generate feedback using Gemini API (fallback)
        
        Args:
            prompt: The prompt to send to the API
            
        Returns:
            Generated response text
        """
        try:
            self.logger.debug(f"Sending prompt to Gemini API (length: {len(prompt)} chars)")
            
            response = self.gemini_client.models.generate_content(
                model=self.gemini_model_name,
                contents=[prompt],
            )
            
            response_text = response.text
            
            # Check if response is None or empty
            if response_text is None:
                self.logger.error("Gemini API returned None content")
                return "ขออภัยค่ะ ไม่สามารถสร้างคำตอบได้ในขณะนี้"
            
            if isinstance(response_text, str) and response_text.strip() == "":
                self.logger.error("Gemini API returned empty content")
                return "ขออภัยค่ะ ไม่สามารถสร้างคำตอบได้ในขณะนี้"
            
            self.logger.info(f"Received response from Gemini API (length: {len(response_text)} chars)")
            self.logger.debug(f"Gemini Response: {response_text[:200]}..." if len(response_text) > 200 else f"Gemini Response: {response_text}")
            
            return str(response_text).strip()
            
        except Exception as e:
            self.logger.error(f"Error generating feedback from Gemini API: {str(e)}")
            return "ขออภัยค่ะ เกิดข้อผิดพลาดในระบบ กรุณาลองใหม่อีกครั้งค่ะ"

    def _generate_feedback(self, chat_history: List[Dict[str, str]], user_message: str, health_status: str, summarized_histories: Optional[List] = None) -> str:
        """
        Generate feedback using the selected API provider
        
        Args:
            chat_history: Previous conversation history
            user_message: Current user message
            health_status: User's health status
            summarized_histories: Optional summarized history
            
        Returns:
            Generated response text
        """
        try:
            if self.api_provider == "sea_lion":
                messages = self._format_chat_history_for_sea_lion(chat_history, user_message, health_status, summarized_histories)
                response = self._generate_feedback_sea_lion(messages)
                if response is None:
                    raise Exception("SEA-Lion API returned None response")
                return response
            elif self.api_provider == "gemini":
                # Keep the original prompt-based approach for Gemini
                prompt = self._build_prompt_for_gemini(chat_history, user_message, health_status, summarized_histories)
                response = self._generate_feedback_gemini(prompt)
                if response is None:
                    raise Exception("Gemini API returned None response")
                return response
            else:
                raise ValueError(f"Unknown API provider: {self.api_provider}")
        except Exception as e:
            self.logger.error(f"Error in _generate_feedback: {str(e)}")
            # Return a fallback response instead of None
            return "ขออภัยค่ะ ระบบมีปัญหาชั่วคราว กรุณาลองใหม่อีกครั้งค่ะ"

    def _build_prompt_for_gemini(self, chat_history: List[Dict[str, str]], user_message: str, health_status: str, summarized_histories: Optional[List] = None) -> str:
        """
        Build prompt for Gemini API (keeping original approach)
        """
        # Convert history dicts into readable text
        truncated_history = self._truncate_history(chat_history, max_messages=5)
        history_text = "\n".join([f"{m['role']}: {m['content']}" for m in truncated_history])
        
        # Convert summarized history if available
        summarized_text = ""
        if summarized_histories:
            summarized_text = "\n".join([f"{m['role']}: {m['content']}" for m in summarized_histories])
        
        now = datetime.now()
        
        inputs = f"""- Chat history (truncated): {history_text}
- current_user_message: {user_message}
- user_health_status: {health_status}
- summarized_history: {summarized_text if summarized_text else "N/A"}"""
        
        specific_content = """
**Your Task:** Analyze the chat history and current message to provide health guidance. Only greet on first interaction.

**Response Strategy:**
- If this is early in conversation, ask 1-2 routine assessment questions
- If problems are mentioned, use probing techniques to understand deeper causes
- Use active listening: reflect back patient's emotions or statements before giving advice
- Always include encouragement and validation
- Balance questioning with supportive statements
- For low motivation: Use persuasion techniques, don't back down easily
- Keep your responses concise given Thai do not like to read
- If user input is short or simple, give a brief response (1-2 short sentences maximum)
- SYMPTOM PROGRESSION: If non co-operation or ignorance is detected explain how current symptoms may worsen if left untreated, using clear timeline and consequences
- Include specific guidance: with duration, timing, and types relevant to patient's condition
- Connect symptoms to future consequences: using clear timelines and medical evidence
- **If patient resists change**: Present medical facts, statistics, and consequences to motivate action
"""
        
        base_prompt = f"""
{self.base_config['character']}

**Current Context:**
- Date/Time: {now.strftime("%Y-%m-%d %H:%M:%S")}
{inputs}

{self.base_config['core_rules']}

{specific_content}

{self.base_config['suggestions']}
"""
        
        return base_prompt

    def _truncate_history(self, chat_history: List[Dict[str, str]], max_messages: int = 20) -> List[Dict[str, str]]:
        """
        Keep only the last few messages from chat history to reduce prompt size.
        
        Args:
            chat_history: List of message dictionaries
            max_messages: Number of recent messages to keep
            
        Returns:
            Truncated chat history
        """
        if len(chat_history) <= max_messages:
            return chat_history
        else:
            truncated = chat_history[-max_messages:]
            self.logger.debug(f"Truncated chat history from {len(chat_history)} to {max_messages} messages")
            return truncated

    def get_chatbot_response(
        self, 
        chat_history: List[Dict[str, str]], 
        user_message: str, 
        health_status: str, 
        summarized_histories: Optional[List] = None
    ) -> str:
        """
        Get main chatbot response from KaLLaM
        
        Args:
            chat_history: Previous conversation history (list of dicts with 'role' and 'content')
            user_message: Current user message
            health_status: User's health status information
            summarized_histories: Optional summarized conversation history
            
        Returns:
            KaLLaM's response
        """
        self.logger.info("Processing chatbot response request")
        self.logger.debug(f"User message: {user_message}")
        self.logger.debug(f"Health status: {health_status}")
        self.logger.debug(f"Chat history length: {len(chat_history)}")
        
        try:
            response = self._generate_feedback(chat_history, user_message, health_status, summarized_histories)
            self.logger.info("Successfully generated chatbot response")
            return response
            
        except Exception as e:
            self.logger.error(f"Error generating chatbot response: {str(e)}")
            raise

    def get_followup_response(
        self, 
        chat_history: List[Dict[str, str]], 
        summarized_histories: Optional[List] = None
    ) -> str:
        """
        Get follow-up response to check patient progress
        
        Args:
            chat_history: Previous conversation history
            summarized_histories: Optional summarized conversation history
            
        Returns:
            KaLLaM's follow-up response
        """
        self.logger.info("Processing follow-up response request")
        
        try:
            # Create a follow-up prompt as a user message
            followup_message = "ตอนนี้เป็นยังไงบ้างคะ? มีความคืบหน้าในการดูแลสุขภาพหรือไม่?"
            
            # Use empty health status for follow-up
            response = self._generate_feedback(chat_history, followup_message, "Follow-up check", summarized_histories)
            self.logger.info("Successfully generated follow-up response")
            return response
            
        except Exception as e:
            self.logger.error(f"Error generating follow-up response: {str(e)}")
            raise

    def summarize_history(self, chat_history: List[Dict[str, str]]) -> str:
        """
        Summarize chat history into concise paragraph
        
        Args:
            chat_history: Full conversation history to summarize
            
        Returns:
            Summarized conversation history
        """
        self.logger.info("Processing history summarization request")
        self.logger.debug(f"Chat history length: {len(chat_history)} messages")
        
        try:
            # Convert chat history to text for summarization
            history_text = "\n".join([f"{msg.get('role', 'unknown')}: {msg.get('content', '')}" for msg in chat_history])
            
            summary_prompt = f"""
Summarize the given chat history into a short paragraph including key events.

**Input:** {history_text}

**Requirements:**
- Keep summary concise with key events and important details
- Include time/date references (group close dates/times together)
- Use timeline format if history is very long
- Respond in Thai language only
- Return "None" if insufficient information
- **Include patient's resistance patterns** and what facts/consequences were effective
- **Note rest and sleep patterns** mentioned by patient
- **Track symptom progression** concerns discussed

**Response Format:** [Summarized content]
"""
            
            # Use summarization as a special case
            if self.api_provider == "sea_lion":
                messages = [{
                    "role": "system",
                    "content": "You are a medical assistant that summarizes patient conversations concisely in Thai."
                }, {
                    "role": "user", 
                    "content": summary_prompt
                }]
                result = self._generate_feedback_sea_lion(messages)
            else:
                result = self._generate_feedback_gemini(summary_prompt)
            
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
            "api_provider": self.api_provider,
            "sea_lion_configured": self.api_provider == "sea_lion" and hasattr(self, 'sea_lion_api_key'),
            "gemini_configured": self.api_provider == "gemini" and hasattr(self, 'gemini_api_key'),
            "active_model": getattr(self, 'gemini_model_name', None) if self.api_provider == "gemini" else "aisingapore/Llama-SEA-LION-v3.5-8B-R",
            "timestamp": datetime.now().isoformat(),
            "logging_enabled": True,
            "log_level": self.logger.level
        }
        
        self.logger.debug(f"Health status check: {status}")
        return status