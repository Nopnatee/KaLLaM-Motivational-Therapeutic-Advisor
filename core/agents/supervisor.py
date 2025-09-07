# pip install "strands-agents[openai]" "python-dotenv"
import os
from dotenv import load_dotenv
load_dotenv()
import logging
import requests
import re
from pathlib import Path
from datetime import datetime

from typing import Literal, Optional, Dict, Any, List
from strands import Agent, tool
from strands.models.openai import OpenAIModel  # <-- swap provider


class SupervisorAgent:
    def __init__(self, log_level: int = logging.INFO):

        self._setup_logging(log_level)
        self._setup_api_clients()
        
        self.logger.info(f"KaLLaM chatbot initialized successfully using ")

        self.system_prompt = """
**Your Role:** 
You are KaLLaM" or "กะหล่ำ" You are a warm, friendly, female, doctor, psychiatrist, chatbot specializing in analyzing and improving patient's physical and mental health. 
Your goal is to provide actionable guidance that motivates patients to take better care of themselves.

**Core Rules:**
- 
- 
"""


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
        """Setup API clients - always setup both for mixed usage"""
        try:
            # Setup SEA-Lion API (for main chat)
            self.sea_lion_api_key = os.getenv("SEA_LION_API_KEY")
            self.sea_lion_base_url = os.getenv("SEA_LION_BASE_URL", "https://api.sea-lion.ai/v1")
            
            if not self.sea_lion_api_key:
                raise ValueError("SEA_LION_API_KEY not provided and not found in environment variables")
                
            self.logger.info("SEA-Lion API client initialized")
                
        except Exception as e:
            self.logger.error(f"Failed to initialize API clients: {str(e)}")
            raise

    def _format_chat_history_for_sea_lion(
            self, 
            chat_histories: List[Dict[str, str]], 
            user_message: str, 
            memory_context: str,
            task: str,
            summarized_histories: Optional[List] = None,
            commentary: Optional[Dict[str, str]] = None
            ) -> List[Dict[str, str]]:
        
        # Create system message with context
        now = datetime.now()
        
        # Build context information
        context_info = f"""
**Current Context:**
- Date/Time: {now.strftime("%Y-%m-%d %H:%M:%S")}
- Persistent Memory: {memory_context}
"""
        # Add summarized histories to context_info if available
        if summarized_histories:
            summarized_text = "\n".join([f"{m.get('role', 'unknown')}: {m.get('content', '')}" for m in summarized_histories])
            context_info += f"- Previous Session Summaries: {summarized_text}\n"
        
        if task == "flag":
            system_message = {
                "role": "system",
                "content": f"""
{self.system_prompt}

{context_info}

**Specific Task:**
Read the user's request and decide which specialist agent to activate via flags response in json format.
- You are an expert in routing requests to the right specialists.
- You can activate multiple of them if clearly needed.
- Always respond according to the **Output Schema:**.

**Output Schema:**
{{
  "language": "[detected language in lowercase, e.g. 'thai', 'english']",
  "doctor": "[true/false]",
  "psychologist": "[true/false]"
}}
"""
                             }
        elif task == "finalize":
            commentaries = f"{commentary}" if commentary else ""
            context_info += f"- Commentary from each agents: {commentaries}\n"
            system_message = {
                "role": "system",
                "content": f"""
{self.system_prompt}

{context_info}

**Specific Task:**
Read the given context and response concisely based on commentary of each agents.
- You are an expert in every medical domain.
- if the .
- Always respond with the same language of the user.

"""
                             }
        
        # Format messages
        messages = [system_message]
        
        # Add full chat history (no truncation)
        for msg in chat_histories:
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
                "max_tokens": 2000,  # for thinking and answering
                "temperature": 0.7,
                "top_p": 0.9,
                "frequency_penalty": 0.1,  # prevent repetition
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
                
            raw_content = choice["message"]["content"]
            
            # Check if response is None or empty
            if raw_content is None:
                self.logger.error("SEA-Lion API returned None content")
                return "ขออภัยค่ะ ไม่สามารถสร้างคำตอบได้ในขณะนี้"
            
            if isinstance(raw_content, str) and raw_content.strip() == "":
                self.logger.error("SEA-Lion API returned empty content")
                return "ขออภัยค่ะ ไม่สามารถสร้างคำตอบได้ในขณะนี้"
            
            # Extract reasoning and answer blocks
            thinking_match = re.search(r"```thinking\s*(.*?)\s*```", raw_content, re.DOTALL)
            answer_match = re.search(r"```answer\s*(.*?)\s*```", raw_content, re.DOTALL)
            
            reasoning = thinking_match.group(1).strip() if thinking_match else None
            final_answer = answer_match.group(1).strip() if answer_match else raw_content.strip()
            
            # Log reasoning privately for debugging
            if reasoning:
                self.logger.debug(f"SEA-Lion reasoning:\n{reasoning}")
            else:
                self.logger.debug("No thinking block found in response")
            
            # Log response information
            self.logger.info(f"Received response from SEA-Lion API (raw length: {len(raw_content)} chars, final answer length: {len(final_answer)} chars)")
            self.logger.debug(f"SEA-Lion Final Answer: {final_answer[:200]}..." if len(final_answer) > 200 else f"SEA-Lion Final Answer: {final_answer}")
            
            return final_answer
            
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Error generating feedback from SEA-Lion API: {str(e)}")
            return "ขออภัยค่ะ เกิดปัญหาในการเชื่อมต่อ กรุณาลองใหม่อีกครั้งค่ะ"
        except KeyError as e:
            self.logger.error(f"Unexpected response format from SEA-Lion API: {str(e)}")
            return "ขออภัยค่ะ รูปแบบข้อมูลไม่ถูกต้อง"
        except Exception as e:
            self.logger.error(f"Error generating feedback from SEA-Lion API: {str(e)}")
            return "ขออภัยค่ะ เกิดข้อผิดพลาดในระบบ"
        
    def generate_feedback(
            self, 
            chat_history: List[Dict[str, str]], 
            user_message: str, 
            memory_context: str, 
            task: str, 
            summarized_histories: Optional[List] = None,
            commentary: Optional[Dict[str, str]] = None
            ) -> str:
        
        self.logger.info("Processing chatbot response request")
        self.logger.debug(f"User message: {user_message}")
        self.logger.debug(f"Health status: {memory_context}")
        self.logger.debug(f"Chat history length: {len(chat_history)}")
        
        try:
            messages = self._format_chat_history_for_sea_lion(chat_history, user_message, memory_context, task, summarized_histories, commentary)
            response = self._generate_feedback_sea_lion(messages)
            if response is None:
                raise Exception("SEA-Lion API returned None response")
            return response
        except Exception as e:
            self.logger.error(f"Error in _generate_feedback: {str(e)}")
            # Return a fallback response instead of None
            return "ขออภัยค่ะ ระบบมีปัญหาชั่วคราว กรุณาลองใหม่อีกครั้งค่ะ"

if __name__ == "__main__":
    # Minimal reproducible demo for SupervisorAgent using existing generate_feedback()
    # Requires SEA_LION_API_KEY in your environment, otherwise the class will raise.

    # 1) Create the agent
    try:
        sup = SupervisorAgent(log_level=logging.DEBUG)
    except Exception as e:
        print(f"[BOOT ERROR] Unable to start SupervisorAgent: {e}")
        raise SystemExit(1)

    # 2) Dummy chat history (what the user and assistant said earlier)
    chat_history = [
        {"role": "user", "content": "Hi, I’ve been feeling tired lately."},
        {"role": "assistant", "content": "Thanks for sharing. How’s your sleep and stress?"}
    ]

    # 3) Persistent memory or health context you want to feed the model
    memory_context = "User: 21 y/o student, midterm week, low sleep (4–5h), high caffeine, history of migraines."

    # 4) Optional: previous sessions summarized
    summarized_histories = [
        {"role": "assistant", "content": "Session 1: sleep hygiene tips; suggested hydration and screen breaks."},
        {"role": "assistant", "content": "Session 2: recommended keeping a headache diary; advised limiting caffeine after 2 PM."}
    ]

    # 5) The current user message we want to route/answer
    user_message = "I have a headache and feel anxious about my exams."

    # ===== Test 1: Routing flags (task='flag') =====
    print("\n=== TEST: FLAG DECISION ===")
    flag_output = sup.generate_feedback(
        chat_history=chat_history,
        user_message=user_message,
        memory_context=memory_context,
        task="flag",
        summarized_histories=summarized_histories
    )
    print(flag_output)

    # ===== Test 2: Finalize response (task='finalize') =====
    # Pretend you have per-specialist comments you aggregated elsewhere
    commentary = {
        "doctor": "Likely tension-type headache aggravated by stress and poor sleep. Suggest hydration, rest, OTC analgesic if not contraindicated.",
        "psychologist": "Teach 4-7-8 breathing, short cognitive reframing for exam anxiety, and a 20-minute study-break cycle."
    }

    print("\n=== TEST: FINALIZED RESPONSE ===")
    final_output = sup.generate_feedback(
        chat_history=chat_history,
        user_message=user_message,
        memory_context=memory_context,
        task="finalize",
        summarized_histories=summarized_histories,
        commentary=commentary
    )
    print(final_output)
