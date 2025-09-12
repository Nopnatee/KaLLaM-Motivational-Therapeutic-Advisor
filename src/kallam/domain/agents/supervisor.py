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


class SupervisorAgent:
    def __init__(self, log_level: int = logging.INFO):

        self._setup_logging(log_level)
        self._setup_agents()
        
        self.logger.info(f"KaLLaM chatbot initialized successfully using Strands Agent with Amazon Bedrock")

        self.system_prompt = """
**Your Role:** 
You are KaLLaM" or "กัลหล่ำ" You are a warm, friendly, female, doctor, psychiatrist, chatbot specializing in analyzing and improving patient's physical and mental health. 
Your goal is to provide actionable guidance that motivates patients to take better care of themselves.

**Core Rules:**
- You are the supervisor agent that decides which specialized agent should handle each user message
- If a message includes both medical and psychological elements, choose the agent that addresses the most urgent or dominant concern (e.g., chest pain + anxiety → Doctor first).
- If unclear, ask a clarifying question before assigning.
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

    def _setup_agents(self) -> None:
        """Setup Strands agents - using Amazon Bedrock with AWS credentials"""
        try:
            # Check for AWS credentials (boto3 will handle the credential resolution)
            aws_access_key = os.getenv("AWS_ACCESS_KEY_ID")
            aws_secret_key = os.getenv("AWS_SECRET_ACCESS_KEY")
            aws_region = os.getenv("AWS_REGION", "sea-2")  # Default to sea-2
            
            if not aws_access_key or not aws_secret_key:
                raise ValueError("AWS credentials not found. Please set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY environment variables")
            
            # Initialize Strands agent with Amazon Bedrock (default provider)
            # The agent will use boto3's credential resolution system
            self.agent = Agent()  # Uses default Amazon Bedrock with Claude 4 Sonnet
                
            self.logger.info(f"Strands Agent with Amazon Bedrock initialized successfully (region: {aws_region})")
                
        except Exception as e:
            self.logger.error(f"Failed to initialize Strands agent with Amazon Bedrock: {str(e)}")
            raise

    def _format_chat_history_for_strands(
            self, 
            chat_histories: List[Dict[str, str]], 
            user_message: str, 
            memory_context: str,
            task: str,
            summarized_histories: Optional[List] = None,
            commentary: Optional[Dict[str, str]] = None
            ) -> str:
        
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
            system_content = f"""
{self.system_prompt}

{context_info}

**Specific Task (strict):**
Return ONLY a single JSON object and nothing else. No intro, no markdown, no code fences.
- If the user reports physical symptoms, illnesses, treatments, or medications → activate **DoctorAgent**.  
- If the user reports emotional struggles, thoughts, relationships, or psychological concerns → activate **PsychologistAgent**.  
- Always respond according to the **Output Schema:**.

**JSON Schema:**
{{
  "language": "english" | "thai",
  "doctor": true | false,
  "psychologist": true | false
}}

**Rules:**
- "language" MUST be exactly "english" or "thai" in lowercase.
- If both medical and psychological aspects are present, you may activate both.  
- For "doctor" and "psychologist" MUST be booleans.
- Output nothing except the JSON object.
"""
        elif task == "finalize":
            commentaries = f"{commentary}" if commentary else ""
            context_info += f"- Commentary from each agents: {commentaries}\n"
            system_content = f"""
{self.system_prompt}

{context_info}

**Specific Task:**
Read the given context and response concisely based on commentary of each agents.
- You are an expert in every medical domain.
- Integrate both medical and psychological perspectives when present.  
- Be clear and supportive, avoiding technical overload.  
- Respect safety protocols: urgent physical symptoms → advise emergency care; suicidal or severe crisis → advise immediate professional help.  
- Always answer in the same language the user used.  

"""
        
        # Build conversation context
        conversation_history = ""
        
        # Add full chat history (no truncation)
        for msg in chat_histories:
            role = msg.get('role', 'user')
            content = msg.get('content', '')
            conversation_history += f"{role}: {content}\n"
        
        # Add current user message
        conversation_history += f"user: {user_message}\n"
        
        # Combine system prompt with conversation
        full_prompt = f"{system_content}\n\n**Conversation History:**\n{conversation_history}\n\nPlease respond:"
        
        return full_prompt
    
    def _clean_json_response(self, raw_content: str) -> str:

        if not raw_content:
            return raw_content
        
        # Remove thinking blocks first (if any)
        thinking_match = re.search(r"</think>", raw_content, re.DOTALL)
        if thinking_match:
            content = re.sub(r".*?</think>\s*", "", raw_content, flags=re.DOTALL).strip()
        else:
            content = raw_content.strip()
        
        content = re.sub(r'^```(?:json|JSON)?\s*', '', content, flags=re.MULTILINE) #remove json markdown
        content = re.sub(r'```\s*$', '', content, flags=re.MULTILINE)
        
        content = content.strip()
        
        json_match = re.search(r'\{.*\}', content, re.DOTALL)
        if json_match:
            content = json_match.group(0)
        
        return content
    
    
    def _generate_feedback_strands(self, prompt: str, show_thinking: bool = False) -> str:
        try:
            self.logger.debug(f"Sending prompt to Strands Agent with Amazon Bedrock (length: {len(prompt)} chars)")
            
            # Use Strands agent to generate response
            response = self.agent.chat(prompt)
            
            if response is None:
                self.logger.error("Strands Agent with Amazon Bedrock returned None response")
                return "ขออภัยค่ะ ไม่สามารถสร้างคำตอบได้ในขณะนี้"
            
            if isinstance(response, str) and response.strip() == "":
                self.logger.error("Strands Agent with Amazon Bedrock returned empty content")
                return "ขออภัยค่ะ ไม่สามารถสร้างคำตอบได้ในขณะนี้"
            
            raw_content = str(response)
            
            # Check if this is a flag task
            is_flag_task = "Return ONLY a single JSON object" in prompt
            
            if is_flag_task:
                # Apply JSON cleaning for flag tasks
                final_answer = self._clean_json_response(raw_content)
                self.logger.debug("Applied JSON cleaning for flag task")
            else:
                # Handle thinking block for non-flag tasks
                thinking_match = re.search(r"</think>", raw_content, re.DOTALL)
                if thinking_match:
                    if show_thinking:
                        # Keep the thinking block visible
                        final_answer = raw_content.strip()
                        self.logger.debug("Thinking block found and kept visible")
                    else:
                        # Remove everything up to and including </think>
                        final_answer = re.sub(r".*?</think>\s*", "", raw_content, flags=re.DOTALL).strip()
                        self.logger.debug("Thinking block found and removed from response")
                else:
                    self.logger.debug("No thinking block found in response")
                    final_answer = raw_content.strip()
            
            # Log response information
            self.logger.info(f"Received response from Strands Agent with Amazon Bedrock (raw length: {len(raw_content)} chars, final answer length: {len(final_answer)} chars)")
            
            return final_answer
            
        except Exception as e:
            self.logger.error(f"Error generating feedback from Strands Agent with Amazon Bedrock: {str(e)}")
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
            prompt = self._format_chat_history_for_strands(chat_history, user_message, memory_context, task, summarized_histories, commentary)
            
            response = self._generate_feedback_strands(prompt)
            if response is None:
                raise Exception("Strands Agent with Amazon Bedrock returned None response")
            return response
        except Exception as e:
            self.logger.error(f"Error in _generate_feedback: {str(e)}")
            # Return a fallback response instead of None
            return "ขออภัยค่ะ ระบบมีปัญหาชั่วคราว กรุณาลองใหม่อีกครั้งค่ะ"


if __name__ == "__main__":
    # Minimal reproducible demo for SupervisorAgent using existing generate_feedback()
    # Requires AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, and optionally AWS_REGION in your environment, otherwise the class will raise.

    # 1) Create the agent
    try:
        sup = SupervisorAgent(log_level=logging.DEBUG)
    except Exception as e:
        print(f"[BOOT ERROR] Unable to start SupervisorAgent: {e}")
        raise SystemExit(1)

    # 2) Dummy chat history (what the user and assistant said earlier)
    chat_history = [
        {"role": "user", "content": "Hi, I've been feeling tired lately."},
        {"role": "assistant", "content": "Thanks for sharing. How's your sleep and stress?"}
    ]

    # 3) Persistent memory or health context you want to feed the model
    memory_context = "User: 21 y/o student, midterm week, low sleep (4—5h), high caffeine, history of migraines."

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