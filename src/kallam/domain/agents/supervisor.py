import os
from dotenv import load_dotenv
load_dotenv()
import logging
import requests
import re
import json
from pathlib import Path
from datetime import datetime

from typing import Literal, Optional, Dict, Any, List
from strands import Agent, tool


class SupervisorAgent:
    def __init__(self, log_level: int = logging.INFO):

        self._setup_logging(log_level)
        self._setup_api_clients()
        
        self.logger.info(f"KaLLaM chatbot initialized successfully using ")

        self.system_prompt = """
**Your Role:** 
You are "KaLLaM" or "กะหล่ำ" with a nickname "Kabby" You are a warm, friendly, female, doctor, psychiatrist, chatbot specializing in analyzing and improving patient's physical and mental health. 
Your goal is to provide actionable guidance that motivates patients to take better care of themselves.

**Core Rules:**
- You are the supervisor agent that handle multiple agents
- You **ALWAYS** need to respond with the language the user used (English or Thai)
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
        """Setup API clients; do not hard-fail if env is missing."""
        # SEA-Lion API (for main chat)
        self.sea_lion_api_key = os.getenv("SEA_LION_API_KEY")
        self.sea_lion_base_url = os.getenv("SEA_LION_BASE_URL", "https://api.sea-lion.ai/v1")
        self.api_enabled = bool(self.sea_lion_api_key)
        if self.api_enabled:
            self.logger.info("SEA-Lion API client initialized")
        else:
            # Keep running; downstream logic will use safe fallbacks
            self.logger.warning(
                "SEA_LION_API_KEY not set. Supervisor will use safe fallback responses."
            )

    def _format_chat_history_for_sea_lion(
            self, 
            chat_histories: List[Dict[str, str]], 
            user_message: str, 
            memory_context: str,
            task: str,
            summarized_histories: Optional[List] = None,
            commentary: Optional[Dict[str, str]] = None,
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

**Previous Activated Agents Commentaries:**
{commentary}

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
- If the user reports physical symptoms, illnesses, treatments, or medications → set "doctor": true
- If the user suggest any emotional struggles, thoughts, relationships, or psychological concerns → set "psychologist": true  
- According to the previous commentaries the "psychologist" should be false only when the conversation clearly don't need psychologist anymore
- "language" MUST be exactly "english" or "thai" (lowercase)
- Both "doctor" and "psychologist" can be true if both aspects are present
- Do not include ANY text before or after the JSON object
- Do not use markdown code blocks or backticks
- Do not add explanations, commentary, or additional text
- Return ONLY the raw JSON object
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
- You are a professional medical advisor.
- Read the given context and response throughly.
- Response concisely and short according to most recommendation from the commentary of each agents (may or maynot given).
- In suicidal or very severe case → reccommend advise immediate professional help at the end of your response.
- Always answer in the same language the user used.
- When reflecting, avoid repeating exact client words. Add depth: infer feelings, values, or reframe the perspective.
- Keep your response very concise unless the user need more context and response.
- Your response should include problem probing since the context is never enough.
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
    
    def _extract_and_validate_json(self, raw_content: str) -> str:
        """Enhanced JSON extraction and validation with fallback"""
        if not raw_content:
            self.logger.warning("Empty response received, returning default JSON")
            return '{"language": "english", "doctor": false, "psychologist": false}'
        
        # Remove thinking blocks first (if any)
        content = raw_content
        thinking_match = re.search(r"</think>", content, re.DOTALL)
        if thinking_match:
            content = re.sub(r".*?</think>\s*", "", content, flags=re.DOTALL).strip()
        
        # Remove markdown code blocks
        content = re.sub(r'^```(?:json|JSON)?\s*', '', content, flags=re.MULTILINE)
        content = re.sub(r'```\s*$', '', content, flags=re.MULTILINE)
        content = content.strip()
        
        # Extract JSON object using multiple strategies
        json_candidates = []
        
        # Strategy 1: Find complete JSON objects
        json_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
        matches = re.findall(json_pattern, content, re.DOTALL)
        json_candidates.extend(matches)
        
        # Strategy 2: Look for specific schema pattern
        schema_pattern = r'\{\s*"language"\s*:\s*"(?:english|thai)"\s*,\s*"doctor"\s*:\s*(?:true|false)\s*,\s*"psychologist"\s*:\s*(?:true|false)\s*\}'
        schema_matches = re.findall(schema_pattern, content, re.IGNORECASE | re.DOTALL)
        json_candidates.extend(schema_matches)
        
        # Strategy 3: If no complete JSON found, try to construct from parts
        if not json_candidates:
            language_match = re.search(r'"language"\s*:\s*"(english|thai)"', content, re.IGNORECASE)
            doctor_match = re.search(r'"doctor"\s*:\s*(true|false)', content, re.IGNORECASE)
            psychologist_match = re.search(r'"psychologist"\s*:\s*(true|false)', content, re.IGNORECASE)
            
            if language_match and doctor_match and psychologist_match:
                constructed_json = f'{{"language": "{language_match.group(1).lower()}", "doctor": {doctor_match.group(1).lower()}, "psychologist": {psychologist_match.group(1).lower()}}}'
                json_candidates.append(constructed_json)
        
        # Validate and return the first working JSON
        for candidate in json_candidates:
            try:
                # Clean up the candidate
                candidate = candidate.strip()
                
                # Parse to validate structure
                parsed = json.loads(candidate)
                
                # Validate schema
                if not isinstance(parsed, dict):
                    continue
                    
                required_keys = {"language", "doctor", "psychologist"}
                if not required_keys.issubset(parsed.keys()):
                    continue
                    
                # Validate language field
                if parsed["language"] not in ["english", "thai"]:
                    continue
                    
                # Validate boolean fields
                if not isinstance(parsed["doctor"], bool) or not isinstance(parsed["psychologist"], bool):
                    continue
                
                # If we reach here, the JSON is valid
                self.logger.debug(f"Successfully validated JSON: {candidate}")
                return candidate
                
            except json.JSONDecodeError as e:
                self.logger.debug(f"JSON parse failed for candidate: {candidate[:100]}... Error: {e}")
                continue
            except Exception as e:
                self.logger.debug(f"Validation failed for candidate: {candidate[:100]}... Error: {e}")
                continue
        
       # If all strategies fail, return a safe default
        self.logger.error("Could not extract valid JSON from response, using safe default")
        return '{"language": "english", "doctor": false, "psychologist": false}'
    
    def _clean_json_response(self, raw_content: str) -> str:
        """Legacy method - now delegates to enhanced extraction"""
        return self._extract_and_validate_json(raw_content)
    
    def _generate_feedback_sea_lion(self, messages: List[Dict[str, str]], show_thinking: bool = False) -> str:
        # If API is disabled, return conservative fallback immediately
        if not getattr(self, "api_enabled", False):
            is_flag_task = any("Return ONLY a single JSON object" in msg.get("content", "")
                               for msg in messages if msg.get("role") == "system")
            if is_flag_task:
                return '{"language": "english", "doctor": false, "psychologist": false}'
            return "ขออภัยค่ะ ระบบมีปัญหาชั่วคราว กรุณาลองใหม่อีกครั้งค่ะ"

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
                "temperature": 0.4,
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
            
            is_flag_task = any("Return ONLY a single JSON object" in msg.get("content", "") 
                        for msg in messages if msg.get("role") == "system")
            
            if is_flag_task:
                # Apply enhanced JSON extraction and validation for flag tasks
                final_answer = self._extract_and_validate_json(raw_content)
                self.logger.debug("Applied enhanced JSON extraction for flag task")
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
            self.logger.info(f"Received response from SEA-Lion API (raw length: {len(raw_content)} chars, final answer length: {len(final_answer)} chars)")
            
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
            if task == "flag":
                # For flag tasks, return a valid JSON structure
                return '{"language": "english", "doctor": false, "psychologist": false}'
            else:
                return "ขออภัยค่ะ ระบบมีปัญหาชั่วคราว กรุณาลองใหม่อีกครั้งค่ะ"


if __name__ == "__main__":
    # Enhanced demo with JSON validation testing
    try:
        sup = SupervisorAgent(log_level=logging.DEBUG)
    except Exception as e:
        print(f"[BOOT ERROR] Unable to start SupervisorAgent: {e}")
        raise SystemExit(1)

    # Test cases for JSON validation
    test_cases = [
        {
            "name": "Medical + Psychological",
            "message": "I have a headache and feel anxious about my exams.",
            "expected": {"doctor": True, "psychologist": True}
        },
        {
            "name": "Thai Medical Only",
            "message": "ปวดหัวมากครับ แล้วก็มีไข้ด้วย",
            "expected": {"doctor": True, "psychologist": False, "language": "thai"}
        },
        {
            "name": "Psychological Only",
            "message": "I'm feeling very stressed and worried about my future.",
            "expected": {"doctor": False, "psychologist": True}
        }
    ]

    chat_history = [
        {"role": "user", "content": "Hi, I've been feeling tired lately."},
        {"role": "assistant", "content": "Thanks for sharing. How's your sleep and stress?"}
    ]
    memory_context = "User: 21 y/o student, midterm week, low sleep (4–5h), high caffeine, history of migraines."

    print("=== TESTING ENHANCED JSON SUPERVISOR ===\n")
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"Test {i}: {test_case['name']}")
        print(f"Message: {test_case['message']}")
        
        flag_output = sup.generate_feedback(
            chat_history=chat_history,
            user_message=test_case['message'],
            memory_context=memory_context,
            task="flag"
        )
        
        print(f"JSON Output: {flag_output}")
        
        # Validate the JSON
        try:
            parsed = json.loads(flag_output)
            print(f"Valid JSON structure")
            print(f"Language: {parsed.get('language')}")
            print(f"Doctor: {parsed.get('doctor')}")
            print(f"Psychologist: {parsed.get('psychologist')}")
        except json.JSONDecodeError as e:
            print(f"Invalid JSON: {e}")
        
        print("-" * 50)

    # Test finalize task
    print("\n=== TEST: FINALIZED RESPONSE ===")
    commentary = {
        "doctor": "Likely tension-type headache aggravated by stress and poor sleep. Suggest hydration, rest, OTC analgesic if not contraindicated.",
        "psychologist": "Teach 4-7-8 breathing, short cognitive reframing for exam anxiety, and a 20-minute study-break cycle."
    }

    final_output = sup.generate_feedback(
        chat_history=chat_history,
        user_message="I have a headache and feel anxious about my exams.",
        memory_context=memory_context,
        task="finalize",
        commentary=commentary
    )
    print(final_output)
