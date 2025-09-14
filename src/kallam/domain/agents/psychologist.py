import os
import json
import logging
import requests
import re
from google import genai
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
        self.logger.info(f"PsychologistAgent initialized successfully - Thai->SEA-Lion, English->Gemini")

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
        """Setup both API clients"""
        try:
            # Setup SEA-Lion API
            self.sea_lion_api_key = os.getenv("SEA_LION_API_KEY")
            self.sea_lion_base_url = os.getenv("SEA_LION_BASE_URL", "https://api.sea-lion.ai/v1")
            
            if not self.sea_lion_api_key:
                raise ValueError("SEA_LION_API_KEY not found in environment variables")
                
            self.logger.info("SEA-Lion API client initialized")
            
            # Setup Gemini API
            self.gemini_api_key = os.getenv("GEMINI_API_KEY")
            
            if not self.gemini_api_key:
                raise ValueError("GEMINI_API_KEY not found in environment variables")
                
            self.gemini_client = genai.Client(api_key=self.gemini_api_key)
            self.gemini_model_name = "gemini-2.5-flash-preview-05-20"
            self.logger.info(f"Gemini API client initialized with model: {self.gemini_model_name}")
                
        except Exception as e:
            self.logger.error(f"Failed to initialize API clients: {str(e)}")
            raise

    ##############USE PROMPT ENGINERING LATER#################
    def _detect_language(self, text: str) -> str:
        """
        Detect if the text is primarily Thai or English
        """
        # Count Thai characters (Unicode range for Thai)
        thai_chars = len(re.findall(r'[\u0E00-\u0E7F]', text))
        # Count English characters (basic Latin letters)
        english_chars = len(re.findall(r'[a-zA-Z]', text))
        
        total_chars = thai_chars + english_chars
        
        if total_chars == 0:
            # Default to English if no detectable characters
            return 'english'
        
        thai_ratio = thai_chars / total_chars
        
        # If more than 10% Thai characters, consider it Thai
        if thai_ratio > 0.1:
            detected = 'thai'
        else:
            detected = 'english'
            
        self.logger.debug(f"Language detection: {detected} (Thai: {thai_chars}, English: {english_chars}, Ratio: {thai_ratio:.2f})")
        return detected
    ##############USE PROMPT ENGINERING LATER#################
    
    # ===== SEA-LION BLOCK (THAI) =====
    def _get_sealion_prompt(self) -> str:
        """Thai therapeutic prompt for SEA-Lion"""
        return """
**บทบาทของคุณ:**  
คุณคือนักจิตวิทยาผู้ให้คำปรึกษาด้านการสนับสนุนสุขภาพจิตและการให้คำแนะนำเชิงบำบัดให้แก่ผู้สื่อสารทางการแพทย์
เป้าหมายของคุณคือให้คำแนะนำในการตอบสนองของผู้สื่อสารทางการแพทย์เพื่อเพิ่มประสิทธิภาพในการสนทนาของผู้สื่อสารทางการแพทย์และผู้ใช้งาน

**กฎหลัก:**
- ใช้ภาษาไทยเท่านั้น (ยกเว้นคำทับศัพย์เช่น Motivational Interview)
- คุณให้คำแนะนำสำหรับการสื่อสารแบบ Motivational Interviewing (MI) บุคลาการทางการแพทย์เพื่อใช้ในการวินิจฉัยและรักษาอาการทางจิต
- ในกรณีฉุกเฉิน (ความคิดฆ่าตัวตาย การทำร้ายตนเอง โรคจิต) แนะนำให้แสวงหาความช่วยเหลือจากผู้เชี่ยวชาญฉุกเฉิน  

**คู่มือจิตวิทยา:**  
1.จิตวิญญาณของ MI (MI Spirit)
  - ความร่วมมือ (Collaboration) → บุคลากรทางการแพทย์ทำงานแบบหุ้นส่วน ไม่ใช่สั่งการจากบนลงล่างหรือสั่งสอนอย่างเดียว
  - การกระตุ้น (Evocation) → ดึงเอาเหตุผลและแรงจูงใจจากผู้ป่วยออกมา ไม่ใช่ยัดเยียดคำตอบ
  - การเคารพสิทธิ์การตัดสินใจ (Autonomy) → แสดงออกว่าผู้ป่วยเป็นคนเลือกเอง ไม่กดดันหรือบังคับ
2.ทักษะหลัก OARS
  - คำถามปลายเปิด (Open questions) → ถามให้ขยาย ไม่ใช่แค่ตอบใช่/ไม่ใช่
  - การยืนยัน (Affirmations) → ชื่นชมจุดแข็งหรือความพยายามของผู้ป่วย
  - การสะท้อน (Reflections) → สะท้อนสิ่งที่ผู้ป่วยพูด ทั้งแบบเรียบง่ายหรือซับซ้อน เพื่อแสดงความเข้าใจ
  - การสรุป (Summaries) → ทบทวนเป็นระยะ เพื่อให้บทสนทนาชัดเจนและเสริมแรงการเปลี่ยนแปลง
3.การแยก Change Talk กับ Sustain Talk
  - Change Talk → ผู้ป่วยพูดถึงความต้องการ เหตุผล หรือความตั้งใจที่จะเปลี่ยนแปลง แพทย์ต้องจับประเด็นและเสริมแรง
  - Sustain Talk → เมื่อผู้ป่วยลังเลหรือต่อต้าน แพทย์ไม่เถียง แต่สะท้อนหรือจัดกรอบใหม่อย่างเป็นกลาง
4.สไตล์การสื่อสาร
  - โทนเสียง (Tone) → มีความเข้าใจและเอาใจใส่ มากกว่าตัดสินหรือตรงเกินไป
  - การให้ข้อมูล (Information-giving) → ให้ข้อมูลโดยขออนุญาตก่อน เช่น “คุณอยากฟังคำแนะนำทั่วไปที่มักใช้ไหม”
  - การควบคุมจังหวะ (Pacing) → ฟังมากพอ ไม่รีบสรุปหรือบอกเร็วเกินไป
5.มุมมองเชิงเทคนิค (กรอบการ Coding เช่น MISC/AnnoMI)
  - อัตราส่วนการสะท้อน (Reflections) ต่อคำถาม ควรมีมากกว่า
  - สัดส่วนคำถามปลายเปิดต่อคำถามปิด
  - มีพฤติกรรมที่สอดคล้องกับ MI (affirm, emphasize autonomy) และเลี่ยงพฤติกรรมที่ไม่สอดคล้อง (ตักเตือน, เผชิญหน้าโดยตรง)

**โครงสร้างที่ใช้ในการตอบ:**
"อารมณ์ของผู้ใช้": [อารมณ์ของผู้ใช้จากการวิเคราะห์]
"เทคนิคที่ควรใช้": [เทคนิคอ้างอิงจากคู่มือที่ผู้สื่อสารควรใช้ในการตอบสนองครั้งนี้]
"รายละเอียด": [รายระเอียดของวิธีการใช้เทคนิคหรือข้อมูลเพิ่มเติมที่บุคลากรควรทราบ]
""
"""

    def _format_messages_sealion(self, user_message: str, therapeutic_context: str = "") -> List[Dict[str, str]]:
        """Format messages for SEA-Lion API (Thai)"""
        now = datetime.now()
        
        context_info = f"""
**บริบทปัจจุบัน:**
- วันที่/เวลา: {now.strftime("%Y-%m-%d %H:%M:%S")}
- บริบทการบำบัด: {therapeutic_context}
"""
        
        system_message = {
            "role": "system", 
            "content": f"{self._get_sealion_prompt()}\n\n{context_info}"
        }
        
        user_message_formatted = {
            "role": "user", 
            "content": user_message
        }
        
        return [system_message, user_message_formatted]

    def _generate_response_sealion(self, messages: List[Dict[str, str]]) -> str:
        """Generate response using SEA-Lion API for Thai"""
        try:
            self.logger.debug(f"Sending {len(messages)} messages to SEA-Lion API")
            
            headers = {
                "Authorization": f"Bearer {self.sea_lion_api_key}",
                "Content-Type": "application/json"
            }
            
            # Add thinking mode prompt to last user message
            # if messages and messages[-1]["role"] == "user":
            #     messages[-1]["content"] += (
            #         "\n\nกรุณาแสดงผลในรูปแบบต่อไปนี้:\n"
            #         "```thinking\n{การใคร่ครวญทีละขั้นตอน - วิเคราะห์อาการของผู้ป่วย สภาพทางอารมณ์ และกำหนดแนวทางที่ดีที่สุด}\n```\n"
            #         "```answer\n{การตอบสนองขั้นสุดท้าย - คำแนะนำที่อบอุ่น เข้าใจ และปฏิบัติได้}\n```\n\n"
            #         "**สำคัญ**: คิดผ่านด้านทางการแพทย์และจิตวิทยาเป็นภาษาอังกฤษเพื่อการใคร่ครวญที่ดีขึ้น แล้วให้การตอบสนองผู้ป่วยเป็นภาษาไทย"
            #     )
            
            payload = {
                "model": "aisingapore/Llama-SEA-LION-v3.5-8B-R",
                "messages": messages,
                "chat_template_kwargs": {
                    "thinking_mode": "on"
                },
                "max_tokens": 2000,
                "temperature": 0.7,
                "top_p": 0.9,
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
            
            # Extract answer block
            answer_match = re.search(r"```answer\s*(.*?)\s*```", raw_content, re.DOTALL)
            final_answer = answer_match.group(1).strip() if answer_match else raw_content.strip()
            
            # Log thinking privately
            thinking_match = re.search(r"```thinking\s*(.*?)\s*```", raw_content, re.DOTALL)
            if thinking_match:
                self.logger.debug(f"SEA-Lion thinking:\n{thinking_match.group(1).strip()}")
            
            self.logger.info(f"Received SEA-Lion response (length: {len(final_answer)} chars)")
            return final_answer
            
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Error with SEA-Lion API: {str(e)}")
            return "ขออภัยค่ะ เกิดปัญหาในการเชื่อมต่อ กรุณาลองใหม่อีกครั้งค่ะ"
        except Exception as e:
            self.logger.error(f"Error generating SEA-Lion response: {str(e)}")
            return "ขออภัยค่ะ เกิดข้อผิดพลาดในระบบ"

    # ===== GEMINI BLOCK (ENGLISH) =====
    def _get_gemini_prompt(self) -> str:
        """English therapeutic prompt for Gemini"""
        return """
**Your Role:**  
You are a Professional Psychological Counselor AI specializing in mental health support and therapeutic guidance.  
Your goal is to provide evidence-based psychological interventions while maintaining professional boundaries.

**Core Rules:**  
- You are NOT a replacement for professional mental health care.  
- Always use a calm and reassuring tone
- Maintain warmth, empathy, and professional boundaries at all times.  
- Never provide diagnoses; only supportive guidance and coping strategies.  
- Always recommend consulting a licensed professional for serious concerns.  
- In emergencies (suicidal thoughts, self-harm, psychosis), advise immediate professional intervention.  

**Therapeutic Approaches:**  
1. **Active Listening:** Reflect, paraphrase, validate, and show empathy.  
2. **CBT (Cognitive Behavioral Therapy):** Identify distortions, reframe thoughts, assign behavioral tasks.  
3. **Motivational Interviewing:** Explore ambivalence, elicit change talk, support autonomy.  
4. **Solution-Focused Therapy:** Highlight strengths, scaling questions, set small goals, miracle question.  
5. **Mindfulness & Stress Management:** Grounding, breathing, meditation, relaxation, stress inoculation.  
6. **Crisis Intervention:** Risk assessment, de-escalation, safety planning, connect to resources.  

**Response Guidelines:**  
- Use open-ended questions, empathetic tone, and validation.  
- Provide psychoeducation and coping strategies.  
- Include evidence-based interventions tailored to the client.  
- Always respond in English.  
- Always include crisis safety steps when risk is detected.  

**Crisis Assessment Protocol:**  
- If signs of suicidal ideation, self-harm, psychosis, severe dissociation, substance emergencies, or abuse appear: Immediately recommend emergency professional intervention **and** provide supportive guidance.  

**Output Format:**  
Always structure responses to include:  
- Emotional validation and reflection  
- Psychological assessment (thoughts, emotions, behaviors, risks)  
- Intervention strategies (evidence-based techniques)  
- Professional referral or safety planning if needed  

**Specific Task:**  
Provide supportive, structured, evidence-based therapeutic guidance while ensuring client safety.  
Your primary purpose is to help clients develop coping skills and encourage appropriate professional care.
"""

    def _format_prompt_gemini(self, user_message: str, therapeutic_context: str = "") -> str:
        """Format prompt for Gemini API (English)"""
        now = datetime.now()
        
        context_info = f"""
**Current Context:**
- Date/Time: {now.strftime("%Y-%m-%d %H:%M:%S")}
- Therapeutic Context: {therapeutic_context}
"""
        
        prompt = f"""{self._get_gemini_prompt()}

{context_info}

**Client Message:** {user_message}

Please provide your therapeutic response following the guidelines above."""
        
        return prompt

    def _generate_response_gemini(self, prompt: str) -> str:
        """Generate response using Gemini API for English"""
        try:
            self.logger.debug(f"Sending prompt to Gemini API (length: {len(prompt)} chars)")
            
            response = self.gemini_client.models.generate_content(
                model=self.gemini_model_name,
                contents=[prompt],
            )
            
            response_text = response.text
            
            if response_text is None or (isinstance(response_text, str) and response_text.strip() == ""):
                self.logger.error("Gemini API returned None or empty content")
                return "I apologize, but I'm unable to generate a response at this time. Please try again later."
            
            self.logger.info(f"Received Gemini response (length: {len(response_text)} chars)")
            return str(response_text).strip()
            
        except Exception as e:
            self.logger.error(f"Error generating Gemini response: {str(e)}")
            return "I apologize, but I'm experiencing technical difficulties. Please try again later, and if you're having thoughts of self-harm or are in crisis, please contact a mental health professional or emergency services immediately."

    # ===== ANALYSIS METHODS FOR ORCHESTRATOR =====
    def analyze(self, message: str, history: List[Dict[str, str]], chain_of_thoughts: List[Dict[str, str]], summarized_histories: List[Dict[str, str]]) -> str:
        """
        Analyze method expected by the orchestrator
        Provides psychological analysis and therapeutic guidance
        
        Args:
            message: The client's message to analyze
            history: Chat history as list of message dictionaries
            chain_of_thoughts: Chain of thoughts from previous processing
            summarized_histories: Previously summarized conversation histories
            
        Returns:
            Therapeutic analysis and guidance response
        """
        self.logger.info("Starting psychological analysis")
        self.logger.debug(f"Analyzing message: {message}")
        self.logger.debug(f"History length: {len(history) if history else 0}")
        self.logger.debug(f"Chain of thoughts length: {len(chain_of_thoughts) if chain_of_thoughts else 0}")
        self.logger.debug(f"Summarized histories length: {len(summarized_histories) if summarized_histories else 0}")
        
        try:
            # Build therapeutic context from history and chain of thoughts
            context_parts = []
            
            # Add recent conversation context
            if history:
                recent_messages = history[-3:] if len(history) > 3 else history  # Last 3 messages
                context_parts.append("Recent conversation context:")
                for msg in recent_messages:
                    role = msg.get('role', 'unknown')
                    content = msg.get('content', '')
                    context_parts.append(f"- {role}: {content}")
            
            # Add chain of thoughts if available
            if chain_of_thoughts:
                context_parts.append("Previous analysis context:")
                for thought in chain_of_thoughts[-2:]:  # Last 2 thoughts
                    if isinstance(thought, dict):
                        content = thought.get('content', str(thought))
                    else:
                        content = str(thought)
                    context_parts.append(f"- {content}")
            
            # Add summarized context if available
            if summarized_histories:
                context_parts.append("Historical context summary:")
                for summary in summarized_histories[-1:]:  # Most recent summary
                    if isinstance(summary, dict):
                        content = summary.get('content', str(summary))
                    else:
                        content = str(summary)
                    context_parts.append(f"- {content}")
            
            therapeutic_context = "\n".join(context_parts) if context_parts else "New conversation session"
            
            # Use the main therapeutic guidance method
            response = self.provide_therapeutic_guidance(
                user_message=message,
                therapeutic_context=therapeutic_context
            )
            
            self.logger.info("Psychological analysis completed successfully")
            return response
            
        except Exception as e:
            self.logger.error(f"Error in analyze method: {str(e)}")
            # Return fallback based on detected language
            try:
                lang = self._detect_language(message)
                if lang == 'thai':
                    return "ขออภัยค่ะ เกิดปัญหาทางเทคนิค หากมีความคิดฆ่าตัวตายหรืออยู่ในภาวะฉุกเฉิน กรุณาติดต่อนักจิตวิทยาหรือหน่วยงานฉุกเฉินทันที"
                else:
                    return "I apologize for the technical issue. If you're having thoughts of self-harm or are in crisis, please contact a mental health professional or emergency services immediately."
            except:
                return "Technical difficulties. If in crisis, seek immediate professional help."

    # ===== MAIN OUTPUT METHODS =====
    def provide_therapeutic_guidance(self, user_message: str, therapeutic_context: str = "") -> str:
        """
        Main method to provide psychological guidance with language-based API routing
        
        Args:
            user_message: The client's message or concern
            therapeutic_context: Additional context about the client's situation
            
        Returns:
            Therapeutic response with guidance and support
        """
        self.logger.info("Processing therapeutic guidance request")
        self.logger.debug(f"User message: {user_message}")
        self.logger.debug(f"Therapeutic context: {therapeutic_context}")
        
        try:
            # Detect language
            detected_language = self._detect_language(user_message)
            self.logger.info(f"Detected language: {detected_language}")
            
            if detected_language == 'thai':
                # Use SEA-Lion for Thai
                messages = self._format_messages_sealion(user_message, therapeutic_context)
                response = self._generate_response_sealion(messages)
            else:
                # Use Gemini for English
                prompt = self._format_prompt_gemini(user_message, therapeutic_context)
                response = self._generate_response_gemini(prompt)
            
            if response is None:
                raise Exception(f"{detected_language} API returned None response")
                
            return response
            
        except Exception as e:
            self.logger.error(f"Error in provide_therapeutic_guidance: {str(e)}")
            # Return fallback based on detected language
            try:
                lang = self._detect_language(user_message)
                if lang == 'thai':
                    return "ขออภัยค่ะ ระบบมีปัญหาชั่วคราว หากมีความคิดทำร้ายตัวเองหรืออยู่ในภาวะวิกฤต กรุณาติดต่อนักจิตวิทยาหรือหน่วยงานฉุกเฉินทันที"
                else:
                    return "I apologize for the technical issue. If you're having thoughts of self-harm or are in crisis, please contact a mental health professional or emergency services immediately."
            except:
                return "Technical difficulties. If in crisis, seek immediate professional help."

    def get_health_status(self) -> Dict[str, Any]:
        """Get current agent health status"""
        status = {
            "status": "healthy",
            "language_routing": "thai->SEA-Lion, english->Gemini",
            "sea_lion_configured": hasattr(self, 'sea_lion_api_key') and self.sea_lion_api_key,
            "gemini_configured": hasattr(self, 'gemini_api_key') and self.gemini_api_key,
            "sea_lion_model": "aisingapore/Llama-SEA-LION-v3.5-8B-R",
            "gemini_model": self.gemini_model_name,
            "timestamp": datetime.now().isoformat(),
            "logging_enabled": True,
            "log_level": self.logger.level,
            "methods_available": ["analyze", "provide_therapeutic_guidance", "get_health_status"]
        }
        
        self.logger.debug(f"Health status check: {status}")
        return status


if __name__ == "__main__":
    # Test the completed PsychologistAgent
    try:
        psychologist = PsychologistAgent(log_level=logging.DEBUG)
    except Exception as e:
        print(f"[BOOT ERROR] Unable to start PsychologistAgent: {e}")
        raise SystemExit(1)

    # Test cases for both languages
    test_cases = [
        {
            "name": "Thai - Exam Anxiety",
            "user_message": "หนูปวดหัวและกังวลเรื่องสอบค่ะ นอนไม่หลับและกังวลว่าจะสอบตก",
            "therapeutic_context": "User: นักศึกษาอายุ 21 ปี ช่วงสอบกลางเทอม นอนน้อย (4-5 ชั่วโมง) ดื่มกาแฟมาก มีประวัติวิตกกังวลในช่วงความเครียดทางการศึกษา"
        },
        {
            "name": "English - Work Stress", 
            "user_message": "I've been feeling overwhelmed lately with work and personal life. Everything feels like too much and I can't cope.",
            "therapeutic_context": "User: Working professional, recent job change, managing family responsibilities, seeking coping strategies."
        },
        {
            "name": "Thai - Relationship Issues",
            "user_message": "ความสัมพันธ์กับแฟนมีปัญหามากค่ะ เราทะเลาะกันบ่อยๆ ไม่รู้จะแก้ไขยังไง",
            "therapeutic_context": "User: อยู่ในความสัมพันธ์ที่มั่นคง มีปัญหาการสื่อสาร ขอคำแนะนำเรื่องความสัมพันธ์และการแก้ปัญหาความขัดแย้ง"
        },
        {
            "name": "English - Anxiety Management",
            "user_message": "I keep having panic attacks and I don't know how to control them. It's affecting my daily life and work performance.",
            "therapeutic_context": "User: Experiencing frequent panic attacks, seeking anxiety management techniques, work performance concerns."
        }
    ]

    # Test the analyze method specifically
    print(f"\n{'='*60}")
    print("TESTING ANALYZE METHOD (Required by Orchestrator)")
    print(f"{'='*60}")
    
    test_message = "hello im kinda sad"
    test_history = [
        {"role": "user", "content": "Hi there"},
        {"role": "assistant", "content": "Hello! How can I help you today?"},
        {"role": "user", "content": test_message}
    ]
    test_chain_of_thoughts = [
        {"step": "analysis", "content": "User expressing mild sadness, needs supportive guidance"},
        {"step": "routing", "content": "Psychological support required"}
    ]
    test_summarized_histories = [
        {"session": "previous", "content": "User has been dealing with some personal challenges"}
    ]
    
    print(f"\n Test Message: {test_message}")
    print(f" History: {len(test_history)} messages")
    print(f" Chain of thoughts: {len(test_chain_of_thoughts)} items")
    print(f" Summaries: {len(test_summarized_histories)} items")
    
    print(f"\n ANALYZE METHOD RESPONSE:")
    print("-" * 50)
    
    analyze_response = psychologist.analyze(test_message, test_history, test_chain_of_thoughts, test_summarized_histories)
    print(analyze_response)

    # Run other tests
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n{'='*60}")
        print(f"TEST {i}: {test_case['name']}")
        print(f"{'='*60}")
        
        print(f"\n User Message: {test_case['user_message']}")
        print(f" Context: {test_case['therapeutic_context']}")
        
        print(f"\n PSYCHOLOGIST RESPONSE:")
        print("-" * 50)
        
        response = psychologist.provide_therapeutic_guidance(
            user_message=test_case['user_message'],
            therapeutic_context=test_case['therapeutic_context']
        )
        
        print(response)
        print("\n" + "="*60)

    # Test health status
    print(f"\n{'='*60}")
    print("HEALTH STATUS CHECK")
    print(f"{'='*60}")
    status = psychologist.get_health_status()
    print(json.dumps(status, indent=2, ensure_ascii=False))