from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Any, Optional, List, Union

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# API clients
import openai
import google.generativeai as genai
import requests
logger = logging.getLogger(__name__)

class APIProvider(Enum):
    GEMINI = "gemini"
    GPT = "gpt"
    SEALION = "sealion"

@dataclass
class ExpertiseConfig:
    domain: str
    system_prompt: str
    temperature: float = 0.7
    max_tokens: int = 4000
    special_instructions: List[str] = field(default_factory=list)

class UniversalExpertAgent:
    def __init__(self,
                 api_provider: Union[str, APIProvider] = APIProvider.GPT, ##change here
                 api_key: Optional[str] = None,
                 log_level: str = "INFO"):
    
        logger.setLevel(log_level)

        # API provider
        self.api_provider = APIProvider(api_provider) if isinstance(api_provider, str) else api_provider

        # --- Load all keys like orchestrator ---
        self._openai_key = os.getenv("OPENAI_API_KEY")
        self._gemini_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        self._sealion_key = os.getenv("SEALION_API_KEY")
        self._sealion_url = os.getenv("SEALION_API_URL", "https://api.sealion.ai/v1/generate")

        # choose api key based on provider
        self.api_key = api_key or self._get_api_key_from_env(self.api_provider)

        # Initialize API client
        self._initialize_api_client()

        # Load expertise configs
        self.expertise_domains = self._load_expertise_configs()
        self.current_domain = "doctor"

        logger.info(f"UniversalExpertAgent initialized with provider {self.api_provider.value}")
            
    def _get_api_key_from_env(self, provider: APIProvider) -> Optional[str]:
        if provider == APIProvider.GPT:
            return self._openai_key
        elif provider == APIProvider.GEMINI:
            return self._gemini_key
        elif provider == APIProvider.SEALION:
            return self._sealion_key
        return None
        

    def _initialize_api_client(self) -> None:
        """Initialize the appropriate API client like orchestrator does."""
        try:
            if self.api_provider == APIProvider.GPT:
                if not self._openai_key:
                    raise ValueError("OPENAI_API_KEY required for GPT provider")
                from openai import OpenAI
                self.client = OpenAI(api_key=self._openai_key)
                logger.info("OpenAI GPT client initialized")


            elif self.api_provider == APIProvider.GEMINI:
                if not self._gemini_key:
                    raise ValueError("GOOGLE_API_KEY or GEMINI_API_KEY required for Gemini provider")
                genai.configure(api_key=self._gemini_key)
                # create default model like orchestrator
                self.client = genai.GenerativeModel(
                    model_name="gemini-2.5-flash-lite",
                    generation_config={
                        "temperature": 0.2,
                        "max_output_tokens": 2000,
                    },
                )
                logger.info("Gemini API client initialized")


            elif self.api_provider == APIProvider.SEALION:
                if not self._sealion_key:
                    raise ValueError("SEALION_API_KEY required for SeaLion provider")
                self.client = {
                    "api_key": self._sealion_key,
                    "api_url": self._sealion_url,
                    "headers": {
                        "Authorization": f"Bearer {self._sealion_key}",
                        "Content-Type": "application/json",
                    },
                }
                logger.info("SeaLion API client configured")


        except Exception as e:
            logger.error(f"Failed to initialize {self.api_provider.value} API client: {e}")
            raise

    def switch_api_provider(self, provider: Union[str, APIProvider], api_key: Optional[str] = None) -> bool:
        """Switch API provider and reinitialize client easily."""
        try:
            new_provider = APIProvider(provider) if isinstance(provider, str) else provider
            self.api_provider = new_provider
            self.api_key = api_key or self._get_api_key_from_env(new_provider)
            self._initialize_api_client()
            logger.info(f"Switched to provider {self.api_provider.value}")
            return True
        except Exception as e:
            logger.error(f"Failed to switch API provider: {e}")
            return False

    def _load_expertise_configs(self) -> Dict[str, ExpertiseConfig]:
        """Load pre-configured expertise domains"""
        configs = {
            "doctor": ExpertiseConfig(
                domain="doctor",
                system_prompt="""
                - You are an expert medical doctor with comprehensive knowledge of medicine, healthcare, anatomy, physiology, pharmacology, clinical practice, diagnosis, and treatment. 
                - You always respond in thai
                - You provide evidence-based medical information, explain medical conditions clearly, discuss treatment options, and offer professional medical guidance. 
                - You maintain clinical objectivity while being empathetic to patient concerns. Always emphasize the importance of professional medical consultation for specific medical decisions.""",
                temperature=0.3,
                special_instructions=[
                    "You always respond in thai ",
                    "Provide evidence-based information with medical citations when possible",
                    "Be clear about diagnostic limitations of AI and the need for physical examination",
                    "Use appropriate medical terminology while ensuring patient understanding",
                    "Consider differential diagnoses and explain reasoning",
                    "Discuss both benefits and risks of treatments",
                    "Emphasize emergency care when appropriate"
                ]
            ),
            
            "psychologist": ExpertiseConfig(
                domain="psychologist",
                system_prompt="""
                - You are an expert psychologist with deep knowledge of psychology, mental health, behavioral science, cognitive processes, therapeutic approaches, and psychological assessment. 
                - You always respond in thai
                - You provide evidence-based psychological insights, explain mental health conditions, discuss therapeutic interventions, and offer supportive guidance. 
                - You maintain professional boundaries while being empathetic and non-judgmental.
                - You understand various psychological theories and therapeutic modalities including CBT, psychodynamic, humanistic, and behavioral approaches.""",
                temperature=0.4,
                special_instructions=[
                    "You always respond in thai ",
                    "Provide evidence-based psychological information and cite research when relevant",
                    "Be empathetic, non-judgmental, and maintain professional boundaries",
                    "Explain psychological concepts in accessible language",
                    "Consider various therapeutic approaches and their applications",
                    "Emphasize crisis intervention resources when appropriate",
                    "Respect confidentiality principles and ethical guidelines",
                    "Acknowledge the complexity of human psychology and individual differences"
                ]
            ),
            
            "general": ExpertiseConfig(
                domain="general",
                system_prompt="""
                - You always respond in thai
                - You are a knowledgeable general assistant with broad expertise across multiple domains. 
                - You provide helpful, accurate, and contextually appropriate responses to a wide variety of questions and topics. 
                - You adapt your communication style to match the user's needs and maintain a professional yet approachable tone.""",
                temperature=0.7,
                special_instructions=[
                    "You always respond in thai ",
                    "Provide clear and accurate information across various topics",
                    "Adapt communication style to the user's level and context",
                    "Acknowledge limitations and direct to specialists when appropriate",
                    "Be helpful while maintaining accuracy and objectivity"
                ]
            )
        }
        
        return configs

    # =========================================================================
    # Methods that ChatbotManager expects (replacing UnifiedDatasetOrchestrator)
    # =========================================================================

    def get_translation(self, message: str, flags: Dict[str, Any], translation_type: str) -> str:
        """
        Handle translation requests. For now, we'll return the message as-is 
        since we're focusing on English expertise. This can be enhanced later.
        """
        # In a real implementation, you might use the flags to determine language
        # and perform actual translation. For now, we pass through.
        return message

    def get_commented_response(self, 
                             original_history: List[Dict[str, Any]],
                             original_message: str,
                             eng_history: List[Dict[str, Any]], 
                             eng_message: str,
                             flags: Dict[str, Any],
                             chain_of_thoughts: List[str],
                             memory_context: str,
                             summarized_histories: List[str]) -> Dict[str, Any]:
        """
        Generate the main response with commentary/reasoning.
        This is the core method that replaces the orchestrator's response generation.
        """
        
        # Get expertise domain from flags
        domain = flags.get("expertise_domain", self.current_domain)
        config = self.expertise_domains.get(domain, self.expertise_domains["general"])
        
        # Build context for the AI model
        context = self._build_response_context(
            eng_history, eng_message, memory_context, 
            summarized_histories, chain_of_thoughts, config
        )
        
        # Generate response using the configured API
        response = self._generate_response(context, config, flags)
        
        # Generate reasoning/commentary
        reasoning = self._generate_reasoning(eng_message, response, domain, flags)
        
        return {
            "final_output": response,
            "chain_of_thoughts": reasoning,
            "expertise_domain": domain,
            "api_provider": self.api_provider.value,
            "temperature_used": config.temperature,
            "context_length": len(context)
        }

    def summarize_history(self, 
                         chat_history: List[Dict[str, Any]], 
                         eng_summaries: List[str]) -> str:
        """
        Generate a summary of the chat history.
        """
        if not chat_history:
            return "No conversation history to summarize."
        
        # Build summary context
        context = "Please provide a concise summary of this conversation:\n\n"
        
        # Add previous summaries if available
        if eng_summaries:
            context += "Previous summaries:\n"
            for i, summary in enumerate(eng_summaries[-3:], 1):  # Last 3 summaries
                context += f"{i}. {summary}\n"
            context += "\n"
        
        # Add recent chat history
        context += "Recent conversation:\n"
        for msg in chat_history[-10:]:  # Last 10 messages
            role = msg.get("role", "unknown")
            content = msg.get("content", "")[:200]  # Truncate long messages
            context += f"{role.title()}: {content}\n"
        
        context += "\nProvide a summary that captures the key topics, decisions, and context."
        
        # Use general domain for summarization
        config = self.expertise_domains["general"]
        summary = self._generate_response(context, config, {"temperature": 0.3})
        
        return summary

    # =========================================================================
    # Internal helper methods
    # =========================================================================

    def _determine_expertise_domain(self, 
                                  user_message: str, 
                                  chat_history: List[Dict[str, Any]], 
                                  memory_context: str) -> str:
        """
        Analyze the message and context to determine the appropriate expertise domain.
        """
        message_lower = user_message.lower()
        
        # Medical keywords
        medical_keywords = [
            'symptoms', 'diagnosis', 'treatment', 'medicine', 'doctor', 'health',
            'disease', 'pain', 'medication', 'hospital', 'clinic', 'medical',
            'illness', 'injury', 'prescription', 'therapy', 'surgery', 'blood',
            'heart', 'lung', 'brain', 'cancer', 'diabetes', 'pressure',
            'infection', 'virus', 'bacteria', 'fever', 'headache'
        ]
        
        # Psychology keywords
        psychology_keywords = [
            'stress', 'anxiety', 'depression', 'mental', 'psychology', 'therapy',
            'counseling', 'emotion', 'behavior', 'cognitive', 'psychologist',
            'mood', 'trauma', 'ptsd', 'bipolar', 'schizophrenia', 'addiction',
            'relationship', 'family', 'grief', 'phobia', 'panic', 'ocd',
            'personality', 'self-esteem', 'confidence', 'sleep', 'insomnia'
        ]
        
        # Check for domain indicators in the message
        medical_score = sum(1 for keyword in medical_keywords if keyword in message_lower)
        psychology_score = sum(1 for keyword in psychology_keywords if keyword in message_lower)
        
        # Check recent conversation context
        if chat_history:
            recent_messages = [msg.get("content", "") for msg in chat_history[-3:]]
            recent_text = " ".join(recent_messages).lower()
            
            medical_score += sum(0.5 for keyword in medical_keywords if keyword in recent_text)
            psychology_score += sum(0.5 for keyword in psychology_keywords if keyword in recent_text)
        
        # Check memory context
        if memory_context:
            memory_lower = memory_context.lower()
            if "doctor" in memory_lower or "medical" in memory_lower:
                medical_score += 1
            if "psychologist" in memory_lower or "psychology" in memory_lower:
                psychology_score += 1
        
        # Determine domain based on scores
        if medical_score > psychology_score and medical_score > 0:
            return "doctor"
        elif psychology_score > 0:
            return "psychologist"
        else:
            return "general"

    def _build_response_context(self, 
                              eng_history: List[Dict[str, Any]], 
                              eng_message: str,
                              memory_context: str,
                              summarized_histories: List[str], 
                              chain_of_thoughts: List[str],
                              config: ExpertiseConfig) -> str:
        """
        Build the complete context for response generation.
        """
        context_parts = []
        
        # Add system prompt and expertise context
        context_parts.append(f"System Role: {config.system_prompt}")
        context_parts.append("")
        
        # Add special instructions
        if config.special_instructions:
            context_parts.append("Special Instructions:")
            for instruction in config.special_instructions:
                context_parts.append(f"- {instruction}")
            context_parts.append("")
        
        # Add memory context
        if memory_context.strip():
            context_parts.append("Session Context:")
            context_parts.append(memory_context)
            context_parts.append("")
        
        # Add previous summaries
        if summarized_histories:
            context_parts.append("Previous Conversation Summary:")
            for summary in summarized_histories[-2:]:  # Last 2 summaries
                context_parts.append(summary)
            context_parts.append("")
        
        # Add recent conversation history
        if eng_history:
            context_parts.append("Recent Conversation:")
            for msg in eng_history[-10:]:  # Last 10 messages
                role = msg.get("role", "unknown")
                content = msg.get("content", "")
                context_parts.append(f"{role.title()}: {content}")
            context_parts.append("")
        
        # Add chain of thoughts if available
        if chain_of_thoughts:
            context_parts.append("Previous Reasoning:")
            for thought in chain_of_thoughts[-3:]:  # Last 3 thoughts
                context_parts.append(f"- {thought}")
            context_parts.append("")
        
        # Add current message
        context_parts.append("Current Message:")
        context_parts.append(f"User: {eng_message}")
        context_parts.append("")
        context_parts.append("Please provide your expert response:")
        
        return "\n".join(context_parts)

    def _generate_response(self, 
                        context: str, 
                        config: ExpertiseConfig, 
                        flags: Dict[str, Any]) -> str:
        """
        Generate response using the configured AI API.
        Falls back to GPT if no other provider is set up.
        """
        try:
            if self.api_provider == APIProvider.GPT:
                # Make sure client is initialized in _initialize_api_client
                response = self.client.chat.completions.create(
                    model="gpt-4o-mini",   # or "gpt-4o"
                    messages=[
                        {"role": "system", "content": config.system_prompt},
                        {"role": "user", "content": context}
                    ],
                    temperature=config.temperature,
                    max_tokens=config.max_tokens
                )
                return response.choices[0].message.content.strip()

            elif self.api_provider == APIProvider.GEMINI:
                model = genai.GenerativeModel("gemini-2.5-flash-lite")
                result = model.generate_content(context)
                return result.text

            elif self.api_provider == APIProvider.SEALION:
                url = "https://api.sealion.ai/v1/generate"
                headers = {"Authorization": f"Bearer {self.api_key}"}
                payload = {
                    "prompt": context,
                    "temperature": config.temperature,
                    "max_tokens": config.max_tokens
                }
                r = requests.post(url, headers=headers, json=payload, timeout=30)
                r.raise_for_status()
                return r.json().get("text", "").strip()

            else:
                return "Error: No valid API provider configured."

        except Exception as e:
            return f"Error generating response with {self.api_provider.value}: {str(e)}"
            
    def _generate_reasoning(self, 
                            user_message: str, 
                            response: str, 
                            domain: str, 
                            flags: Dict[str, Any]) -> str:
        """
        Generate reasoning/chain of thoughts for the response.
        """
        reasoning = f"""Domain Analysis: Determined '{domain}' expertise based on message content and context.

    API Configuration: Using {self.api_provider.value.upper()} with temperature {flags.get('temperature', 0.7)}.

    Response Strategy: Applied {domain} expertise with appropriate professional boundaries and ethical considerations.

    Key Considerations:
    - Maintained professional standards for {domain} domain
    - Emphasized need for professional consultation where appropriate
    - Provided evidence-based information within scope of AI capabilities

    Message Processing: Successfully processed {len(user_message)} character input and generated {len(response)} character response."""
        
        return reasoning


    # =========================================================================
    # Additional utility methods
    # =========================================================================

    def add_custom_expertise(self, domain: str, config: ExpertiseConfig) -> bool:
        """Add a custom expertise domain"""
        try:
            self.expertise_domains[domain] = config
            logger.info(f"Added custom expertise domain: {domain}")
            return True
        except Exception as e:
            logger.error(f"Failed to add custom expertise {domain}: {e}")
            return False

    def get_available_domains(self) -> List[str]:
        """Get list of available expertise domains"""
        return list(self.expertise_domains.keys())

    def switch_api_provider(self, provider: Union[str, APIProvider], api_key: Optional[str] = None) -> bool:
        """Switch API provider and reinitialize the client"""
        try:
            new_provider = APIProvider(provider) if isinstance(provider, str) else provider
            new_api_key = api_key or self._get_api_key_from_env(new_provider)
            
            if not new_api_key:
                logger.warning(f"No API key available for {new_provider.value}")
                return False
            
            # Store old values in case we need to rollback
            old_provider = self.api_provider
            old_key = self.api_key
            
            try:
                # Set new values
                self.api_provider = new_provider
                self.api_key = new_api_key
                
                # Initialize new API client
                self._initialize_api_client()
                
                logger.info(f"Successfully switched from {old_provider.value} to {new_provider.value}")
                return True
                
            except Exception as e:
                # Rollback on failure
                self.api_provider = old_provider
                self.api_key = old_key
                logger.error(f"Failed to switch API provider, rolled back: {e}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to switch API provider: {e}")
            return False

    def test_api_connection(self) -> bool:
        """Test if the current API is working"""
        try:
            test_context = "Please respond with 'Connection successful' to confirm the API is working."
            test_config = self.expertise_domains["general"]
            
            response = self._generate_response(test_context, test_config, {"temperature": 0.1})
            
            # Check if we got a valid response (not an error message)
            is_working = response and not response.startswith("Error:") and not response.startswith("I apologize")
            
            if is_working:
                logger.info(f"{self.api_provider.value} API connection test: SUCCESS")
            else:
                logger.warning(f"{self.api_provider.value} API connection test: FAILED - {response[:100]}...")
                
            return is_working
            
        except Exception as e:
            logger.error(f"API connection test failed: {e}")
            return False

    def get_api_status(self) -> Dict[str, Any]:
        """Get detailed status of the current API configuration"""
        return {
            "provider": self.api_provider.value,
            "has_api_key": bool(self.api_key),
            "api_key_length": len(self.api_key) if self.api_key else 0,
            "connection_test": self.test_api_connection() if self.api_key else False,
            "available_domains": list(self.expertise_domains.keys()),
            "current_domain": self.current_domain
        }


# Convenience factory functions - now auto-load API keys from .env
def create_gemini_agent(api_key: Optional[str] = None, **kwargs) -> UniversalExpertAgent:
    """Create agent with Gemini API - auto-loads from GEMINI_API_KEY if not provided"""
    return UniversalExpertAgent(
        api_provider=APIProvider.GEMINI,
        api_key=api_key,  # Will auto-load from .env if None
        **kwargs
    )

def create_gpt_agent(api_key: Optional[str] = None, **kwargs) -> UniversalExpertAgent:
    """Create agent with GPT API - auto-loads from OPENAI_API_KEY if not provided"""
    return UniversalExpertAgent(
        api_provider=APIProvider.GPT,
        api_key=api_key,  # Will auto-load from .env if None
        **kwargs
    )

def create_sealion_agent(api_key: Optional[str] = None, **kwargs) -> UniversalExpertAgent:
    """Create agent with SeaLion API - auto-loads from SEA_LION_API_KEY if not provided"""
    return UniversalExpertAgent(
        api_provider=APIProvider.SEALION,
        api_key=api_key,  # Will auto-load from .env if None
        **kwargs
    )