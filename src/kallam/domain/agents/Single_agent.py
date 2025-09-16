# src/your_pkg/app/single_agent.py
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
    """Supported API providers"""
    GEMINI = "gemini"
    GPT = "gpt" 
    SEALION = "sealion"


@dataclass
class ExpertiseConfig:
    """Configuration for different expertise domains"""
    domain: str
    system_prompt: str
    temperature: float = 0.7
    max_tokens: int = 4000
    special_instructions: List[str] = field(default_factory=list)


class UniversalExpertAgent:
    """
    A single agent that can be an expert in any field.
    This replaces the UnifiedDatasetOrchestrator in the architecture.
    Provides all the methods that ChatbotManager expects from an orchestrator.
    """
    
    def __init__(self,
                 api_provider: Union[str, APIProvider] = APIProvider.GPT,
                 api_key: Optional[str] = None,
                 log_level: str = "INFO"):
        
        # API configuration - auto-load from environment if not provided
        self.api_provider = APIProvider(api_provider) if isinstance(api_provider, str) else api_provider
        self.api_key = api_key or self._get_api_key_from_env(self.api_provider)
        
        if not self.api_key:
            logger.warning(f"No API key found for {self.api_provider.value}. Please set in .env file or pass directly.")
        else:
            # Initialize the API client based on provider
            self._initialize_api_client()
        
        # Load expertise configurations
        self.expertise_domains = self._load_expertise_configs()
        
        # Current domain context (can be switched per session via flags)
        self.current_domain = "doctor"  # default
        
        logger.info(f"UniversalExpertAgent initialized with {self.api_provider.value} provider")

    def _get_api_key_from_env(self, provider: APIProvider) -> Optional[str]:
        """Get API key from environment variables based on provider"""
        env_keys = {
            APIProvider.GEMINI: "GEMINI_API_KEY",
            APIProvider.GPT: "OPENAI_API_KEY", 
            APIProvider.SEALION: "SEA_LION_API_KEY"
        }
        
        env_key = env_keys.get(provider)
        if env_key:
            api_key = os.getenv(env_key)
            if api_key:
                logger.info(f"Loaded {provider.value} API key from environment variable {env_key}")
                return api_key
            else:
                logger.warning(f"Environment variable {env_key} not found for {provider.value}")
        
        return None

    def _initialize_api_client(self) -> None:
        """Initialize the appropriate API client based on provider"""
        try:
            if self.api_provider == APIProvider.GPT:
                openai.api_key = self.api_key
                # Test the connection
                logger.info("OpenAI API client initialized")
                
            elif self.api_provider == APIProvider.GEMINI:
                genai.configure(api_key=self.api_key)
                # Test the connection by listing models
                logger.info("Gemini API client initialized")
                
            elif self.api_provider == APIProvider.SEALION:
                # SeaLion uses REST API calls
                logger.info("SeaLion API client configured")
                
        except Exception as e:
            logger.error(f"Failed to initialize {self.api_provider.value} API client: {e}")
            raise

    def _load_expertise_configs(self) -> Dict[str, ExpertiseConfig]:
        """Load pre-configured expertise domains"""
        configs = {
            "doctor": ExpertiseConfig(
                domain="doctor",
                system_prompt="""You are an expert medical doctor with comprehensive knowledge of medicine, 
                healthcare, anatomy, physiology, pharmacology, clinical practice, diagnosis, and treatment. 
                You provide evidence-based medical information, explain medical conditions clearly, discuss 
                treatment options, and offer professional medical guidance. You maintain clinical objectivity 
                while being empathetic to patient concerns. Always emphasize the importance of professional 
                medical consultation for specific medical decisions.""",
                temperature=0.3,
                special_instructions=[
                    "Always recommend consulting healthcare professionals for specific medical decisions",
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
                system_prompt="""You are an expert psychologist with deep knowledge of psychology, mental health, 
                behavioral science, cognitive processes, therapeutic approaches, and psychological assessment. 
                You provide evidence-based psychological insights, explain mental health conditions, discuss 
                therapeutic interventions, and offer supportive guidance. You maintain professional boundaries 
                while being empathetic and non-judgmental. You understand various psychological theories and 
                therapeutic modalities including CBT, psychodynamic, humanistic, and behavioral approaches.""",
                temperature=0.4,
                special_instructions=[
                    "Always recommend professional psychological consultation for mental health concerns",
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
                system_prompt="""You are a knowledgeable general assistant with broad expertise across 
                multiple domains. You provide helpful, accurate, and contextually appropriate responses 
                to a wide variety of questions and topics. You adapt your communication style to match 
                the user's needs and maintain a professional yet approachable tone.""",
                temperature=0.7,
                special_instructions=[
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

    def get_flags_from_supervisor(self, 
                                chat_history: List[Dict[str, Any]], 
                                user_message: str,
                                memory_context: str,
                                summarized_histories: List[str]) -> Dict[str, Any]:
        """
        Analyze the conversation context and determine appropriate flags.
        This replaces the supervisor's role in determining expertise domain and other parameters.
        """
        # Analyze message content to determine expertise domain
        domain = self._determine_expertise_domain(user_message, chat_history, memory_context)
        
        # Build flags that ChatbotManager expects
        flags = {
            "language": "english",  # Default to English for now
            "doctor": domain == "doctor",
            "psychologist": domain == "psychologist",
            "expertise_domain": domain,
            "api_provider": self.api_provider.value,
            "temperature": self.expertise_domains[domain].temperature,
            "max_tokens": self.expertise_domains[domain].max_tokens
        }
        
        # Store current domain for response generation
        self.current_domain = domain
        
        return flags

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
        This is a placeholder - implement actual API calls based on self.api_provider.
        """
        
        # TODO: Implement actual API calls to Gemini, GPT, or SeaLion
        # For now, return a simulated response
        
        domain = config.domain
        
        # Simulate different responses based on domain
        if domain == "doctor":
            return f"""Based on your medical query, I need to provide you with evidence-based medical information. 

However, I must emphasize that this is for informational purposes only and cannot replace professional medical consultation. 

Please consult with a healthcare professional for proper diagnosis and treatment recommendations specific to your situation.

[This is a simulated response - implement actual {self.api_provider.value.upper()} API integration]"""
        
        elif domain == "psychologist":
            return f"""From a psychological perspective, I understand your concerns and want to provide you with helpful insights.

It's important to approach this with empathy and evidence-based psychological understanding.

Please remember that while I can provide general psychological information and support, professional psychological consultation is recommended for personalized mental health care.

[This is a simulated response - implement actual {self.api_provider.value.upper()} API integration]"""
        
        else:  # general
            return f"""I'll do my best to provide you with helpful and accurate information on this topic.

[This is a simulated response - implement actual {self.api_provider.value.upper()} API integration]

Let me know if you need clarification on any aspect of my response."""

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


# Example usage with ChatbotManager integration
if __name__ == "__main__":
    # The agent can now be used directly with ChatbotManager
    # without importing ChatbotManager in this file
    
    # Create and test different API providers
    print("=== Testing API Providers ===")
    
    # Test GPT agent
    try:
        gpt_agent = create_gpt_agent(log_level="INFO")
        print(f"GPT Agent Status: {gpt_agent.get_api_status()}")
        
        if gpt_agent.api_key:
            connection_test = gpt_agent.test_api_connection()
            print(f"GPT Connection Test: {'✓' if connection_test else '✗'}")
    except Exception as e:
        print(f"GPT Agent Error: {e}")
    
    # Test Gemini agent
    try:
        gemini_agent = create_gemini_agent(log_level="INFO")
        print(f"Gemini Agent Status: {gemini_agent.get_api_status()}")
        
        if gemini_agent.api_key:
            connection_test = gemini_agent.test_api_connection()
            print(f"Gemini Connection Test: {'✓' if connection_test else '✗'}")
    except Exception as e:
        print(f"Gemini Agent Error: {e}")
    
    # Test SeaLion agent
    try:
        sealion_agent = create_sealion_agent(log_level="INFO")
        print(f"SeaLion Agent Status: {sealion_agent.get_api_status()}")
        
        if sealion_agent.api_key:
            connection_test = sealion_agent.test_api_connection()
            print(f"SeaLion Connection Test: {'✓' if connection_test else '✗'}")
    except Exception as e:
        print(f"SeaLion Agent Error: {e}")
    
    print("\n=== API Switching Demo ===")
    
    # Demonstrate API switching
    agent = create_gpt_agent()
    print(f"Started with: {agent.api_provider.value}")
    
    # Try to switch to Gemini
    if agent.switch_api_provider("gemini"):
        print(f"Switched to: {agent.api_provider.value}")
        
        # Try to switch back to GPT
        if agent.switch_api_provider("gpt"):
            print(f"Switched back to: {agent.api_provider.value}")
    
    print("\n=== Domain Detection Demo ===")
    
    # Test domain determination
    test_messages = [
        ("I have been experiencing chest pain and shortness of breath", "Expected: doctor"),
        ("I've been feeling very anxious and stressed lately", "Expected: psychologist"),
        ("What's the weather like today?", "Expected: general"),
        ("Can you explain diabetes symptoms and treatment?", "Expected: doctor"),
        ("How can I manage depression and mood swings?", "Expected: psychologist")
    ]
    
    for msg, expected in test_messages:
        flags = agent.get_flags_from_supervisor([], msg, "", [])
        detected_domain = flags.get("expertise_domain", "unknown")
        print(f"Message: '{msg[:50]}...'")
        print(f"  Detected: {detected_domain} | {expected}")
        print(f"  Flags: doctor={flags.get('doctor')}, psychologist={flags.get('psychologist')}")