# src/your_pkg/app/universal_agent.py
from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Any, Optional, List, Union
from pathlib import Path

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

from .chatbot_manager import ChatbotManager

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
    A single agent that can be an expert in any field using ChatbotManager as wrapper.
    Supports multiple API providers (Gemini, GPT, SeaLion) and maintains expertise
    across all domains through dynamic prompt engineering.
    """
    
    def __init__(self,
                 api_provider: Union[str, APIProvider] = APIProvider.GPT,
                 api_key: Optional[str] = None,
                 db_path: str = "universal_agent.db",
                 log_level: str = "INFO"):
        
        # Initialize ChatbotManager wrapper
        self.chatbot = ChatbotManager(
            db_path=db_path,
            summarize_every_n_messages=15,
            message_limit=20,
            sunmmary_limit=10,
            chain_of_thoughts_limit=8,
            log_level=log_level,
            log_name="universal_agent"
        )
        
        # API configuration - auto-load from environment if not provided
        self.api_provider = APIProvider(api_provider) if isinstance(api_provider, str) else api_provider
        self.api_key = api_key or self._get_api_key_from_env(self.api_provider)
        
        if not self.api_key:
            logger.warning(f"No API key found for {self.api_provider.value}. Please set in .env file or pass directly.")
        
        # Load expertise configurations
        self.expertise_domains = self._load_expertise_configs()
        
        # Session tracking for context switching
        self.active_sessions: Dict[str, Dict[str, Any]] = {}
        
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
            )
        }
        
        return configs

    def start_conversation(self, 
                          expertise_domain: str = "doctor",
                          initial_context: Optional[str] = None,
                          custom_instructions: Optional[List[str]] = None) -> str:
        """Start a new conversation session with specific expertise focus"""
        
        # Get expertise configuration
        if expertise_domain not in self.expertise_domains:
            logger.warning(f"Unknown domain '{expertise_domain}', using doctor expertise")
            expertise_domain = "doctor"
        
        config = self.expertise_domains[expertise_domain]
        
        # Build system context
        system_context = self._build_system_context(config, initial_context, custom_instructions)
        
        # Start session through ChatbotManager
        session_id = self.chatbot.start_session(saved_memories=system_context)
        
        # Track session
        self.active_sessions[session_id] = {
            "domain": expertise_domain,
            "config": config,
            "start_time": time.time(),
            "message_count": 0
        }
        
        logger.info(f"Started {expertise_domain} session: {session_id}")
        return session_id

    def _build_system_context(self, 
                            config: ExpertiseConfig,
                            initial_context: Optional[str] = None,
                            custom_instructions: Optional[List[str]] = None) -> str:
        """Build comprehensive system context for the agent"""
        
        context_parts = [
            f"API Provider: {self.api_provider.value.upper()}",
            f"Expertise Domain: {config.domain.title()}",
            f"Temperature: {config.temperature}",
            "",
            "System Prompt:",
            config.system_prompt,
            ""
        ]
        
        if config.special_instructions:
            context_parts.extend([
                "Special Instructions:",
                *[f"- {instruction}" for instruction in config.special_instructions],
                ""
            ])
        
        if custom_instructions:
            context_parts.extend([
                "Custom Instructions:",
                *[f"- {instruction}" for instruction in custom_instructions],
                ""
            ])
        
        if initial_context:
            context_parts.extend([
                "Initial Context:",
                initial_context,
                ""
            ])
        
        context_parts.extend([
            f"Session started at: {time.strftime('%Y-%m-%d %H:%M:%S')}",
            "Ready to provide expert assistance in this domain."
        ])
        
        return "\n".join(context_parts)

    def ask(self, session_id: str, question: str) -> str:
        """Ask a question to the universal expert agent"""
        
        if session_id not in self.active_sessions:
            raise ValueError(f"Session {session_id} not found. Start a conversation first.")
        
        # Update session tracking
        session_info = self.active_sessions[session_id]
        session_info["message_count"] += 1
        session_info["last_activity"] = time.time()
        
        # Get response through ChatbotManager
        try:
            response = self.chatbot.handle_message(session_id, question)
            
            # Log interaction
            logger.debug(f"Question processed in {session_info['domain']} domain: {len(question)} chars -> {len(response)} chars")
            
            return response
            
        except Exception as e:
            logger.error(f"Error processing question in session {session_id}: {e}")
            raise

    def switch_expertise(self, session_id: str, new_domain: str) -> bool:
        """Switch expertise domain mid-conversation"""
        
        if session_id not in self.active_sessions:
            raise ValueError(f"Session {session_id} not found")
        
        if new_domain not in self.expertise_domains:
            logger.warning(f"Unknown domain '{new_domain}', switch failed")
            return False
        
        # Update session tracking
        old_domain = self.active_sessions[session_id]["domain"]
        self.active_sessions[session_id]["domain"] = new_domain
        self.active_sessions[session_id]["config"] = self.expertise_domains[new_domain]
        
        # Add context switch message
        switch_message = f"""
        EXPERTISE DOMAIN SWITCH:
        Previous: {old_domain.title()}
        Current: {new_domain.title()}
        
        New System Prompt: {self.expertise_domains[new_domain].system_prompt}
        
        Please adapt your responses to reflect expertise in {new_domain.title()}.
        """
        
        try:
            self.chatbot.handle_message(session_id, switch_message)
            logger.info(f"Switched expertise from {old_domain} to {new_domain} for session {session_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to switch expertise: {e}")
            # Rollback
            self.active_sessions[session_id]["domain"] = old_domain
            self.active_sessions[session_id]["config"] = self.expertise_domains[old_domain]
            return False

    def get_session_info(self, session_id: str) -> Dict[str, Any]:
        """Get detailed session information"""
        
        if session_id not in self.active_sessions:
            return {}
        
        # Get base session info from ChatbotManager
        base_info = self.chatbot.get_session_stats(session_id)
        
        # Add agent-specific info
        agent_info = self.active_sessions[session_id].copy()
        agent_info["duration_seconds"] = time.time() - agent_info["start_time"]
        
        return {
            **base_info,
            "agent_info": agent_info
        }

    def list_active_sessions(self) -> List[Dict[str, Any]]:
        """List all active sessions with their domains"""
        
        sessions = []
        for session_id, info in self.active_sessions.items():
            sessions.append({
                "session_id": session_id,
                "domain": info["domain"],
                "message_count": info["message_count"],
                "duration": time.time() - info["start_time"],
                "last_activity": info.get("last_activity", info["start_time"])
            })
        
        return sorted(sessions, key=lambda x: x["last_activity"], reverse=True)

    def close_session(self, session_id: str) -> bool:
        """Close a session"""
        
        # Close in ChatbotManager
        closed = self.chatbot.close_session(session_id)
        
        # Remove from tracking
        if session_id in self.active_sessions:
            del self.active_sessions[session_id]
            logger.info(f"Closed session: {session_id}")
        
        return closed

    def export_session(self, session_id: str) -> str:
        """Export session to JSON"""
        return self.chatbot.export_session_json(session_id)

    def export_all_sessions(self) -> str:
        """Export all sessions to JSON"""
        return self.chatbot.export_all_sessions_json()

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

    def cleanup_old_sessions(self, days_old: int = 7) -> int:
        """Clean up old sessions"""
        count = self.chatbot.cleanup_old_sessions(days_old)
        
        # Clean up tracking for non-existent sessions
        current_sessions = {s["session_id"] for s in self.chatbot.list_sessions(active_only=False)}
        to_remove = [sid for sid in self.active_sessions.keys() if sid not in current_sessions]
        
        for sid in to_remove:
            del self.active_sessions[sid]
        
        logger.info(f"Cleaned up {count} old sessions, removed {len(to_remove)} from tracking")
        return count


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


# Example usage
if __name__ == "__main__":
    # Example of how to use the Universal Expert Agent
    
    # Create agent with auto-loaded API keys from .env
    agent = create_gpt_agent(log_level="INFO")  # Will use OPENAI_API_KEY from .env
    
    # Or create with specific provider
    # agent = create_gemini_agent()  # Uses GEMINI_API_KEY from .env
    # agent = create_sealion_agent()  # Uses SEA_LION_API_KEY from .env
    
    # Start a doctor expertise session
    doctor_session = agent.start_conversation(
        expertise_domain="doctor",
        initial_context="Patient consultation context",
        custom_instructions=["Focus on evidence-based medical recommendations"]
    )
    
    # Ask medical questions
    response1 = agent.ask(doctor_session, "What are the symptoms and treatment options for hypertension?")
    print("Doctor Response:", response1)
    
    # Switch to psychology expertise mid-conversation
    agent.switch_expertise(doctor_session, "psychologist")
    response2 = agent.ask(doctor_session, "How does chronic stress affect mental health and what coping strategies would you recommend?")
    print("Psychologist Response:", response2)
    
    # Start a separate psychology session
    psychology_session = agent.start_conversation(
        expertise_domain="psychologist",
        initial_context="Mental health counseling session"
    )
    
    response3 = agent.ask(psychology_session, "Can you explain cognitive behavioral therapy and its applications?")
    print("Psychology Response:", response3)
    
    # List all active sessions
    sessions = agent.list_active_sessions()
    print("Active Sessions:", sessions)
    
    # Export a session
    export_path = agent.export_session(doctor_session)
    print("Exported to:", export_path)
    
    # Close sessions
    agent.close_session(doctor_session)
    agent.close_session(psychology_session)