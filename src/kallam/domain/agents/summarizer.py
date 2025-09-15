import os
import json
import logging
import re
from google import genai
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional

from dotenv import load_dotenv
load_dotenv()


class SummarizerAgent:
    def __init__(self, log_level: int = logging.INFO):
        self._setup_logging(log_level)
        self._setup_api_clients()
        self.logger.info("Summarizer Agent initialized successfully")

    def _setup_logging(self, log_level: int) -> None:
        self.logger = logging.getLogger(f"{__name__}.SummarizerAgent")
        self.logger.setLevel(log_level)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
            self.logger.addHandler(handler)

    def _setup_api_clients(self) -> None:
        self.gemini_api_key = os.getenv("GEMINI_API_KEY")
        if not self.gemini_api_key:
            raise ValueError("GEMINI_API_KEY not found in environment variables")
        
        self.gemini_client = genai.Client(api_key=self.gemini_api_key)
        self.gemini_model_name = "gemini-1.5-flash"

    def _generate_response(self, prompt: str) -> str:
        try:
            response = self.gemini_client.models.generate_content(
                model=self.gemini_model_name,
                contents=[prompt]
            )
            return response.text.strip() if response.text else "Unable to generate summary"
        except Exception as e:
            self.logger.error(f"Error generating response: {str(e)}")
            return "Error generating summary"

    def _format_history(self, history: List[Dict[str, str]]) -> str:
        if not history:
            return "No conversation history"
        
        formatted = []
        for msg in history:
            role = msg.get('role', 'unknown')
            content = msg.get('content', '')
            formatted.append(f"{role}: {content}")
        return "\n".join(formatted)

    def summarize_conversation_history(self, response_history: List[Dict[str, str]], **kwargs) -> str:
        """Summarize conversation history"""
        formatted_history = self._format_history(response_history)
        
        prompt = f"""
**Your Role:** 
you are a medical/psychological summarization assistant.

**Core Rules:**
- Summarize the conversation focusing on key points and medical/health or psychological information:

{formatted_history}

Create a brief summary highlighting:
- Main topics discussed
- Any health concerns or symptoms mentioned  
- Important advice or recommendations given
- Patient's emotional state or concerns

Keep it concise and medically relevant."""

        return self._generate_response(prompt)

    def summarize_medical_session(self, session_history: List[Dict[str, str]], **kwargs) -> str:
        """Summarize a medical session"""
        formatted_history = self._format_history(session_history)
        
        prompt = f"""Summarize this medical session:

{formatted_history}

Focus on:
- Chief complaints and symptoms
- Assessment and observations
- Treatment recommendations
- Follow-up requirements

Keep it professional and structured for medical records."""

        return self._generate_response(prompt)

    def summarize(self, chat_history: List[Dict[str, str]], existing_summaries: List[Dict[str, str]]) -> str:
        """Main summarization method called by orchestrator"""
        formatted_history = self._format_history(chat_history)
        
        # Include previous summaries if available
        context = ""
        if existing_summaries:
            summaries_text = "\n".join([s.get('summary', s.get('content', '')) for s in existing_summaries])
            context = f"\nPrevious summaries:\n{summaries_text}\n"

        prompt = f"""Create a comprehensive summary of this conversation:{context}

Recent conversation:
{formatted_history}

Provide a summary that:
- Combines new information with previous context
- Highlights medical/psychological insights
- Notes patient progress or changes
- Maintains continuity of care focus

Keep it concise but comprehensive."""

        return self._generate_response(prompt)


if __name__ == "__main__":
    # Test the Summarizer Agent
    try:
        summarizer = SummarizerAgent(log_level=logging.DEBUG)
        
        # Test conversation summary
        print("=== TEST: CONVERSATION SUMMARY ===")
        chat_history = [
            {"role": "user", "content": "I've been having headaches for the past week"},
            {"role": "assistant", "content": "Tell me more about these headaches - when do they occur and how severe are they?"},
            {"role": "user", "content": "They're usually worse in the afternoon, around 7/10 pain level"},
            {"role": "assistant", "content": "That sounds concerning. Have you noticed any triggers like stress, lack of sleep, or screen time?"}
        ]
        
        summary = summarizer.summarize_conversation_history(response_history=chat_history)
        print(summary)
        
        # Test medical session summary
        print("\n=== TEST: MEDICAL SESSION SUMMARY ===")
        session_summary = summarizer.summarize_medical_session(session_history=chat_history)
        print(session_summary)
        
        # Test comprehensive summary (orchestrator method)
        print("\n=== TEST: COMPREHENSIVE SUMMARY ===")
        existing_summaries = [{"summary": "Patient reported initial headache symptoms"}]
        comprehensive = summarizer.summarize(chat_history=chat_history, existing_summaries=existing_summaries)
        print(comprehensive)
        
    except Exception as e:
        print(f"Error testing Summarizer Agent: {e}")