import json
import sqlite3
import uuid
import logging
from datetime import datetime
from pathlib import Path
import time
import threading
from typing import Optional, Dict, List, Any
from contextlib import contextmanager
from datetime import timedelta

# Import your prompt.py functions
from chatbot_prompt import KaLLaMChatbot

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

EXPORT_FOLDER = "exported_sessions"

class ChatbotManager:
    def __init__(self, db_path: str = "chatbot_data.db"):
        self.chatbot = KaLLaMChatbot(api_provider="sea_lion")
        self.db_path = Path(db_path)
        self.lock = threading.Lock()
        self._create_tables()
        logger.info(f"ChatbotManager initialized with database: {self.db_path}")

    @contextmanager
    def _get_connection(self):
        """Context manager for database connections with proper error handling."""
        conn = None
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row  # Enable column access by name
            yield conn
        except sqlite3.Error as e:
            if conn:
                conn.rollback()
            logger.error(f"Database error: {e}")
            raise RuntimeError(f"Database operation failed: {e}")
        finally:
            if conn:
                conn.close()

    def _create_tables(self):
        """Create database tables with improved schema."""
        with self._get_connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    timestamp TEXT NOT NULL,
                    last_activity TEXT NOT NULL,
                    condition TEXT,
                    total_messages INTEGER DEFAULT 0,
                    total_summaries INTEGER DEFAULT 0,
                    model_used TEXT DEFAULT 'gemini-pro',
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    is_active BOOLEAN DEFAULT 1
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    message_id TEXT UNIQUE NOT NULL,
                    timestamp TEXT NOT NULL,
                    role TEXT NOT NULL CHECK(role IN ('user', 'assistant', 'system')),
                    content TEXT NOT NULL,
                    tokens_input INTEGER DEFAULT 0,
                    tokens_output INTEGER DEFAULT 0,
                    latency_ms INTEGER,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY(session_id) REFERENCES sessions(session_id) ON DELETE CASCADE
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS summaries (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    summary TEXT NOT NULL,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY(session_id) REFERENCES sessions(session_id) ON DELETE CASCADE
                )
            """)
            
            # Create indexes for better performance
            conn.execute("CREATE INDEX IF NOT EXISTS idx_messages_session_id ON messages(session_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_messages_timestamp ON messages(timestamp)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_sessions_last_activity ON sessions(last_activity)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_summaries_session_id ON summaries(session_id)")

            conn.commit()

    def _validate_inputs(self, **kwargs):
        """Validate input parameters."""
        for key, value in kwargs.items():
            if key == 'user_message' and (not value or not value.strip()):
                raise ValueError("User message cannot be empty")
            if key == 'session_id' and not value:
                raise ValueError("Session ID is required")

    def _count_tokens(self, text: str) -> int:
        """
        Improved token counting with caching.
        Replace with actual tokenizer for production use.
        """
        if not hasattr(self, '_token_cache'):
            self._token_cache = {}
        
        text_hash = hash(text)
        if text_hash in self._token_cache:
            return self._token_cache[text_hash]
        
        # Simple approximation - replace with tiktoken or similar
        token_count = len(text.split())
        self._token_cache[text_hash] = token_count
        
        # Limit cache size
        if len(self._token_cache) > 1000:
            # Remove oldest half of entries
            items = list(self._token_cache.items())
            self._token_cache = dict(items[500:])
        
        return token_count

    def start_session(self, condition: Optional[str] = None, model_used: str = "gemini-pro") -> str:
        """Start a new chatbot session."""
        session_id = f"ID-{uuid.uuid4().hex[:8].upper()}"
        now = datetime.now().isoformat()
        
        try:
            with self._get_connection() as conn:
                conn.execute("""
                    INSERT INTO sessions (session_id, timestamp, last_activity, condition, model_used)
                    VALUES (?, ?, ?, ?, ?)
                """, (session_id, now, now, condition, model_used))
                conn.commit()
            
            logger.info(f"Started new session: {session_id}")
            return session_id
            
        except Exception as e:
            logger.error(f"Failed to start session: {e}")
            raise

    def handle_message(self, session_id: str, user_message: str, health_status: Optional[str] = None) -> str:
        """Handle user message and generate bot response."""
        self._validate_inputs(session_id=session_id, user_message=user_message)
        
        with self.lock:
            try:
                session = self.get_session(session_id)
                if not session:
                    raise ValueError(f"Session {session_id} not found")

                # Get chat history
                chat_history = self._get_chat_history(session_id)
                summarized_histories = self._get_chat_summaries(session_id)
                print(f"Chat history for session {session_id}: {chat_history}")
                print(f"Summarized histories for session {session_id}: {summarized_histories}")

                # Generate response
                start_time = time.time()
                bot_reply = self.chatbot.get_chatbot_response(
                    chat_history,
                    user_message=user_message,
                    health_status=health_status or session.get("condition"),
                    summarized_histories=summarized_histories
                )
                latency_ms = int((time.time() - start_time) * 1000)

                # Store messages in a transaction
                with self._get_connection() as conn:
                    # Store user message
                    self._add_message_to_conn(conn, session_id, "user", user_message)
                    # Store bot reply
                    self._add_message_to_conn(conn, session_id, "assistant", bot_reply, latency_ms)
                    
                    # Update session stats
                    conn.execute("""
                        UPDATE sessions
                        SET total_messages = total_messages + 2,
                            last_activity = ?
                        WHERE session_id = ?
                    """, (datetime.now().isoformat(), session_id))
                    
                    conn.commit()

                logger.info(f"Processed message for session {session_id} in {latency_ms}ms")
                return bot_reply

            except Exception as e:
                logger.error(f"Error handling message for session {session_id}: {e}")
                raise

    def _get_chat_history(self, session_id: str, limit: Optional[int] = None) -> List[Dict[str, str]]:
        """Get chat history for a session."""
        with self._get_connection() as conn:
            query = "SELECT role, content FROM messages WHERE session_id=? ORDER BY id"
            params = [session_id]
            
            if limit:
                query += " LIMIT ?"
                params.append(limit)
                
            return [
                {"role": row["role"], "content": row["content"]}
                for row in conn.execute(query, params)
            ]
    
    def _get_chat_summaries(self, session_id: str, limit: Optional[int] = None) -> List[Dict[str, str]]:
        """Get chat summaries for a session."""
        with self._get_connection() as conn:
            query = "SELECT timestamp, summary FROM summaries WHERE session_id=? ORDER BY id"
            params = [session_id]

            if limit:
                query += " LIMIT ?"
                params.append(limit)

            return [
                {"timestamp": row["timestamp"], "summary": row["summary"]}
                for row in conn.execute(query, params)
            ]

    def _add_message_to_conn(self, conn, session_id: str, role: str, content: str, latency_ms: Optional[int] = None):
        """Add message using existing connection."""
        tokens_count = self._count_tokens(content)
        tokens_in = tokens_count if role == "user" else 0
        tokens_out = tokens_count if role == "assistant" else 0
        
        message_id = f"MSG-{uuid.uuid4().hex[:8].upper()}"
        now = datetime.now().isoformat()

        conn.execute("""
            INSERT INTO messages (
                session_id, message_id, timestamp, role, content,
                tokens_input, tokens_output, latency_ms
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (session_id, message_id, now, role, content, tokens_in, tokens_out, latency_ms))

    def summarize_session(self, session_id: str) -> str:
        """Summarize session chat history."""
        self._validate_inputs(session_id=session_id)
        
        with self.lock:
            try:
                chat_history = self._get_chat_history(session_id)
                if not chat_history:
                    raise ValueError("No chat history found for session")

                summary = self.chatbot.summarize_history(chat_history)

                with self._get_connection() as conn:
                    conn.execute("""
                        UPDATE sessions
                        SET total_summaries = total_summaries + 1,
                            last_activity = ?
                        WHERE session_id = ?
                    """, (datetime.now().isoformat(), session_id))
                    conn.execute(
                        "INSERT INTO summaries (session_id, timestamp, summary) VALUES (?, ?, ?)",
                        (session_id, datetime.now().isoformat(), summary)
                    )
                    conn.commit()

                logger.info(f"Summarized session {session_id}")
                return summary

            except Exception as e:
                logger.error(f"Error summarizing session {session_id}: {e}")
                raise

    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get session information."""
        with self._get_connection() as conn:
            row = conn.execute("SELECT * FROM sessions WHERE session_id=?", (session_id,)).fetchone()
            return dict(row) if row else None

    def list_sessions(self, active_only: bool = True, limit: int = 50) -> List[Dict[str, Any]]:
        """List sessions with optional filtering."""
        with self._get_connection() as conn:
            query = "SELECT * FROM sessions"
            params = []
            
            if active_only:
                query += " WHERE is_active = 1"
            
            query += " ORDER BY last_activity DESC LIMIT ?"
            params.append(limit)
            print([dict(row) for row in conn.execute(query, params)])
            return [dict(row) for row in conn.execute(query, params)]

    def close_session(self, session_id: str):
        """Mark session as inactive."""
        with self._get_connection() as conn:
            conn.execute("""
                UPDATE sessions 
                SET is_active = 0, last_activity = ? 
                WHERE session_id = ?
            """, (datetime.now().isoformat(), session_id))
            conn.commit()
        
        logger.info(f"Closed session {session_id}")

    def delete_session(self, session_id: str):
        """Delete session and all associated messages."""
        with self._get_connection() as conn:
            conn.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))
            conn.commit()
        
        logger.info(f"Deleted session {session_id}")

    def get_session_stats(self, session_id: str) -> Dict[str, Any]:
        """Get detailed session statistics."""
        with self._get_connection() as conn:
            session = conn.execute("SELECT * FROM sessions WHERE session_id=?", (session_id,)).fetchone()
            if not session:
                raise ValueError("Session not found")
            
            stats = conn.execute("""
                SELECT 
                    COUNT(*) as message_count,
                    SUM(tokens_input) as total_tokens_in,
                    SUM(tokens_output) as total_tokens_out,
                    AVG(latency_ms) as avg_latency,
                    MIN(timestamp) as first_message,
                    MAX(timestamp) as last_message
                FROM messages WHERE session_id=?
            """, (session_id,)).fetchone()
            
            return {
                **dict(session),
                "stats": dict(stats) if stats else {}
            }

    def export_session_json(self, session_id: str) -> str:
        """Export full session data as JSON."""
        with self._get_connection() as conn:
            session_row = conn.execute(
                "SELECT * FROM sessions WHERE session_id = ?",
                (session_id,)
            ).fetchone()

            messages = [
                dict(row) for row in conn.execute(
                    "SELECT * FROM messages WHERE session_id = ? ORDER BY id",
                    (session_id,)
                )
            ]

            summaries = [
                dict(row) for row in conn.execute(
                    "SELECT * FROM summaries WHERE session_id = ? ORDER BY id",
                    (session_id,)
                )
            ]

        data = {
            "session_info": dict(session_row),
            "summaries": summaries,
            "chat_history": messages,
            "export_metadata": {
                "exported_at": datetime.now().isoformat(),
                "total_messages": len(messages),
                "total_summaries": len(summaries),
                "export_version": "2.0"
            }
        }

        # Ensure folder exists
        export_folder = Path(EXPORT_FOLDER)
        export_folder.mkdir(exist_ok=True)

        # Create file path (session_id.json)
        export_path = export_folder / f"{session_id}.json"

        # Save to file
        with open(export_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        return str(export_path)  # Return the path to the saved file

    def cleanup_old_sessions(self, days_old: int = 30):
        """Clean up sessions older than specified days."""
        cutoff_date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        cutoff_date = datetime.now() - timedelta(days=days_old)
        
        with self._get_connection() as conn:
            result = conn.execute("""
                DELETE FROM sessions 
                WHERE datetime(last_activity) < datetime(?)
            """, (cutoff_date.isoformat(),))
            
            deleted_count = result.rowcount
            conn.commit()
        
        logger.info(f"Cleaned up {deleted_count} old sessions")
        return deleted_count