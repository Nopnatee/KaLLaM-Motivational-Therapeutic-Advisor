import json
import sqlite3
import uuid
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict, List, Any
from contextlib import contextmanager
from dataclasses import dataclass
import time
import threading

# Import your prompt.py functions (maintained for compatibility)
from agents.chatbot_prompt import KaLLaMChatbot
from core.orchestrator import Orchestrator
from agents.supervisor import SupervisorAgent

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

EXPORT_FOLDER = "exported_sessions"

@dataclass
class SessionStats:
    """Data class for session statistics."""
    message_count: int = 0
    total_tokens_in: int = 0
    total_tokens_out: int = 0
    avg_latency: float = 0.0
    first_message: Optional[str] = None
    last_message: Optional[str] = None

class ChatbotManager:
    """
    ChatbotManager handles chatbot sessions, message storage, summaries,
    and exports using a SQLite database with improved performance and reliability.
    """
    
    def __init__(self, 
                 db_path: str = "chatbot_data.db", 
                 summarize_every_n_messages: int = 10,
                 message_limit: int = 20):
        """Initialize the ChatbotManager with validation."""
        if summarize_every_n_messages <= 0:
            raise ValueError("summarize_every_n_messages must be positive")
        if message_limit <= 0:
            raise ValueError("message_limit must be positive")
            
        self.orchestrator = Orchestrator()
        self.sum_every_n = summarize_every_n_messages
        self.message_limit = message_limit
        self.db_path = Path(db_path)
        self.lock = threading.RLock()
        self._token_cache = {}
        
        self._create_tables()
        logger.info(f"ChatbotManager initialized with database: {self.db_path}")

    @contextmanager
    def _get_connection(self):
        """Context manager for database connections with proper error handling."""
        conn = None
        try:
            conn = sqlite3.connect(self.db_path, timeout=30.0)
            conn.execute("PRAGMA foreign_keys = ON")
            conn.execute("PRAGMA journal_mode = WAL")  # Better concurrency
            conn.row_factory = sqlite3.Row
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
        """Create database tables with optimized schema."""
        with self._get_connection() as conn:
            # Sessions table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    timestamp TEXT NOT NULL,
                    last_activity TEXT NOT NULL,
                    saved_memories TEXT, -- JSON string
                    total_messages INTEGER DEFAULT 0,
                    total_user_messages INTEGER DEFAULT 0,
                    total_assistant_messages INTEGER DEFAULT 0,
                    total_summaries INTEGER DEFAULT 0,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    is_active BOOLEAN DEFAULT 1
                )
            """)
            
            # Messages table with better constraints
            conn.execute("""
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    message_id TEXT UNIQUE NOT NULL,
                    timestamp TEXT NOT NULL,
                    role TEXT NOT NULL CHECK(role IN ('user', 'assistant', 'system')),
                    content TEXT NOT NULL,
                    translated_content TEXT,
                    chain_of_thoughts TEXT, -- JSON string
                    tokens_input INTEGER DEFAULT 0,
                    tokens_output INTEGER DEFAULT 0,
                    latency_ms INTEGER,
                    flags TEXT, -- JSON string
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY(session_id) REFERENCES sessions(session_id) ON DELETE CASCADE
                )
            """)
            
            # Summaries table
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
            
            # Optimized indexes
            indexes = [
                "CREATE INDEX IF NOT EXISTS idx_messages_session_id ON messages(session_id)",
                "CREATE INDEX IF NOT EXISTS idx_messages_timestamp ON messages(timestamp DESC)",
                "CREATE INDEX IF NOT EXISTS idx_sessions_last_activity ON sessions(last_activity DESC)",
                "CREATE INDEX IF NOT EXISTS idx_summaries_session_id ON summaries(session_id)",
                "CREATE INDEX IF NOT EXISTS idx_messages_role ON messages(role)"
            ]
            
            for idx in indexes:
                conn.execute(idx)
            
            conn.commit()

    def _validate_inputs(self, **kwargs):
        """Validate input parameters with specific error messages."""
        validators = {
            'user_message': lambda x: bool(x and x.strip()),
            'session_id': lambda x: bool(x),
            'role': lambda x: x in ('user', 'assistant', 'system')
        }
        
        for key, value in kwargs.items():
            validator = validators.get(key)
            if validator and not validator(value):
                raise ValueError(f"Invalid {key}: {value}")

    def _count_tokens(self, text: str) -> int:
        """Improved token counting with LRU-like cache management."""
        text_hash = hash(text)
        
        if text_hash in self._token_cache:
            return self._token_cache[text_hash]
        
        # Simple approximation - replace with proper tokenizer
        token_count = max(1, len(text.split()))  # Minimum 1 token
        
        # Manage cache size
        if len(self._token_cache) >= 1000:
            # Remove oldest 50% of entries (simple LRU approximation)
            items = list(self._token_cache.items())
            self._token_cache = dict(items[500:])
        
        self._token_cache[text_hash] = token_count
        return token_count

    def start_session(self, saved_memories: Optional[str] = None) -> str:
        """Start a new chatbot session with better error handling."""
        session_id = f"ID-{uuid.uuid4().hex[:8].upper()}"
        now = datetime.now().isoformat()
        
        try:
            with self._get_connection() as conn:
                conn.execute("""
                    INSERT INTO sessions (session_id, timestamp, last_activity, saved_memories)
                    VALUES (?, ?, ?, ?)
                """, (session_id, now, now, saved_memories))
                conn.commit()
            
            logger.info(f"Started new session: {session_id}")
            return session_id
            
        except sqlite3.IntegrityError as e:
            logger.error(f"Session ID conflict: {e}")
            # Retry with new ID
            return self.start_session(saved_memories)
        except Exception as e:
            logger.error(f"Failed to start session: {e}")
            raise

    def handle_message(self, session_id: str, user_message: str, 
                      health_status: Optional[str] = None) -> str:
        """
        Handle user message and generate bot response.
        Maintains compatibility with existing orchestrator interface.

        Args:
            session_id (str): ID of the session.
            user_message (str): Message text from the user.
            health_status (Optional[str]): Optional current health status for context.

        Returns:
            str: The chatbot's response.
        """
        self._validate_inputs(session_id=session_id, user_message=user_message)
        
        with self.lock:
            start_time = time.time()
            
            try:
                # Verify session exists
                if not self.get_session(session_id):
                    raise ValueError(f"Session {session_id} not found")
                
                # Get English chat history
                eng_chat_history = self._get_eng_chat_history(session_id)
                eng_summaries = self._get_eng_chat_summaries(session_id)
                chain_of_thoughts = self._get_chain_of_thoughts(session_id)
                
                logger.debug(f"Fetched {len(eng_chat_history)} messages for session {session_id}")
                logger.debug(f"Fetched {len(eng_summaries)} summaries for session {session_id}")
                logger.debug(f"Fetched {len(chain_of_thoughts)} chain of thoughts for session {session_id}")
                
                # Get activation flags
                dict_flags = self._get_flags_dict(session_id, user_message)
                translate_flag = dict_flags.get("translate")  # Preserve original flag extraction pattern
                
                # Translate the input message if needed
                eng_message = self.orchestrator.get_translation(
                    message=user_message, 
                    flags=dict_flags,
                    translation_type="forward"
                )
                
                # Generate response
                dict_response = self.orchestrator.get_response(
                    chat_history=eng_chat_history,
                    user_message=eng_message,
                    flags=dict_flags,
                    chain_of_thoughts=chain_of_thoughts,
                    summarized_histories=eng_summaries
                )
                
                bot_eng = dict_response["final_output"]
                
                # Translate back to original language if needed
                bot_reply = self.orchestrator.get_translation(
                    message=bot_eng, 
                    flags=dict_flags,
                    translation_type="backward"
                )
                
                latency_ms = int((time.time() - start_time) * 1000)
                
                # Store messages in transaction
                with self._get_connection() as conn:
                    # Store user message
                    self._add_message_to_conn(
                        conn=conn,
                        session_id=session_id, 
                        role="user", 
                        content=user_message,
                        translated_content=eng_message,
                        flags=json.dumps(dict_flags)
                    )
                    
                    # Store bot reply
                    self._add_message_to_conn(
                        conn=conn,
                        session_id=session_id, 
                        role="assistant", 
                        content=bot_reply,
                        translated_content=bot_eng,
                        chain_of_thoughts=json.dumps(dict_response),
                        latency_ms=latency_ms
                    )
                    conn.commit()
                
                logger.info(f"Processed message for {session_id} in {latency_ms}ms")
                
                # Re-fetch updated session counts after storing messages and check for summarization (original logic)
                updated_session = self.get_session(session_id)
                if updated_session and updated_session["total_user_messages"] % self.sum_every_n == 0:
                    self.summarize_session(session_id)
                
                return bot_reply
                
            except Exception as e:
                logger.error(f"Error handling message for session {session_id}: {e}")
                raise

    def _get_flags_dict(self, session_id: str, user_message: str) -> Dict[str, Any]:
        """
        Get activation flags from supervisor agent.
        
        Returns:
            Dict[str, Any]: Dictionary with flags as keys and values as activation signals, e.g.,
                {
                    "translate": "thai",   # force translation
                    "summarize": True,
                    "doctor": False,
                    "psychologist": True,
                    "to_core_memory": True
                }
        """
        self._validate_inputs(session_id=session_id)
        try:
            dict_flags = self.orchestrator.get_flags_from_supervisor(user_message=user_message)
            return dict_flags
        except Exception as e:
            logger.warning(f"Failed to get flags from supervisor: {e}, using defaults")
            return {"translate": False}

    def _get_eng_chat_history(self, session_id: str) -> List[Dict[str, str]]:
        """Get English chat history with better query."""
        with self._get_connection() as conn:
            query = """
                SELECT role, COALESCE(translated_content, content) as content
                FROM messages
                WHERE session_id = ? AND role IN ('user', 'assistant')
                ORDER BY id DESC
                LIMIT ?
            """
            
            rows = conn.execute(query, (session_id, self.message_limit)).fetchall()
            # Reverse to get chronological order
            return [{"role": row["role"], "content": row["content"]} 
                   for row in reversed(rows)]

    def _get_eng_chat_summaries(self, session_id: str, 
                               limit: Optional[int] = None) -> List[Dict[str, str]]:
        """Get chat summaries efficiently."""
        with self._get_connection() as conn:
            query = """
                SELECT timestamp, summary 
                FROM summaries 
                WHERE session_id = ? 
                ORDER BY id DESC
            """
            params = [session_id]
            
            if limit:
                query += " LIMIT ?"
                params.append(limit)
            
            return [{"timestamp": row["timestamp"], "summary": row["summary"]}
                   for row in conn.execute(query, params)]

    def _get_chain_of_thoughts(self, session_id: str) -> List[Dict[str, Any]]:
        """Get chain of thoughts with JSON validation."""
        with self._get_connection() as conn:
            query = """
                SELECT message_id, chain_of_thoughts
                FROM messages
                WHERE session_id = ? AND chain_of_thoughts IS NOT NULL
                ORDER BY id DESC
                LIMIT 10
            """
            
            thoughts = []
            for row in conn.execute(query, (session_id,)):
                try:
                    thoughts.append({
                        "message_id": row["message_id"],
                        "contents": json.loads(row["chain_of_thoughts"])
                    })
                except json.JSONDecodeError as e:
                    logger.warning(f"Invalid JSON in chain_of_thoughts for {row['message_id']}: {e}")
            
            return thoughts

    def _add_message_to_conn(self, conn, session_id: str, role: str, content: str,
                           translated_content: Optional[str] = None,
                           chain_of_thoughts: Optional[str] = None,
                           latency_ms: Optional[int] = None,
                           flags: Optional[str] = None):
        """
        Add message using existing database connection (maintains original interface).
        
        Args:
            conn: Active SQLite connection.
            session_id (str): Session ID to which the message belongs.
            role (str): Role of the message sender ("user", "assistant", "system").
            content (str): Message content.
            translated_content (Optional[str]): Translated version of content.
            chain_of_thoughts (Optional[str]): JSON string of internal reasoning.
            latency_ms (Optional[int]): Optional latency in milliseconds for the assistant response.
            flags (Optional[str]): JSON string of activation flags.
        """
        self._validate_inputs(role=role)
        
        tokens_count = self._count_tokens(content)
        tokens_in = tokens_count if role == "user" else 0
        tokens_out = tokens_count if role == "assistant" else 0
        
        message_id = f"MSG-{uuid.uuid4().hex[:8].upper()}"
        now = datetime.now().isoformat()
        
        # Insert message
        conn.execute("""
            INSERT INTO messages (
                session_id, message_id, timestamp, role, content, 
                translated_content, chain_of_thoughts, tokens_input, 
                tokens_output, latency_ms, flags
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (session_id, message_id, now, role, content, translated_content,
              chain_of_thoughts, tokens_in, tokens_out, latency_ms, flags))
        
        # Update session counters (maintains original logic)
        if role == "user":
            conn.execute("""
                UPDATE sessions
                SET total_messages = total_messages + 1,
                    total_user_messages = COALESCE(total_user_messages, 0) + 1,
                    last_activity = ?
                WHERE session_id = ?
            """, (now, session_id))
        elif role == "assistant":
            conn.execute("""
                UPDATE sessions
                SET total_messages = total_messages + 1,
                    total_assistant_messages = COALESCE(total_assistant_messages, 0) + 1,
                    last_activity = ?
                WHERE session_id = ?
            """, (now, session_id))
        else:
            conn.execute("""
                UPDATE sessions
                SET total_messages = total_messages + 1,
                    last_activity = ?
                WHERE session_id = ?
            """, (now, session_id))

    def summarize_session(self, session_id: str) -> str:
        """Summarize session with better error handling."""
        with self.lock:
            try:
                eng_chat_history = self._get_eng_chat_history(session_id)
                if not eng_chat_history:
                    raise ValueError("No chat history found for session")
                
                eng_summaries = self._get_eng_chat_summaries(session_id)
                summary = self.orchestrator.summarize_history(
                    chat_history=eng_chat_history, 
                    eng_summaries=eng_summaries
                )
                
                now = datetime.now().isoformat()
                with self._get_connection() as conn:
                    conn.execute("""
                        INSERT INTO summaries (session_id, timestamp, summary) 
                        VALUES (?, ?, ?)
                    """, (session_id, now, summary))
                    
                    conn.execute("""
                        UPDATE sessions
                        SET total_summaries = total_summaries + 1,
                            last_activity = ?
                        WHERE session_id = ?
                    """, (now, session_id))
                    
                    conn.commit()
                
                logger.info(f"Summarized session {session_id}")
                return summary
                
            except Exception as e:
                logger.error(f"Error summarizing session {session_id}: {e}")
                raise

    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get session information with caching consideration."""
        try:
            with self._get_connection() as conn:
                row = conn.execute(
                    "SELECT * FROM sessions WHERE session_id = ?", 
                    (session_id,)
                ).fetchone()
                return dict(row) if row else None
        except Exception as e:
            logger.error(f"Error getting session {session_id}: {e}")
            return None

    def list_sessions(self, active_only: bool = True, limit: int = 50) -> List[Dict[str, Any]]:
        """List sessions with improved query."""
        try:
            with self._get_connection() as conn:
                query = """
                    SELECT session_id, timestamp, last_activity, total_messages,
                           total_user_messages, total_assistant_messages, 
                           total_summaries, is_active
                    FROM sessions
                """
                params = []
                
                if active_only:
                    query += " WHERE is_active = 1"
                
                query += " ORDER BY last_activity DESC LIMIT ?"
                params.append(limit)
                
                return [dict(row) for row in conn.execute(query, params)]
        except Exception as e:
            logger.error(f"Error listing sessions: {e}")
            return []

    def close_session(self, session_id: str) -> bool:
        """Mark session as inactive with return status."""
        try:
            with self._get_connection() as conn:
                cursor = conn.execute("""
                    UPDATE sessions 
                    SET is_active = 0, last_activity = ? 
                    WHERE session_id = ?
                """, (datetime.now().isoformat(), session_id))
                conn.commit()
                
                if cursor.rowcount > 0:
                    logger.info(f"Closed session {session_id}")
                    return True
                else:
                    logger.warning(f"Session {session_id} not found for closing")
                    return False
        except Exception as e:
            logger.error(f"Error closing session {session_id}: {e}")
            return False

    def delete_session(self, session_id: str) -> bool:
        """Delete session with return status."""
        try:
            with self._get_connection() as conn:
                cursor = conn.execute(
                    "DELETE FROM sessions WHERE session_id = ?", 
                    (session_id,)
                )
                conn.commit()
                
                if cursor.rowcount > 0:
                    logger.info(f"Deleted session {session_id}")
                    return True
                else:
                    logger.warning(f"Session {session_id} not found for deletion")
                    return False
        except Exception as e:
            logger.error(f"Error deleting session {session_id}: {e}")
            return False

    def get_session_stats(self, session_id: str) -> Dict[str, Any]:
        """Get session statistics with better structure."""
        try:
            with self._get_connection() as conn:
                session = conn.execute(
                    "SELECT * FROM sessions WHERE session_id = ?", 
                    (session_id,)
                ).fetchone()
                
                if not session:
                    raise ValueError("Session not found")
                
                stats_row = conn.execute("""
                    SELECT 
                        COUNT(*) as message_count,
                        COALESCE(SUM(tokens_input), 0) as total_tokens_in,
                        COALESCE(SUM(tokens_output), 0) as total_tokens_out,
                        COALESCE(AVG(latency_ms), 0) as avg_latency,
                        MIN(timestamp) as first_message,
                        MAX(timestamp) as last_message
                    FROM messages WHERE session_id = ?
                """, (session_id,)).fetchone()
                
                return {
                    "session_info": dict(session),
                    "stats": SessionStats(
                        message_count=stats_row["message_count"] or 0,
                        total_tokens_in=stats_row["total_tokens_in"] or 0,
                        total_tokens_out=stats_row["total_tokens_out"] or 0,
                        avg_latency=float(stats_row["avg_latency"] or 0),
                        first_message=stats_row["first_message"],
                        last_message=stats_row["last_message"]
                    )
                }
        except Exception as e:
            logger.error(f"Error getting session stats for {session_id}: {e}")
            raise

    def export_session_json(self, session_id: str) -> str:
        """Export session data with improved error handling."""
        try:
            with self._get_connection() as conn:
                # Get all data in one transaction
                session_row = conn.execute(
                    "SELECT * FROM sessions WHERE session_id = ?", 
                    (session_id,)
                ).fetchone()
                
                if not session_row:
                    raise ValueError(f"Session {session_id} does not exist")
                
                messages = [dict(row) for row in conn.execute(
                    "SELECT * FROM messages WHERE session_id = ? ORDER BY id", 
                    (session_id,)
                )]
                
                summaries = [dict(row) for row in conn.execute(
                    "SELECT * FROM summaries WHERE session_id = ? ORDER BY id", 
                    (session_id,)
                )]
            
            # Prepare export data
            data = {
                "session_info": dict(session_row),
                "summaries": summaries,
                "chat_history": messages,
                "export_metadata": {
                    "exported_at": datetime.now().isoformat(),
                    "total_messages": len(messages),
                    "total_summaries": len(summaries),
                    "export_version": "2.1"
                }
            }
            
            # Write to file
            export_folder = Path(EXPORT_FOLDER)
            export_folder.mkdir(parents=True, exist_ok=True)
            export_path = export_folder / f"{session_id}.json"
            
            with open(export_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2, default=str)
            
            logger.info(f"Session exported successfully: {export_path}")
            return str(export_path)
            
        except Exception as e:
            logger.error(f"Error exporting session {session_id}: {e}")
            raise

    def cleanup_old_sessions(self, days_old: int = 30) -> int:
        """Clean up old sessions with better date handling."""
        if days_old <= 0:
            raise ValueError("days_old must be positive")
            
        try:
            cutoff_date = (datetime.now() - timedelta(days=days_old)).isoformat()
            
            with self._get_connection() as conn:
                cursor = conn.execute("""
                    DELETE FROM sessions 
                    WHERE last_activity < ?
                """, (cutoff_date,))
                
                deleted_count = cursor.rowcount
                conn.commit()
            
            logger.info(f"Cleaned up {deleted_count} sessions older than {days_old} days")
            return deleted_count
            
        except Exception as e:
            logger.error(f"Error cleaning up old sessions: {e}")
            raise