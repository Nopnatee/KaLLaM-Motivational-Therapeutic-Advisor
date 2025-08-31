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
from agents.chatbot_prompt import KaLLaMChatbot
from core.orchestrator import Orchestrator
from agents.supervisor import SupervisorAgent

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

EXPORT_FOLDER = "exported_sessions"

class ChatbotManager:
    """
    ChatbotManager handles chatbot sessions, message storage, summaries,
    and exports using a SQLite database.

    Attributes:
        chatbot (KaLLaMChatbot): The chatbot instance to generate responses.
        summarize_every_n_messages (int): How often to summarize chat history.
        db_path (Path): Path to the SQLite database file.
        lock (RLock): Thread-safe lock for concurrent access.
    """
    def __init__(self, 
                 db_path: str = "chatbot_data.db", 
                 summarize_every_n_messages: Optional[int] = 10,
                 message_limit: Optional[int] = 20,
                 ):
        """
        Initialize the ChatbotManager.

        Args:
            db_path (str): Path to the SQLite database file. Defaults to "chatbot_data.db".
            api_provider (Optional[str]): API provider for KaLLaMChatbot. Defaults to "sea_lion".
            summarize_every_n_messages (Optional[int]): How many messages before summarization. Defaults to 10.
        """
        self.orchestrator = Orchestrator()
        self.sum_every_n = summarize_every_n_messages
        self.message_limit = message_limit
        self.db_path = Path(db_path)
        self.lock = threading.RLock()
        self._create_tables()
        logger.info(f"ChatbotManager initialized with database: {self.db_path}")

    @contextmanager
    def _get_connection(self):
        """
        Context manager for database connections with proper error handling.

        Yields:
            sqlite3.Connection: An active SQLite connection.
        """
        try:
            conn = sqlite3.connect(self.db_path)
            conn.execute("PRAGMA foreign_keys = ON;")  # Enforce cascade deletes
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
        """
        Create necessary database tables (sessions, messages, summaries)
        and indexes if they do not exist.
        """
        with self._get_connection() as conn:
            conn.execute(f"""
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    timestamp TEXT NOT NULL,
                    last_activity TEXT NOT NULL,
                    saved_memories TEXT, -- json based memory
                    total_messages INTEGER DEFAULT 0,
                    total_user_messages INTEGER DEFAULT 0,
                    total_assistant_messages INTEGER DEFAULT 0,
                    total_summaries INTEGER DEFAULT 0,
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
                    translated_content TEXT,
                    chain_of_thoughts TEXT, -- json based internal reasoning
                    tokens_input INTEGER DEFAULT 0,
                    tokens_output INTEGER DEFAULT 0,
                    latency_ms INTEGER,
                    flags TEXT, -- json activation flags
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
        """
        Validate input parameters for various methods.

        Args:
            **kwargs: Arbitrary keyword arguments, e.g., session_id, user_message.

        Raises:
            ValueError: If required arguments are missing or empty.
        """
        for key, value in kwargs.items():
            if key == 'user_message' and (not value or not value.strip()):
                raise ValueError("User message cannot be empty")
            if key == 'session_id' and not value:
                raise ValueError("Session ID is required")

    def _count_tokens(self, text: str, cache_limit: Optional[int] = 1000) -> int:
        """
        Improved token counting with caching.
        Replace with actual tokenizer for production use.

        Args:
            text (str): The text to count tokens for.
            cache_limit (Optional[int]): Maximum number of items to store in the cache. Defaults to 1000.

        Returns:
            int: Number of tokens estimated in the text.
        """
        if not hasattr(self, '_token_cache'):
            self._token_cache = {}
        
        text_hash = hash(text)
        if text_hash in self._token_cache:
            return self._token_cache[text_hash]
        
        # Simple approximation - should replace with tiktoken or similar
        token_count = len(text.split())
        self._token_cache[text_hash] = token_count
        
        # Limit cache size
        if len(self._token_cache) > cache_limit:
            # Remove oldest half of entries
            items = list(self._token_cache.items())
            half_cache_size = cache_limit // 2
            self._token_cache = dict(items[half_cache_size:])
        
        return token_count

    def start_session(self, saved_memories: Optional[str] = None) -> str:
        """
        Start a new chatbot session.

        Args:
            saved_memories (Optional[str]): Optional health saved_memories or context for the session.

        Returns:
            str: The newly created session ID.
        """
        session_id = f"ID-{uuid.uuid4().hex[:8].upper()}"
        now = datetime.now().isoformat()
        
        try:
            with self._get_connection() as conn:
                conn.execute(f"""
                    INSERT INTO sessions (session_id, timestamp, last_activity, saved_memories)
                    VALUES (?, ?, ?, ?)
                """, (session_id, now, now, saved_memories))
                conn.commit()
            
            logger.info(f"Started new session: {session_id}")
            return session_id
            
        except Exception as e:
            logger.error(f"Failed to start session: {e}")
            raise

    def handle_message(self, session_id: str, user_message: str, health_status: Optional[str] = None) -> str: # Need translated_content
        """
        Handle user message and generate bot response.

        Args:
            session_id (str): ID of the session.
            user_message (str): Message text from the user.
            health_status (Optional[str]): Optional current health status for context.

        Returns:
            str: The chatbot's response.
        """
        self._validate_inputs(session_id=session_id, user_message=user_message)
        
        with self.lock:
            try:
                session = self.get_session(session_id)
                if not session:
                    raise ValueError(f"Session {session_id} not found")
                
                # Start timer for latency measurement
                start_time = time.time()

                # Get eng chat history
                eng_chat_history = self._get_eng_chat_history(session_id)
                eng_summarized_histories = self._get_eng_chat_summaries(session_id)
                chain_of_thoughts = self._get_chain_of_thoughts(session_id)
                logger.debug(f"Fetched {len(eng_chat_history)} messages for session {session_id}")
                logger.debug(f"Fetched {len(eng_summarized_histories)} summaries for session {session_id}")
                logger.debug(f"Fetched {len(chain_of_thoughts)} chain of thoughts for session {session_id}")

                # Get activation flags
                dict_flags = self._get_flags_dict(session_id, user_message)
                translate_flag = dict_flags.get("translate")

                # Translate the input message if needed
                eng_message = self.orchestrator.get_translation(message=user_message, 
                                                                flags=dict_flags,
                                                                type="forward")

                # Generate response
                dict_response = self.orchestrator.get_response(
                    chat_history=eng_chat_history,
                    user_message=eng_message,
                    flags=dict_flags,
                    chain_of_thoughts=chain_of_thoughts,
                    summarized_histories=eng_summarized_histories
                )

                bot_eng = dict_response["final_output"]

                # Translate back to original language if needed
                bot_reply = self.orchestrator.get_translation(message=bot_eng, 
                                                              flags=dict_flags,
                                                              type="backward")

                # Measure latency
                latency_ms = int((time.time() - start_time) * 1000)

                # Store messages in a transaction
                with self._get_connection() as conn:
                    # Store user message
                    self._add_message_to_conn(conn=conn, 
                                              session_id=session_id, 
                                              role="user", 
                                              content=user_message,
                                              translated_content=eng_message,
                                              flags=json.dumps(dict_flags))

                    # Store bot reply
                    self._add_message_to_conn(conn=conn, 
                                              session_id=session_id, 
                                              role="assistant", 
                                              content=bot_reply,
                                              translated_content=bot_eng,
                                              chain_of_thoughts=json.dumps(dict_response),
                                              latency_ms=latency_ms,)
                    
                    conn.commit()

                logger.info("Processed message", extra={"session_id": session_id, "latency_ms": latency_ms})

                # Re-fetch updated session counts after storing messages and check for summarization
                updated_session = self.get_session(session_id)
                if updated_session["total_user_messages"] % self.sum_every_n == 0:
                    self.summarize_session(session_id)

                return bot_reply

            except Exception as e:
                logger.error(f"Error handling message for session {session_id}: {e}")
                raise
    
    def _get_flags_dict(self, session_id: str, user_message: str) -> Dict[str, bool]: # Need dict return
        """
        Returns:
            Dict[str, bool]: Dictionary with flags as keys and True as values.
            flags (dict): Activation signals, e.g.,
                {
                    "translate": "thai",   # force translation
                    "summarize": True,
                    "doctor": False,
                    "psychologist": True
                    "to core memory": True
                }
        """
        self._validate_inputs(session_id=session_id)
        dict_flags = self.orchestrator.get_flags_from_supervisor(user_message=user_message)

        return dict_flags

    def _get_eng_chat_history(self, session_id: str) -> List[Dict[str, str]]:
        """
        Get the English version of chat history for a session.

        Args:
            session_id (str): ID of the session.
            limit (Optional[int]): Maximum number of messages to fetch. Defaults to None (all).

        Returns:
            List[Dict[str, str]]: List of messages with role and English content.
        """
        with self._get_connection() as conn:
            query = """
                SELECT role, translated_content AS content
                FROM messages
                WHERE session_id=?
                ORDER BY id
            """
            params = [session_id]

            # Check for message limit
            limit = self.message_limit
            query += " LIMIT ?"
            params.append(limit)

            history = [{"role": row["role"], "content": row["content"]}
                    for row in conn.execute(query, params)]

            return history
        
    def _get_original_chat_history(self, session_id: str) -> List[Dict[str, str]]:
        """
        Get the original chat history for a session.

        Args:
            session_id (str): ID of the session.

        Returns:
            List[Dict[str, str]]: List of messages with role and original content.
        """
        with self._get_connection() as conn:
            query = """
                SELECT role, content
                FROM messages
                WHERE session_id=?
                ORDER BY id
            """
            params = [session_id]

            history = [{"role": row["role"], "content": row["content"]}
                    for row in conn.execute(query, params)]

            return history
        
    def _get_eng_chat_summaries(self, session_id: str, limit: Optional[int] = None) -> List[Dict[str, str]]:
        """
        Get chat summaries for a session.

        Args:
            session_id (str): ID of the session.
            limit (Optional[int]): Maximum number of summaries to fetch. Defaults to None (all).

        Returns:
            List[Dict[str, str]]: List of summaries with timestamp and summary text.
        """
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
        
    def _get_chain_of_thoughts(self, session_id: str) -> List[Dict[str, Any]]:
        """
        Get chain of thoughts for a session.

        Args:
            session_id (str): ID of the session.

        Returns:
            List[Dict[str, Any]]: List of chain of thoughts with message_id and details.
        """
        with self._get_connection() as conn:
            query = """
                SELECT message_id, chain_of_thoughts
                FROM messages
                WHERE session_id=? AND chain_of_thoughts IS NOT NULL
                ORDER BY id
            """
            params = [session_id]

            thoughts = []
            for row in conn.execute(query, params):
                try:
                    thoughts.append({
                        "message_id": row["message_id"], # e.g., MSG-XXXXXXXX
                        "contents": json.loads(row["chain_of_thoughts"]) # as Dict
                    })
                except json.JSONDecodeError:
                    logger.warning(f"Invalid JSON in chain_of_thoughts for message {row['message_id']}")
                    continue

            return thoughts

    def _add_message_to_conn(self, 
                             conn, 
                             session_id: str, 
                             role: str, 
                             content: str,
                             translated_content: Optional[str] = None, 
                             latency_ms: Optional[int] = None, 
                             flags: Dict[str, Optional[bool]] = None):
        
        """
        Add message using existing database connection.

        Args:
            conn: Active SQLite connection.
            session_id (str): Session ID to which the message belongs.
            role (str): Role of the message sender ("user", "assistant", "system").
            content (str): Message content.
            latency_ms (Optional[int]): Optional latency in milliseconds for the assistant response.
        """
        tokens_count = self._count_tokens(content)
        tokens_in = tokens_count if role == "user" else 0
        tokens_out = tokens_count if role == "assistant" else 0
        
        message_id = f"MSG-{uuid.uuid4().hex[:8].upper()}"
        now = datetime.now().isoformat()

        conn.execute("""
            INSERT INTO messages (
                session_id, message_id, timestamp, role, content, translated_content,
                tokens_input, tokens_output, latency_ms, flags
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (session_id, message_id, now, role, content, translated_content, tokens_in, tokens_out, latency_ms, flags))

        # Additionally store messages count in session
        if role == "user":
            conn.execute("""
                            UPDATE sessions
                            SET total_messages = total_messages + 1,
                                total_user_messages = COALESCE(total_user_messages, 0) + 1,
                                last_activity = ?
                            WHERE session_id = ?
                        """, (datetime.now().isoformat(), session_id))
        elif role == "assistant":
            conn.execute("""
                            UPDATE sessions
                            SET total_messages = total_messages + 1,
                                total_assistant_messages = COALESCE(total_assistant_messages, 0) + 1,
                                last_activity = ?
                            WHERE session_id = ?
                        """, (datetime.now().isoformat(), session_id))
        else:
            conn.execute("""
                UPDATE sessions
                SET total_messages = total_messages + 1,
                    last_activity = ?
                WHERE session_id = ?
            """, (datetime.now().isoformat(), session_id))

    def summarize_session(self, session_id: str) -> str:
        """
        Summarize session chat history.

        Args:
            session_id (str): ID of the session to summarize.

        Returns:
            str: Generated summary text.
        """
        self._validate_inputs(session_id=session_id)

        with self.lock:
            try:
                eng_response_history = self._get_eng_chat_history(session_id)
                if not eng_response_history:
                    raise ValueError("No chat history found for session")
                summarized_histories = self._get_eng_chat_summaries(session_id)
                if not eng_response_history:
                    raise ValueError("No messages found in session history")
                logger.debug(f"Fetched {len(eng_response_history)} messages for session {session_id}")
                logger.debug(f"Fetched {len(summarized_histories)} summaries for session {session_id}")


                summary = self.orchestrator.summarize_history(response_history=eng_response_history, summarized_histories=summarized_histories)

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
                logger.exception(f"Error summarizing session {session_id}: {e}")
                raise

    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        Get session information.

        Args:
            session_id (str): ID of the session.

        Returns:
            Optional[Dict[str, Any]]: Session data or None if not found.
        """
        with self._get_connection() as conn:
            row = conn.execute("SELECT * FROM sessions WHERE session_id=?", (session_id,)).fetchone()
            return dict(row) if row else None

    def list_sessions(self, active_only: bool = True, limit: int = 50) -> List[Dict[str, Any]]:
        """
        List sessions with optional filtering.

        Args:
            active_only (bool): Whether to return only active sessions. Defaults to True.
            limit (int): Maximum number of sessions to return. Defaults to 50.

        Returns:
            List[Dict[str, Any]]: List of session dictionaries.
        """
        with self._get_connection() as conn:
            query = "SELECT * FROM sessions"
            params = []
            
            if active_only:
                query += " WHERE is_active = 1"
            
            query += " ORDER BY last_activity DESC LIMIT ?"
            params.append(limit)
            rows = [dict(row) for row in conn.execute(query, params)]
            logger.debug(rows)
            return rows

    def close_session(self, session_id: str):
        """
        Mark session as inactive.

        Args:
            session_id (str): ID of the session to close.
        """
        with self._get_connection() as conn:
            conn.execute("""
                UPDATE sessions 
                SET is_active = 0, last_activity = ? 
                WHERE session_id = ?
            """, (datetime.now().isoformat(), session_id))
            conn.commit()
        
        logger.info(f"Closed session {session_id}")

    def delete_session(self, session_id: str):
        """
        Delete session and all associated messages.

        Args:
            session_id (str): ID of the session to delete.
        """
        with self._get_connection() as conn:
            conn.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))
            conn.commit()
        
        logger.info(f"Deleted session {session_id}")

    def get_session_stats(self, session_id: str) -> Dict[str, Any]:
        """
        Get detailed session statistics.

        Args:
            session_id (str): ID of the session.

        Returns:
            Dict[str, Any]: Session data with statistics.
        """
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
        """
        Export full session data as JSON with error handling and debug logs.

        Args:
            session_id (str): ID of the session to export.

        Returns:
            str: Path to the exported JSON file.
        """
        try:
            with self._get_connection() as conn:
                session_row = conn.execute(
                    "SELECT * FROM sessions WHERE session_id = ?",
                    (session_id,)
                ).fetchone()

                if not session_row:
                    logger.debug(f"No session found with ID: {session_id}")
                    raise ValueError(f"Session {session_id} does not exist")

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
            export_folder.mkdir(parents=True, exist_ok=True)

            # Create file path (session_id.json)
            export_path = export_folder / f"{session_id}.json"

            try:
                with open(export_path, "w", encoding="utf-8") as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
                logger.info(f"Session exported successfully: {export_path}")
            except Exception as e:
                logger.exception(f"Failed to write session JSON file: {e}")
                raise

            return str(export_path)

        except Exception as e:
            logger.exception(f"Error exporting session {session_id}: {e}")
            raise

    def cleanup_old_sessions(self, days_old: int = 30):
        """
        Clean up sessions older than specified days.

        Args:
            days_old (int): Number of days; sessions older than this will be deleted. Defaults to 30.

        Returns:
            int: Number of sessions deleted.
        """
        cutoff_date = (datetime.now() - timedelta(days=days_old)).replace(hour=0, minute=0, second=0, microsecond=0)
        
        with self._get_connection() as conn:
            result = conn.execute("""
                DELETE FROM sessions 
                WHERE datetime(last_activity) < datetime(?)
            """, (cutoff_date.isoformat(),))
            
            deleted_count = result.rowcount
            conn.commit()
        
        logger.info(f"Cleaned up {deleted_count} old sessions")
        return deleted_count