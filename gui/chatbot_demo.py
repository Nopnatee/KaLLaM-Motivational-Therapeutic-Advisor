# enhanced_kallam_app.py
import gradio as gr
import logging
from datetime import datetime
from typing import List, Tuple, Optional
import os
import socket

from kallam.app.chatbot_manager import ChatbotManager
from kallam.infra.db import sqlite_conn  # use the shared helper

# -----------------------
# Init
# -----------------------
chatbot_manager = ChatbotManager(log_level="DEBUG")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# INLINE SVG for icons
CABBAGE_SVG = """
<svg width="64" height="64" viewBox="0 0 64 64" xmlns="http://www.w3.org/2000/svg"
     role="img" aria-label="cabbage">
  <defs>
    <linearGradient id="leaf" x1="0" x2="0" y1="0" y2="1">
      <stop offset="0%" stop-color="#9be58b"/>
      <stop offset="100%" stop-color="#5cc46a"/>
    </linearGradient>
  </defs>
  <g fill="none">
    <circle cx="32" cy="32" r="26" fill="url(#leaf)"/>
    <path d="M12,34 C18,28 22,26 28,27 C34,28 38,32 44,30 C48,29 52,26 56,22"
          stroke="#2e7d32" stroke-width="3" stroke-linecap="round"/>
    <path d="M10,40 C18,36 22,42 28,42 C34,42 38,38 44,40 C50,42 54,40 58,36"
          stroke="#2e7d32" stroke-width="3" stroke-linecap="round"/>
    <path d="M24 24 C28 20 36 20 40 24 C44 28 44 36 32 38 C20 36 20 28 24 24 Z"
          fill="#bff5b6"/>
  </g>
</svg>
"""

class AppState:
    def __init__(self):
        self.current_session_id: str = ""
        self.message_count: int = 0
        self.followup_note: str = "Request follow-up analysis..."

    def reset(self):
        self.current_session_id = ""
        self.message_count = 0

app_state = AppState()

# -----------------------
# Helpers
# -----------------------
def _safe_latency_str(v) -> str:
    try:
        return f"{float(v):.1f}"
    except Exception:
        return "0.0"

def _extract_stats_pack() -> tuple[dict, dict]:
    # returns (session_info, stats_dict)
    data = chatbot_manager.get_session_stats(app_state.current_session_id)  # new shape
    session_info = data.get("session_info", {}) if isinstance(data, dict) else {}
    stats = data.get("stats", {}) if isinstance(data, dict) else {}
    return session_info, stats

def get_current_session_status() -> str:
    if not app_state.current_session_id:
        return "🔴 **ไม่มี Session ที่ใช้งาน** - กรุณาสร้าง Session ใหม่"

    try:
        session_info, stats = _extract_stats_pack()

        avg_latency = _safe_latency_str(stats.get("avg_latency"))
        saved_memories = session_info.get("saved_memories") or "ไม่ได้ระบุ"
        return f"""
### 🏥 **สถานะสุขภาพ:** {saved_memories}  
🟢 **Session ปัจจุบัน:** `{app_state.current_session_id}`
📅 **สร้างเมื่อ:** {session_info.get('timestamp', 'N/A')}
💬 **จำนวนข้อความ:** {stats.get('message_count', 0) or 0} ข้อความ
📋 **จำนวนสรุป:** {session_info.get('total_summaries', 0) or 0} ครั้ง
⚡ **Latency เฉลี่ย:** {avg_latency} ms
🔧 **Model:** {session_info.get('model_used', 'N/A')}
        """.strip()
    except Exception as e:
        logger.error(f"Error getting session status: {e}")
        return f"❌ **Error:** ไม่สามารถโหลดข้อมูล Session {app_state.current_session_id}"

def get_session_list() -> List[str]:
    try:
        sessions = chatbot_manager.list_sessions(active_only=True, limit=50)
        opts = []
        for s in sessions:
            saved = (s.get("saved_memories") or "ไม่ระบุ")[:20]
            msgs = s.get("total_messages", 0)
            sums = s.get("total_summaries", 0)
            opts.append(f"{s['session_id']} | {msgs}💬 {sums}📋 | {saved}")
        return opts or ["ไม่มี Session"]
    except Exception as e:
        logger.error(f"Error getting session list: {e}")
        return ["Error loading sessions"]

def extract_session_id(dropdown_value: str) -> Optional[str]:
    if not dropdown_value or dropdown_value in ["ไม่มี Session", "Error loading sessions"]:
        return None
    return dropdown_value.split(" | ")[0]

def refresh_session_list():
    sessions = get_session_list()
    return gr.update(choices=sessions, value=sessions[0] if sessions else None)

def create_new_session(saved_memories: str = "") -> Tuple[List, str, str, str, str]:
    try:
        sid = chatbot_manager.start_session(saved_memories=saved_memories or None)
        app_state.current_session_id = sid
        app_state.message_count = 0
        status = get_current_session_status()
        result = f"✅ **สร้าง Session ใหม่สำเร็จ!**\n\n🆔 Session ID: `{sid}`"
        return [], "", result, status, saved_memories
    except Exception as e:
        logger.error(f"Error creating new session: {e}")
        return [], "", f"❌ **ไม่สามารถสร้าง Session ใหม่:** {e}", get_current_session_status(), ""

def switch_session(dropdown_value: str) -> Tuple[List, str, str, str, str]:
    sid = extract_session_id(dropdown_value)
    if not sid:
        return [], "", "❌ **Session ID ไม่ถูกต้อง**", get_current_session_status(), ""

    try:
        session = chatbot_manager.get_session(sid)
        if not session:
            return [], "", f"❌ **ไม่พบ Session:** {sid}", get_current_session_status(), ""
        app_state.current_session_id = sid
        app_state.message_count = session.get("total_messages", 0)

        # use the new helper on manager (provided)
        chat_history = chatbot_manager.get_original_chat_history(sid)
        gr_history = []
        for m in chat_history:
            role = m.get("role")
            content = m.get("content", "")
            if role == "user":
                gr_history.append({"role": "user", "content": content})
            elif role == "assistant":
                gr_history.append({"role": "assistant", "content": content})

        status = get_current_session_status()
        result = f"✅ **เปลี่ยน Session สำเร็จ!**\n\n🆔 Session ID: `{sid}`"
        saved_memories = session.get("saved_memories", "")
        return gr_history, "", result, status, saved_memories
    except Exception as e:
        logger.error(f"Error switching session: {e}")
        return [], "", f"❌ **ไม่สามารถเปลี่ยน Session:** {e}", get_current_session_status(), ""

def get_session_info() -> str:
    if not app_state.current_session_id:
        return "❌ **ไม่มี Session ที่ใช้งาน**"
    try:
        session_info, stats = _extract_stats_pack()

        latency_str = f"{float(stats.get('avg_latency') or 0):.2f}"
        total_tokens_in = stats.get("total_tokens_in") or 0
        total_tokens_out = stats.get("total_tokens_out") or 0
        saved_memories = session_info.get("saved_memories") or "ไม่ได้ระบุ"
        summarized_history = session_info.get("summarized_history") or "ยังไม่มีการสรุป"

        return f"""
## 📊 ข้อมูลรายละเอียด Session: `{app_state.current_session_id}`

### 🔧 ข้อมูลพื้นฐาน
- **Session ID:** `{session_info.get('session_id', 'N/A')}`
- **สร้างเมื่อ:** {session_info.get('timestamp', 'N/A')}
- **ใช้งานล่าสุด:** {session_info.get('last_activity', 'N/A')}
- **Model:** {session_info.get('model_used', 'N/A')}
- **สถานะ:** {'🟢 Active' if session_info.get('is_active') else '🔴 Inactive'}

### 🏥 ข้อมูลสถานะสุขภาพ
- **สถานะสุขภาพ:** {saved_memories}

### 📈 สถิติการใช้งาน
- **จำนวนข้อความทั้งหมด:** {stats.get('message_count', 0) or 0} ข้อความ
- **จำนวนสรุป:** {session_info.get('total_summaries', 0) or 0} ครั้ง
- **Token Input รวม:** {total_tokens_in:,} tokens
- **Token Output รวม:** {total_tokens_out:,} tokens
- **Latency เฉลี่ย:** {latency_str} ms
- **ข้อความแรก:** {stats.get('first_message', 'N/A') or 'N/A'}
- **ข้อความล่าสุด:** {stats.get('last_message', 'N/A') or 'N/A'}

### 📋 ประวัติการสรุป
{summarized_history}
        """.strip()
    except Exception as e:
        logger.error(f"Error getting session info: {e}")
        return f"❌ **Error:** {e}"

def get_all_sessions_info() -> str:
    try:
        sessions = chatbot_manager.list_sessions(active_only=False, limit=20)
        if not sessions:
            return "📭 **ไม่มี Session ในระบบ**"

        parts = ["# 📁 ข้อมูล Session ทั้งหมด\n"]
        for i, s in enumerate(sessions, 1):
            status_icon = "🟢" if s.get("is_active") else "🔴"
            saved = (s.get("saved_memories") or "ไม่ระบุ")[:30]
            parts.append(f"""
## {i}. {status_icon} `{s['session_id']}`
- **สร้าง:** {s.get('timestamp', 'N/A')}
- **ใช้งานล่าสุด:** {s.get('last_activity', 'N/A')}
- **ข้อความ:** {s.get('total_messages', 0)} | **สรุป:** {s.get('total_summaries', 0)}
- **สภาวะ:** {saved}
- **Model:** {s.get('model_used', 'N/A')}
            """.strip())
        return "\n\n".join(parts)
    except Exception as e:
        logger.error(f"Error getting all sessions info: {e}")
        return f"❌ **Error:** {e}"

def update_medical_saved_memories(saved_memories: str) -> Tuple[str, str]:
    if not app_state.current_session_id:
        return get_current_session_status(), "❌ **ไม่มี Session ที่ใช้งาน**"
    if not saved_memories.strip():
        return get_current_session_status(), "❌ **กรุณาระบุสถานะสุขภาพ**"

    try:
        with sqlite_conn(str(chatbot_manager.db_path)) as conn:
            conn.execute(
                "UPDATE sessions SET saved_memories = ?, last_activity = ? WHERE session_id = ?",
                (saved_memories.strip(), datetime.now().isoformat(), app_state.current_session_id),
            )
        status = get_current_session_status()
        result = f"✅ **อัปเดตสถานะสุขภาพสำเร็จ!**\n\n📝 **ข้อมูลใหม่:** {saved_memories.strip()}"
        return status, result
    except Exception as e:
        logger.error(f"Error updating saved_memories: {e}")
        return get_current_session_status(), f"❌ **ไม่สามารถอัปเดตสถานะสุขภาพ:** {e}"

def process_chat_message(user_message: str, history: List) -> Tuple[List, str]:
    if not app_state.current_session_id:
        history.append({"role": "assistant", "content": "❌ **กรุณาสร้าง Session ใหม่ก่อนใช้งาน**"})
        return history, ""
    if not user_message.strip():
        return history, ""

    try:
        history.append({"role": "user", "content": user_message})
        bot = chatbot_manager.handle_message(
            session_id=app_state.current_session_id,
            user_message=user_message,
        )
        history.append({"role": "assistant", "content": bot})
        app_state.message_count += 2
        return history, ""
    except Exception as e:
        logger.error(f"Error processing chat message: {e}")
        history.append({"role": "assistant", "content": f"❌ **ข้อผิดพลาด:** {e}"})
        return history, ""

def generate_followup(history: List) -> List:
    # No dedicated handle_followup in new manager.
    # We just inject the follow-up note as a plain user turn.
    if not app_state.current_session_id:
        history.append({"role": "assistant", "content": "❌ **กรุณาสร้าง Session ใหม่ก่อนใช้งาน**"})
        return history
    try:
        note = app_state.followup_note
        history.append({"role": "user", "content": note})
        bot = chatbot_manager.handle_message(
            session_id=app_state.current_session_id,
            user_message=note,
        )
        history.append({"role": "assistant", "content": bot})
        app_state.message_count += 2
        return history
    except Exception as e:
        logger.error(f"Error generating follow-up: {e}")
        history.append({"role": "assistant", "content": f"❌ **ไม่สามารถสร้างการวิเคราะห์:** {e}"})
        return history



def clear_session() -> Tuple[List, str, str, str, str]:
    if not app_state.current_session_id:
        return [], "", "❌ **ไม่มี Session ที่ใช้งาน**", get_current_session_status(), ""
    try:
        old = app_state.current_session_id
        chatbot_manager.delete_session(old)
        app_state.reset()
        return [], "", f"✅ **ลบ Session สำเร็จ!**\n\n🗑️ **Session ที่ลบ:** `{old}`", get_current_session_status(), ""
    except Exception as e:
        logger.error(f"Error clearing session: {e}")
        return [], "", f"❌ **ไม่สามารถลบ Session:** {e}", get_current_session_status(), ""

def clear_all_summaries() -> str:
    if not app_state.current_session_id:
        return "❌ **ไม่มี Session ที่ใช้งาน**"
    try:
        with sqlite_conn(str(chatbot_manager.db_path)) as conn:
            conn.execute("DELETE FROM summaries WHERE session_id = ?", (app_state.current_session_id,))
        return f"✅ **ล้างสรุปสำเร็จ!**\n\n🗑️ **Session:** `{app_state.current_session_id}`"
    except Exception as e:
        logger.error(f"Error clearing summaries: {e}")
        return f"❌ **ไม่สามารถล้างสรุป:** {e}"

def export_session() -> str:
    if not app_state.current_session_id:
        return "❌ **ไม่มี Session ที่ใช้งาน**"
    try:
        chatbot_manager.export_session_json(app_state.current_session_id)
        return "✅ **ส่งออกข้อมูลสำเร็จ!**"
    except Exception as e:
        logger.error(f"Error exporting session: {e}")
        return f"❌ **ไม่สามารถส่งออกข้อมูล:** {e}"
    
def export_all_sessions() -> str:
    try:
        chatbot_manager.export_all_sessions_json()
        return "✅ **ส่งออกข้อมูลสำเร็จ!**"
    except Exception as e:
        logger.error(f"Error exporting session: {e}")
        return f"❌ **ไม่สามารถส่งออกข้อมูล:** {e}"

# UI Helper Functions
def show_buttons():
    return gr.update(visible=True)

def hide_buttons():
    return gr.update(visible=False)

def clear_all_buttons():
    return [
        gr.update(visible=False),  # chatbot_window
        gr.update(visible=False),  # session_management
        gr.update(visible=False)   # summary_page
    ]

def set_button_loading(text="⏳ ประมวลผล..."):
    return gr.update(value=text, variant="stop")

def reset_button(text, variant):
    return gr.update(value=text, variant=variant)

# -----------------------
# UI
# -----------------------
def create_app() -> gr.Blocks:
    custom_css = """
    :root {
        --kallam-primary: #659435;
        --kallam-secondary: #5ea0bd;
        --kallam-accent: #b8aa54;
        --kallam-light: #f8fdf5;
        --kallam-dark: #2d3748;
        --shadow-soft: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        --shadow-medium: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05);
        --border-radius: 12px;
        --transition: all 0.3s ease;
    }

    .gradio-container {
        max-width: 100% !important;
        width: 100% !important;
        margin: 0 auto !important;
        background: linear-gradient(135deg, #f8fdf5 0%, #ffffff 100%);
        min-height: 100vh;
    }

    /* Header styling */
    .kallam-header {
        background: linear-gradient(135deg, var(--kallam-secondary) 0%, var(--kallam-primary) 50%, var(--kallam-accent) 100%);
        border-radius: var(--border-radius);
        padding: 2rem;
        margin-bottom: 1.5rem;
        text-align: center;
        box-shadow: var(--shadow-medium);
        position: relative;
        overflow: hidden;
    }

    .kallam-header::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: url("data:image/svg+xml,%3Csvg width='60' height='60' viewBox='0 0 60 60' xmlns='http://www.w3.org/2000/svg'%3E%3Cg fill='none' fill-rule='evenodd'%3E%3Cg fill='%23ffffff' fill-opacity='0.05'%3E%3Ccircle cx='30' cy='30' r='2'/%3E%3C/g%3E%3C/g%3E%3C/svg%3E") repeat;
        pointer-events: none;
    }

    .kallam-header h1 {
        color: white !important;
        font-size: 2.5rem !important;
        font-weight: 700 !important;
        margin: 0 !important;
        text-shadow: 0 2px 4px rgba(0,0,0,0.2);
        position: relative;
        z-index: 1;
    }

    .kallam-subtitle {
        color: rgba(255,255,255,0.9) !important;
        font-size: 1.1rem !important;
        margin-top: 0.5rem !important;
        position: relative;
        z-index: 1;
    }

    /* Tab navigation */
    .tab-nav {
        background: linear-gradient(135deg, var(--kallam-secondary) 0%, var(--kallam-primary) 50%, var(--kallam-accent) 100%) !important;
        border-radius: var(--border-radius) var(--border-radius) 0 0 !important;
        padding: 0.5rem !important;
        box-shadow: var(--shadow-soft) !important;
    }

    .tab-nav button {
        background: rgba(255,255,255,0.1) !important;
        color: white !important;
        border: 1px solid rgba(255,255,255,0.2) !important;
        border-radius: 8px !important;
        margin: 0 0.25rem !important;
        padding: 0.75rem 1.5rem !important;
        font-weight: 600 !important;
        transition: var(--transition) !important;
        backdrop-filter: blur(10px) !important;
    }

    .tab-nav button:hover {
        background: rgba(255,255,255,0.2) !important;
        transform: translateY(-1px) !important;
        box-shadow: var(--shadow-soft) !important;
    }

    .tab-nav button.selected {
        background: white !important;
        color: var(--kallam-primary) !important;
        box-shadow: var(--shadow-soft) !important;
    }

    /* Session info card */
    .session-info {
        background: linear-gradient(135deg, var(--kallam-secondary) 0%, var(--kallam-primary) 100%) !important;
        border-radius: var(--border-radius) !important;
        padding: 1.5rem !important;
        margin: 1rem 0 !important;
        color: white !important;
        font-weight: 500 !important;
        box-shadow: var(--shadow-medium) !important;
        border: none !important;
        position: relative !important;
        overflow: hidden !important;
    }

    .session-info::before {
        content: '' !important;
        position: absolute !important;
        top: 0 !important;
        right: 0 !important;
        width: 100px !important;
        height: 100px !important;
        background: rgba(255,255,255,0.1) !important;
        border-radius: 50% !important;
        transform: translate(30px, -30px) !important;
    }

    .session-info h2, .session-info h3 {
        color: white !important;
        margin-bottom: 1rem !important;
        position: relative !important;
        z-index: 1 !important;
    }

    /* Chat container */
    .chat-container {
        background: white !important;
        border-radius: var(--border-radius) !important;
        border: 1px solid rgba(101, 148, 53, 0.1) !important;
        box-shadow: var(--shadow-medium) !important;
        overflow: hidden !important;
    }

    .chat-container .message {
        padding: 1rem !important;
        border-radius: 8px !important;
        margin: 0.5rem !important;
        transition: var(--transition) !important;
    }

    .chat-container .message.user {
        background: linear-gradient(135deg, var(--kallam-light) 0%, #f0f9ff 100%) !important;
        border-left: 4px solid var(--kallam-primary) !important;
    }

    .chat-container .message.bot {
        background: linear-gradient(135deg, #fff 0%, #fafafa 100%) !important;
        border-left: 4px solid var(--kallam-secondary) !important;
    }

    /* Sidebar styling */
    .gradio-column.gradio-sidebar {
        background: rgba(255,255,255,0.8) !important;
        backdrop-filter: blur(10px) !important;
        border-radius: var(--border-radius) !important;
        border: 1px solid rgba(101, 148, 53, 0.1) !important;
        box-shadow: var(--shadow-soft) !important;
        padding: 1.5rem !important;
    }

    /* Button styling */
    .btn {
        border-radius: 8px !important;
        font-weight: 600 !important;
        padding: 0.75rem 1.5rem !important;
        transition: var(--transition) !important;
        border: none !important;
        box-shadow: var(--shadow-soft) !important;
        cursor: pointer !important;
    }

    .btn:hover {
        transform: translateY(-2px) !important;
        box-shadow: var(--shadow-medium) !important;
    }

    .btn.btn-primary {
        background: linear-gradient(135deg, var(--kallam-primary) 0%, var(--kallam-secondary) 100%) !important;
        color: white !important;
    }

    .btn.btn-secondary {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%) !important;
        color: var(--kallam-dark) !important;
        border: 1px solid rgba(101, 148, 53, 0.2) !important;
    }

    .btn.btn-secondary:hover {
        background: linear-gradient(135deg, var(--kallam-light) 0%, #f8f9fa 100%) !important;
        border-color: var(--kallam-primary) !important;
    }

    /* Form elements */
    .gradio-textbox, .gradio-dropdown {
        border-radius: 8px !important;
        border: 2px solid rgba(101, 148, 53, 0.1) !important;
        transition: var(--transition) !important;
        background: white !important;
    }

    .gradio-textbox:focus, .gradio-dropdown:focus {
        border-color: var(--kallam-primary) !important;
        box-shadow: 0 0 0 3px rgba(101, 148, 53, 0.1) !important;
    }

    /* Summary box */
    .summary-box {
        background: white !important;
        border-radius: var(--border-radius) !important;
        padding: 1.5rem !important;
        margin: 1rem 0 !important;
        border: 1px solid rgba(101, 148, 53, 0.1) !important;
        box-shadow: var(--shadow-soft) !important;
        position: relative !important;
        overflow: hidden !important;
    }

    .summary-box::before {
        content: '' !important;
        position: absolute !important;
        top: 0 !important;
        left: 0 !important;
        width: 4px !important;
        height: 100% !important;
        background: linear-gradient(135deg, var(--kallam-primary) 0%, var(--kallam-secondary) 100%) !important;
    }

    /* Health profile box */
    .saved_memories-box textarea {
        background: linear-gradient(135deg, var(--kallam-light) 0%, #ffffff 100%) !important;
        border: 2px solid rgba(101, 148, 53, 0.1) !important;
        border-radius: 8px !important;
        transition: var(--transition) !important;
    }

    .saved_memories-box textarea:focus {
        border-color: var(--kallam-primary) !important;
        box-shadow: 0 0 0 3px rgba(101, 148, 53, 0.1) !important;
    }

    /* Loading animation */
    .loading {
        position: relative !important;
        overflow: hidden !important;
    }

    .loading::after {
        content: '' !important;
        position: absolute !important;
        top: 0 !important;
        left: -100% !important;
        width: 100% !important;
        height: 100% !important;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.2), transparent) !important;
        animation: loading-shine 2s infinite !important;
    }

    @keyframes loading-shine {
        0% { left: -100%; }
        100% { left: 100%; }
    }

    /* Responsive design */
    @media (max-width: 768px) {
        .kallam-header {
            padding: 1.5rem !important;
        }
        
        .kallam-header h1 {
            font-size: 2rem !important;
        }
        
        .gradio-column.gradio-sidebar {
            margin-bottom: 1rem !important;
        }
    }

    /* Smooth scrolling */
    html {
        scroll-behavior: smooth !important;
    }

    /* Custom scrollbar */
    ::-webkit-scrollbar {
        width: 8px !important;
    }

    ::-webkit-scrollbar-track {
        background: #f1f1f1 !important;
        border-radius: 4px !important;
    }

    ::-webkit-scrollbar-thumb {
        background: var(--kallam-primary) !important;
        border-radius: 4px !important;
    }

    ::-webkit-scrollbar-thumb:hover {
        background: var(--kallam-secondary) !important;
    }
    """

    with gr.Blocks(
        title="🥬 KaLLaM - Thai Motivational Therapeutic Advisor",
        theme=gr.themes.Soft(primary_hue="green", secondary_hue="blue", neutral_hue="slate"), # type: ignore
        css=custom_css,
    ) as app:
        # Beautiful header section
        gr.HTML(f"""      
            <div class="kallam-header">
                <h1>{CABBAGE_SVG} KaLLaM</h1>
                <p class="kallam-subtitle">Thai Motivational Therapeutic Advisor - ระบบให้คำปรึกษาสุขภาพอัจฉริยะ</p>
            </div>
            """)
        
        # Welcome message with better styling
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("""
                🌟 **Welcome to KaLLaM:** Example sessions can be accessed via Session Management in the sidebar, then select any available session.
                """, elem_classes=["welcome-message"])
        
        with gr.Tab("TH Ver."):
            # Session Status Display
            with gr.Column(elem_classes=["session-info"]):
                gr.Markdown(value="## ข้อมูล Session")
                with gr.Row():
                    with gr.Column():
                        session_status = gr.Markdown(value=get_current_session_status())
                        
            with gr.Sidebar():
                with gr.Column():
                    gr.HTML("""
                    <div style="text-align: center; padding: 0.5rem 0 1rem 0;">
                        <h3 style="color: #659435; margin: 0; font-size: 1.2rem;">🎛️ การควบคุม</h3>
                        <p style="color: #666; margin: 0.25rem 0 0 0; font-size: 0.9rem;">จัดการ Session และข้อมูลสุขภาพ</p>
                    </div>
                    """)
                    
                    with gr.Group():
                        new_session_btn = gr.Button(
                            "➕ Session ใหม่", 
                            variant="primary", 
                            size="lg",
                            elem_classes=["btn", "btn-primary"]
                        )
                        manage_session_btn = gr.Button(
                            "🗂️ จัดการ Session", 
                            variant="secondary",
                            elem_classes=["btn", "btn-secondary"]
                        )
                        edit_profile_btn = gr.Button(
                            "✏️ แก้ไขข้อมูลสุขภาพ", 
                            variant="secondary",
                            elem_classes=["btn", "btn-secondary"]
                        )
                    
                    gr.HTML('<div style="margin: 1rem 0;"><hr style="border: none; border-top: 1px solid #e0e0e0;"></div>')
                    
                    gr.HTML("""
                    <div style="text-align: center; padding: 0.5rem 0;">
                        <h4 style="color: #659435; margin: 0; font-size: 1rem;">📊 สถานะปัจจุบัน</h4>
                    </div>
                    """)
                    
                # Session Details with improved styling
                session_result = gr.Markdown(
                    value="**กำลังรอการอัปเดต...**", 
                    elem_classes=["summary-box"],
                )
                
            with gr.Column(visible=False) as summary_page:
                back_btn_2 = gr.Button("⏪ กลับไปยังแชท", variant="primary")
                summary_result = gr.Markdown(
                    value="**กำลังรอคำสั่งสรุปแชท...**", 
                    elem_classes=["summary-box"],
                )
                
            with gr.Row(visible=False) as session_management:
                with gr.Column(scale=3):
                    gr.Markdown("### 🗂️ การจัดการ Session")
                    session_ids = get_session_list()
                    initial_session = session_ids[0] if session_ids else None

                    session_dropdown = gr.Dropdown(
                        choices=session_ids,
                        value=initial_session,
                        label="🗒️ เลือก Session",
                        info="เลือก session ที่ต้องการเปลี่ยนไป",
                    )
                    with gr.Column():
                        with gr.Row():
                            switch_btn = gr.Button("🔀 โหลด Session", variant="secondary")
                            refresh_btn = gr.Button("🔄 รีเฟรช", variant="primary")
                        with gr.Row():
                            clear_chat_btn = gr.Button("🗑️ ล้าง Session", variant="secondary")
                        close_management_btn = gr.Button("❌ ปิดการจัดการ Session", variant="primary")

            # Health Management Section
            with gr.Column(visible=False) as health_management:
                health_context = gr.Textbox(
                    label="🏥 ข้อมูลสุขภาพของผู้ป่วย",
                    placeholder="เช่น: ชื่อผู้ป่วย, อายุ, ความดันโลหิตสูง, เบาหวาน, ปัญหาการนอน, ระดับความเครียด...",
                    value="",
                    max_lines=5,
                    lines=3,
                    info="ข้อมูลนี้จะถูกเก็บใน session และใช้ปรับแต่งคำแนะนำ",
                    elem_classes=["saved_memories-box"]
                )
                update_saved_memories_btn = gr.Button("💾 อัปเดตข้อมูลสุขภาพ", variant="primary")
                back_btn_1 = gr.Button("⏪ กลับไปยังแชท", variant="primary")

            with gr.Column() as chatbot_window:
                # Chat Interface Section with beautiful design
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.HTML("""
                        <div style="text-align: center; padding: 1rem 0;">
                            <h2 style="color: #659435; margin: 0; font-size: 1.5rem;">💬 แชทบอทให้คำปรึกษาสุขภาพ</h2>
                            <p style="color: #666; margin: 0.5rem 0 0 0;">พูดคุยกับ KaLLaM เพื่อรับคำปรึกษาด้านสุขภาพ</p>
                        </div>
                        """)

                chatbot = gr.Chatbot(
                    label="💭 พูดคุยกับ KaLLaM",
                    height=400,
                    show_label=False,
                    type="messages",
                    elem_classes=["chat-container"],
                    value=[{"role": "assistant", "content": "สวัสดีค่ะ 😊 ฉันชื่อกะหล่ำ 🌿 เป็นคุณหมอที่จะคอยดูแลสุขภาพกายและใจของคุณนะคะ 💖 วันนี้รู้สึกยังไงบ้างคะ? 🌸"}],
                )

                with gr.Row():
                    with gr.Column(scale=5):
                        msg = gr.Textbox(
                            label="💬 พิมพ์ข้อความของคุณ",
                            placeholder="พิมพ์คำถามหรือข้อมูลสุขภาพที่ต้องการปรึกษา...",
                            lines=2,
                            max_lines=4,
                            show_label=False,
                        )
                    with gr.Column(scale=1, min_width=120):
                        send_btn = gr.Button(
                            "📤 ส่งข้อความ", 
                            variant="primary", 
                            size="lg",
                            elem_classes=["btn", "btn-primary"]
                        )

        with gr.Tab("ENG Ver."):
            # Session Status Display
            with gr.Column(elem_classes=["session-info"]):
                gr.Markdown(value="## User Profile")
                with gr.Row():
                    with gr.Column():
                        session_status_en = gr.Markdown(value=get_current_session_status())
                        
            with gr.Sidebar():
                with gr.Column():
                    gr.HTML("""
                    <div style="text-align: center; padding: 0.5rem 0 1rem 0;">
                        <h3 style="color: #659435; margin: 0; font-size: 1.2rem;">🎛️ Controls</h3>
                        <p style="color: #666; margin: 0.25rem 0 0 0; font-size: 0.9rem;">Manage sessions and health profile</p>
                    </div>
                    """)
                    
                    with gr.Group():
                        new_session_btn_en = gr.Button(
                            "➕ New Session", 
                            variant="primary", 
                            size="lg",
                            elem_classes=["btn", "btn-primary"]
                        )
                        manage_session_btn_en = gr.Button(
                            "🗂️ Manage Session", 
                            variant="secondary",
                            elem_classes=["btn", "btn-secondary"]
                        )
                        edit_profile_btn_en = gr.Button(
                            "✏️ Edit Health Profile", 
                            variant="secondary",
                            elem_classes=["btn", "btn-secondary"]
                        )
                    
                    gr.HTML('<div style="margin: 1rem 0;"><hr style="border: none; border-top: 1px solid #e0e0e0;"></div>')
                    
                    gr.HTML("""
                    <div style="text-align: center; padding: 0.5rem 0;">
                        <h4 style="color: #659435; margin: 0; font-size: 1rem;">📊 Current Status</h4>
                    </div>
                    """)
                    
                # Session Details with improved styling
                session_result_en = gr.Markdown(
                    value="**Waiting for update...**", 
                    elem_classes=["summary-box"],
                )
                
            with gr.Column(visible=False) as summary_page_en:
                back_btn_2_en = gr.Button("⏪ Back to Chat", variant="primary")
                summary_result_en = gr.Markdown(
                    value="**Waiting for summary command...**", 
                    elem_classes=["summary-box"],
                )
                
            with gr.Row(visible=False) as session_management_en:
                with gr.Column(scale=3):
                    gr.Markdown("### 🗂️ Session Management")
                    session_ids_en = get_session_list()
                    initial_session_en = session_ids_en[0] if session_ids_en else None

                    session_dropdown_en = gr.Dropdown(
                        choices=session_ids_en,
                        value=initial_session_en,
                        label="🗒️ Select Session",
                        info="Select the session you want to switch to",
                    )
                    with gr.Column():
                        with gr.Row():
                            switch_btn_en = gr.Button("🔀 Load Session", variant="secondary")
                            refresh_btn_en = gr.Button("🔄 Refresh", variant="primary")
                        with gr.Row():
                            clear_chat_btn_en = gr.Button("🗑️ Clear Session", variant="secondary")
                        close_management_btn_en = gr.Button("❌ Close Session Management", variant="primary")

            # Health Management Section
            with gr.Column(visible=False) as health_management_en:
                health_context_en = gr.Textbox(
                    label="🏥 Patient's Health Information",
                    placeholder="e.g., Patient's name, age, high blood pressure, diabetes, sleep issues, stress level...",
                    value="",
                    max_lines=5,
                    lines=3,
                    info="This information will be saved in the session and used to personalize advice",
                    elem_classes=["saved_memories-box"]
                )
                update_saved_memories_btn_en = gr.Button("💾 Update Health Information", variant="primary")
                back_btn_1_en = gr.Button("⏪ Back to Chat", variant="primary")

            with gr.Column() as chatbot_window_en:
                # Chat Interface Section with beautiful design
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.HTML("""
                        <div style="text-align: center; padding: 1rem 0;">
                            <h2 style="color: #659435; margin: 0; font-size: 1.5rem;">💬 Health Consultation Chatbot</h2>
                            <p style="color: #666; margin: 0.5rem 0 0 0;">Chat with KaLLaM for personalized health guidance</p>
                        </div>
                        """)

                chatbot_en = gr.Chatbot(
                    label="💭 Chat with KaLLaM",
                    height=400,
                    show_label=False,
                    type="messages",
                    elem_classes=["chat-container"],
                    value=[{"role": "assistant", "content": "Hello there! I'm KaLLaM 🌿, your caring doctor chatbot 💖 I'll be here to support your health and well-being. How are you feeling today? 😊"}],
                )

                with gr.Row():
                    with gr.Column(scale=5):
                        msg_en = gr.Textbox(
                            label="💬 Type your message",
                            placeholder="Type your question or health information for consultation...",
                            lines=2,
                            max_lines=4,
                            show_label=False,
                        )
                    with gr.Column(scale=1, min_width=120):
                        send_btn_en = gr.Button(
                            "📤 Send Message", 
                            variant="primary", 
                            size="lg",
                            elem_classes=["btn", "btn-primary"]
                        )

        with gr.Tab("Contact Us"):
            gr.Markdown("""
### **Built with ❤️ by:**
                        
**👨‍💻 Nopnatee Trivoravong** 📧 nopnatee.triv@gmail.com 🐙 [GitHub Profile](https://github.com/Nopnatee)

**👨‍💻 Khamic Srisutrapon** 📧 khamic.sk@gmail.com 🐙 [GitHub Profile](https://github.com/Khamic672)

**👩‍💻 Napas Siripala** 📧 millynapas@gmail.com 🐙 [GitHub Profile](https://github.com/kaoqueri)

---
""")

        # ====== EVENT HANDLERS ======
        
        # Thai Version Event Handlers
        refresh_btn.click(
            fn=refresh_session_list, 
            outputs=[session_dropdown]
        )

        switch_btn.click(
            fn=switch_session,
            inputs=[session_dropdown],
            outputs=[chatbot, msg, session_result, session_status, health_context]
        ).then(
            fn=refresh_session_list, 
            outputs=[session_dropdown]
        )

        back_btn_1.click(
            fn=hide_buttons,
            inputs=None,
            outputs=[health_management]
        ).then( 
            fn=show_buttons,
            inputs=None,
            outputs=[chatbot_window]
        )

        back_btn_2.click(
            fn=hide_buttons,
            inputs=None,
            outputs=[summary_page]
        ).then( 
            fn=show_buttons,
            inputs=None,
            outputs=[chatbot_window]
        )

        new_session_btn.click(
            fn=create_new_session,
            inputs=[health_context],
            outputs=[chatbot, msg, session_result, session_status, health_context]
        ).then(
            fn=refresh_session_list, 
            outputs=[session_dropdown]
        )

        edit_profile_btn.click( 
            fn=clear_all_buttons,
            inputs=None,
            outputs=[chatbot_window, session_management, summary_page]
        ).then( 
            fn=show_buttons,
            inputs=None,
            outputs=[health_management]
        ).then(
            fn=refresh_session_list, 
            outputs=[session_dropdown]
        )

        manage_session_btn.click( 
            fn=clear_all_buttons,
            inputs=None,
            outputs=[chatbot_window, health_management, summary_page]
        ).then( 
            fn=show_buttons,
            inputs=None,
            outputs=[session_management]
        ).then(
            fn=refresh_session_list, 
            outputs=[session_dropdown]
        )

        close_management_btn.click(
            fn=hide_buttons,
            inputs=None,
            outputs=[session_management]
        ).then(
            fn=show_buttons,
            inputs=None,
            outputs=[chatbot_window]
        ).then(
            fn=refresh_session_list, 
            outputs=[session_dropdown]
        )

        update_saved_memories_btn.click(
            fn=update_medical_saved_memories,
            inputs=[health_context],
            outputs=[session_status, session_result]
        ).then(
            fn=show_buttons,
            inputs=None,
            outputs=[chatbot_window]
        ).then( 
            fn=hide_buttons,
            inputs=None,
            outputs=[health_management]
        ).then(
            fn=refresh_session_list, 
            outputs=[session_dropdown]
        )

        send_btn.click(
            fn=lambda: set_button_loading("⏳ ประมวลผล..."),
            outputs=[send_btn]
        ).then(
            fn=process_chat_message,
            inputs=[msg, chatbot],
            outputs=[chatbot, msg]
        ).then(
            fn=refresh_session_list, 
            outputs=[session_dropdown]
        ).then(
            fn=lambda: reset_button("📤 ส่งข้อความ", "primary"),
            outputs=[send_btn]
        )

        msg.submit(
            fn=process_chat_message,
            inputs=[msg, chatbot],
            outputs=[chatbot, msg]
        ).then(
            fn=refresh_session_list, 
            outputs=[session_dropdown]
        )

        clear_chat_btn.click(
            fn=lambda: set_button_loading("⏳ กำลังลบ..."),
            outputs=[clear_chat_btn]
        ).then(
            fn=clear_session,
            outputs=[chatbot, msg, session_result, session_status, health_context]
        ).then(
            fn=lambda: reset_button("🗑️ ล้าง Session", "secondary"),
            outputs=[clear_chat_btn]
        )

        # English Version Event Handlers (mirror Thai version)
        refresh_btn_en.click(
            fn=refresh_session_list, 
            outputs=[session_dropdown_en]
        )

        switch_btn_en.click(
            fn=switch_session,
            inputs=[session_dropdown_en],
            outputs=[chatbot_en, msg_en, session_result_en, session_status_en, health_context_en]
        ).then(
            fn=refresh_session_list, 
            outputs=[session_dropdown_en]
        )

        back_btn_1_en.click(
            fn=hide_buttons,
            inputs=None,
            outputs=[health_management_en]
        ).then( 
            fn=show_buttons,
            inputs=None,
            outputs=[chatbot_window_en]
        )

        back_btn_2_en.click(
            fn=hide_buttons,
            inputs=None,
            outputs=[summary_page_en]
        ).then( 
            fn=show_buttons,
            inputs=None,
            outputs=[chatbot_window_en]
        )

        new_session_btn_en.click(
            fn=create_new_session,
            inputs=[health_context_en],
            outputs=[chatbot_en, msg_en, session_result_en, session_status_en, health_context_en]
        ).then(
            fn=refresh_session_list, 
            outputs=[session_dropdown_en]
        )

        edit_profile_btn_en.click( 
            fn=clear_all_buttons,
            inputs=None,
            outputs=[chatbot_window_en, session_management_en, summary_page_en]
        ).then( 
            fn=show_buttons,
            inputs=None,
            outputs=[health_management_en]
        ).then(
            fn=refresh_session_list, 
            outputs=[session_dropdown_en]
        )

        manage_session_btn_en.click( 
            fn=clear_all_buttons,
            inputs=None,
            outputs=[chatbot_window_en, health_management_en, summary_page_en]
        ).then( 
            fn=show_buttons,
            inputs=None,
            outputs=[session_management_en]
        ).then(
            fn=refresh_session_list, 
            outputs=[session_dropdown_en]
        )

        close_management_btn_en.click(
            fn=hide_buttons,
            inputs=None,
            outputs=[session_management_en]
        ).then(
            fn=show_buttons,
            inputs=None,
            outputs=[chatbot_window_en]
        ).then(
            fn=refresh_session_list, 
            outputs=[session_dropdown_en]
        )

        update_saved_memories_btn_en.click(
            fn=update_medical_saved_memories,
            inputs=[health_context_en],
            outputs=[session_status_en, session_result_en]
        ).then(
            fn=show_buttons,
            inputs=None,
            outputs=[chatbot_window_en]
        ).then( 
            fn=hide_buttons,
            inputs=None,
            outputs=[health_management_en]
        ).then(
            fn=refresh_session_list, 
            outputs=[session_dropdown_en]
        )

        send_btn_en.click(
            fn=lambda: set_button_loading("⏳ Processing..."),
            outputs=[send_btn_en]
        ).then(
            fn=process_chat_message,
            inputs=[msg_en, chatbot_en],
            outputs=[chatbot_en, msg_en]
        ).then(
            fn=refresh_session_list, 
            outputs=[session_dropdown_en]
        ).then(
            fn=lambda: reset_button("📤 Send Message", "primary"),
            outputs=[send_btn_en]
        )

        msg_en.submit(
            fn=process_chat_message,
            inputs=[msg_en, chatbot_en],
            outputs=[chatbot_en, msg_en]
        ).then(
            fn=refresh_session_list, 
            outputs=[session_dropdown_en]
        )

        clear_chat_btn_en.click(
            fn=lambda: set_button_loading("⏳ Deleting..."),
            outputs=[clear_chat_btn_en]
        ).then(
            fn=clear_session,
            outputs=[chatbot_en, msg_en, session_result_en, session_status_en, health_context_en]
        ).then(
            fn=lambda: reset_button("🗑️ Clear Session", "secondary"),
            outputs=[clear_chat_btn_en]
        )

    return app

def main():
    app = create_app()
    # Resolve bind address and port
    server_name = os.getenv("GRADIO_SERVER_NAME", "0.0.0.0")
    server_port = int(os.getenv("PORT", os.getenv("GRADIO_SERVER_PORT", 8080)))

    # Basic health log to confirm listening address
    try:
        hostname = socket.gethostname()
        ip_addr = socket.gethostbyname(hostname)
    except Exception:
        hostname = "unknown"
        ip_addr = "unknown"

    logger.info(
        "Starting Gradio app | bind=%s:%s | host=%s ip=%s",
        server_name,
        server_port,
        hostname,
        ip_addr,
    )
    logger.info(
        "Env: PORT=%s GRADIO_SERVER_NAME=%s GRADIO_SERVER_PORT=%s",
        os.getenv("PORT"),
        os.getenv("GRADIO_SERVER_NAME"),
        os.getenv("GRADIO_SERVER_PORT"),
    )
    # Secrets presence check (mask values)
    def _mask(v: str | None) -> str:
        if not v:
            return "<missing>"
        return f"set(len={len(v)})"
    logger.info(
        "Secrets: SEA_LION_API_KEY=%s GEMINI_API_KEY=%s",
        _mask(os.getenv("SEA_LION_API_KEY")),
        _mask(os.getenv("GEMINI_API_KEY")),
    )

    app.launch(
        share=True,
        server_name=server_name,  # cloud: 0.0.0.0, local: 127.0.0.1
        server_port=server_port,  # cloud: $PORT, local: 7860/8080
        debug=True,
        show_error=True,
        inbrowser=True,
    )

def main_local_only():
    """Launch application locally only (no share link attempt)"""
    try:
        app = create_app()
        print("Launching locally only...")
        print("Your app will be available at: http://localhost:7860")
        
        app.launch(
            share=True,
            server_name="127.0.0.1",
            server_port=7860,
            debug=True,
            show_error=True,
            inbrowser=True
        )
    except Exception as e:
        logger.error(f"Failed to launch application: {e}")
        raise

if __name__ == "__main__":
    # Choose which version to use:
    
    # Option 1: Try share link (current behavior - will run locally if share fails)
    # main()
    
    # Option 2: Local only (uncomment to use instead)
    main_local_only()