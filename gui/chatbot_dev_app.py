# chatbot_dev_app.py
import gradio as gr
import logging
import socket
from datetime import datetime
from typing import List, Tuple, Optional
import os

from kallam.app.chatbot_manager import ChatbotManager
from kallam.infra.db import sqlite_conn  # use the shared helper

# -----------------------
# Init
# -----------------------
chatbot_manager = ChatbotManager(log_level="DEBUG")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
🟢 **Session ปัจจุบัน:** `{app_state.current_session_id}`
📅 **สร้างเมื่อ:** {session_info.get('timestamp', 'N/A')}
💬 **จำนวนข้อความ:** {stats.get('message_count', 0) or 0} ข้อความ
📋 **จำนวนสรุป:** {session_info.get('total_summaries', 0) or 0} ครั้ง
🏥 **สถานะกำหนดเอง:** {saved_memories}
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

### 🏥 ข้อมูลสถานะกำหนดเอง
- **สถานะกำหนดเอง:** {saved_memories}

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
        return get_current_session_status(), "❌ **กรุณาสถานะกำหนดเอง**"

    try:
        with sqlite_conn(str(chatbot_manager.db_path)) as conn:
            conn.execute(
                "UPDATE sessions SET saved_memories = ?, last_activity = ? WHERE session_id = ?",
                (saved_memories.strip(), datetime.now().isoformat(), app_state.current_session_id),
            )
        status = get_current_session_status()
        result = f"✅ **อัปเดตสถานะกำหนดเองสำเร็จ!**\n\n📝 **ข้อมูลใหม่:** {saved_memories.strip()}"
        return status, result
    except Exception as e:
        logger.error(f"Error updating saved_memories: {e}")
        return get_current_session_status(), f"❌ **ไม่สามารถอัปเดตสถานะกำหนดเอง:** {e}"

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

def force_update_summary() -> str:
    if not app_state.current_session_id:
        return "❌ **ไม่มี Session ที่ใช้งาน**"
    try:
        s = chatbot_manager.summarize_session(app_state.current_session_id)
        return f"✅ **สรุปข้อมูลสำเร็จ!**\n\n📋 **สรุป:** {s}"
    except Exception as e:
        logger.error(f"Error forcing summary update: {e}")
        return f"❌ **ไม่สามารถสรุปข้อมูล:** {e}"

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
# -----------------------
# UI
# -----------------------
def create_app() -> gr.Blocks:
    custom_css = """
    .gradio-container { max-width: 1400px !important; margin: 0 auto !important; }
    .tab-nav { background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); }
    .chat-container { border-radius: 10px; border: 1px solid #e0e0e0; box-shadow: 0 2px 8px rgba(0,0,0,0.1); }
    .summary-box { border-radius: 8px; padding: 15px; margin: 10px 0; }
    .session-info { border-radius: 8px; padding: 15px; margin: 10px 0;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white; font-weight: 500; }
    .saved_memories-box { border-radius: 8px; padding: 10px; margin: 5px 0; border: 1px solid #ddd; }
    .stat-card { background: white; border-radius: 8px; padding: 15px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); border-left: 4px solid #28a745; }
    .red-button { background-color: red !important; color: white !important; }
    """

    with gr.Blocks(title="🌟 DEMO ระบบจำลองการคุยกับลูกและให้คำแนะนำสำหรับผู้ปกครอง",
                   css=custom_css) as app:

        gr.Markdown("""
# 🌟 ระบบจำลองการคุยกับลูกและให้คำแนะนำสำหรับผู้ปกครอง

🔄 **สรุปอัตโนมัติ:** ทุก 10 ข้อความ  
🔔 **วิเคราะห์บทสนทนา:** เรียกใช้ด้วยปุ่ม Follow-up  
💾 **Session Management**  
🏥 **Custom saved_memories**  
📊 **Analytics**
        """)

        session_status = gr.Markdown(value=get_current_session_status(), elem_classes=["session-info"])

        gr.Markdown("## 🗂️ การจัดการ Session")
        with gr.Row():
            with gr.Column(scale=3):
                session_dropdown = gr.Dropdown(
                    choices=get_session_list(),
                    value=None,
                    label="🗒️ เลือก Session",
                    info="เลือก session ที่ต้องการเปลี่ยนไป",
                )
            with gr.Column(scale=1):
                refresh_btn = gr.Button("🔄 รีเฟรช", variant="primary")
                switch_btn = gr.Button("🔀 โหลด Session", variant="secondary")
                new_session_btn = gr.Button("➕ Session ใหม่", variant="secondary")

        with gr.Row():
            session_info_btn = gr.Button("👀 ข้อมูล Session", variant="secondary")
            all_sessions_btn = gr.Button("📁 ดู Session ทั้งหมด", variant="secondary")
            export_btn = gr.Button("📤 ส่งออกข้อมูลทั้งหมดเป็น.json (dev)", variant="secondary")

        with gr.Accordion("📊 ข้อมูลรายละเอียด Session", open=False):
            session_result = gr.Markdown(value="**กำลังรอการอัปเดต...**", elem_classes=["summary-box"])
            session_info_display = gr.Markdown(value="", elem_classes=["summary-box"])

        gr.Markdown("---")
        gr.Markdown("## 🏥 การจัดการสถานะการสนทนา")

        with gr.Row():
            health_context = gr.Textbox(
                label="🏥 ข้อมูลสถานะของการสนทนา",
                placeholder="เช่น: ชื่อเด็ก, อายุ, พฤติกรรมที่อยากโฟกัส",
                value="",
                max_lines=5, lines=3,
                info="ข้อมูลนี้จะถูกเก็บใน session และใช้ปรับแต่งบทสนทนา",
                elem_classes=["saved_memories-box"],
            )
            update_saved_memories_btn = gr.Button("💾 อัปเดตข้อมูล", variant="primary")

        gr.Markdown("---")
        gr.Markdown("## 💬 แชทบอทจำลองการสนทนา")

        chatbot = gr.Chatbot(
            label="💭 การสนทนากับ AI",
            height=500, show_label=True, type="messages",
            elem_classes=["chat-container"], avatar_images=("👤", "🤖")
        )

        with gr.Row():
            with gr.Column(scale=4):
                msg = gr.Textbox(
                    label="💬 พิมพ์ข้อความของคุณ",
                    placeholder="พิมพ์คำถามหรือข้อมูล...",
                    lines=2, max_lines=8,
                )
            with gr.Column(scale=1):
                send_btn = gr.Button("📤 ส่งข้อความ", variant="primary")
                followup_btn = gr.Button("🔔 สร้างการวิเคราะห์บทสนทนา", variant="secondary")
                update_summary_btn = gr.Button("📋 บังคับสรุปแชท (dev)", variant="secondary")

        with gr.Row():
            clear_chat_btn = gr.Button("🗑️ ล้าง Session", variant="secondary")
            clear_summary_btn = gr.Button("📝 ล้างสรุป", variant="secondary")

        # Small helpers for button UX
        def set_button_loading(text): return gr.update(value=text, elem_classes=["red-button"], variant="stop")
        def reset_button(text, variant): return gr.update(value=text, elem_classes=[], variant=variant)

        # Wiring
        refresh_btn.click(fn=refresh_session_list, outputs=[session_dropdown])

        switch_btn.click(
            fn=switch_session,
            inputs=[session_dropdown],
            outputs=[chatbot, msg, session_result, session_status, health_context],
        )

        new_session_btn.click(
            fn=create_new_session,
            inputs=[health_context],
            outputs=[chatbot, msg, session_result, session_status, health_context],
        )

        session_info_btn.click(fn=get_session_info, outputs=[session_info_display])
        all_sessions_btn.click(fn=get_all_sessions_info, outputs=[session_info_display])
        export_btn.click(fn=export_all_sessions, outputs=[session_info_display])
        export_btn.click(fn=lambda: set_button_loading("⏳ ประมวลผล..."), outputs=[export_btn]) \
            .then(fn=export_all_sessions) \
            .then(fn=lambda: reset_button("📤 ส่งออกข้อมูลทั้งหมดเป็น.json (dev)", variant="secondary"), outputs=[export_btn])

        update_saved_memories_btn.click(
            fn=update_medical_saved_memories,
            inputs=[health_context],
            outputs=[session_status, session_result],
        )

        send_btn.click(fn=lambda: set_button_loading("⏳ ประมวลผล..."), outputs=[send_btn]) \
            .then(fn=process_chat_message, inputs=[msg, chatbot], outputs=[chatbot, msg]) \
            .then(fn=lambda: reset_button("📤 ส่งข้อความ", "primary"), outputs=[send_btn])

        msg.submit(fn=process_chat_message, inputs=[msg, chatbot], outputs=[chatbot, msg])

        followup_btn.click(fn=lambda: set_button_loading("⏳ ประมวลผล..."), outputs=[followup_btn]) \
            .then(fn=generate_followup, inputs=[chatbot], outputs=[chatbot]) \
            .then(fn=lambda: reset_button("🔔 สร้างการวิเคราะห์บทสนทนา", "secondary"), outputs=[followup_btn])

        update_summary_btn.click(fn=lambda: set_button_loading("⏳ กำลังสรุป..."), outputs=[update_summary_btn]) \
            .then(fn=force_update_summary, outputs=[session_result]) \
            .then(fn=lambda: reset_button("📋 บังคับสรุปแชท (dev)", "secondary"), outputs=[update_summary_btn])

        clear_chat_btn.click(fn=lambda: set_button_loading("⏳ กำลังลบ..."), outputs=[clear_chat_btn]) \
            .then(fn=clear_session, outputs=[chatbot, msg, session_result, session_status, health_context]) \
            .then(fn=lambda: reset_button("🗑️ ล้าง Session", "secondary"), outputs=[clear_chat_btn])

        clear_summary_btn.click(fn=lambda: set_button_loading("⏳ กำลังล้าง..."), outputs=[clear_summary_btn]) \
            .then(fn=clear_all_summaries, outputs=[session_result]) \
            .then(fn=lambda: reset_button("📝 ล้างสรุป", "secondary"), outputs=[clear_summary_btn])

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

    app.launch(
        share=False,
        server_name=server_name,  # cloud: 0.0.0.0, local: 127.0.0.1
        server_port=server_port,  # cloud: $PORT, local: 7860/8080
        debug=True,
        show_error=True,
        inbrowser=False,
    )

if __name__ == "__main__":
    main()
