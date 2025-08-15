import gradio as gr
import json
import logging
from datetime import datetime
from typing import List, Tuple, Optional, Dict, Any
from data_manager import ChatbotManager

# key improvement: - delete notification
# key improvement: - color
# key improvement: - language
# key improvement: - UI
# key improvement: - ease of usage
# key improvement: - loading bar (indicator)

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



# Initialize the chatbot manager
chatbot_manager = ChatbotManager(api_provider="sea_lion")
current_session_id = None

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global state management
class AppState:
    def __init__(self):
        self.current_session_id = None
        self.message_count = 0
        # self.auto_summary_threshold = 5
        # self.auto_notification_threshold = 10
    
    def reset(self):
        self.current_session_id = None
        self.message_count = 0

app_state = AppState()

# fine
def get_current_session_status() -> str:
    """Get current session status information."""
    if not app_state.current_session_id:
        return "🔴 **ไม่มี Session ที่ใช้งาน** - กรุณาสร้าง Session ใหม่"
    
    try:
        session_stats = chatbot_manager.get_session_stats(app_state.current_session_id)
        session_info = session_stats
        stats = session_stats.get("stats", {})
        
        # Safe formatting for potentially None values
        avg_latency = stats.get('avg_latency')
        latency_str = f"{float(avg_latency):.1f}" if avg_latency is not None else "0.0"
        
        condition = session_info.get('condition')
        condition_str = condition if condition else 'ไม่ได้ระบุ'
        
        return f"""
### 🏥 **สภาวะสุขภาพ:** {condition_str}  
🟢 **Session ปัจจุบัน:** `{app_state.current_session_id}`  
📅 **สร้างเมื่อ:** {session_info.get('timestamp', 'N/A')}  
💬 **จำนวนข้อความ:** {stats.get('message_count', 0) or 0} ข้อความ  
📋 **จำนวนสรุป:** {session_info.get('total_summaries', 0) or 0} ครั้ง  
⚡ **Latency เฉลี่ย:** {latency_str} ms  
🔧 **Model:** {session_info.get('model_used', 'N/A')}
        """.strip()
    except Exception as e:
        logger.error(f"Error getting session status: {e}")
        return f"❌ **Error:** ไม่สามารถโหลดข้อมูล Session {app_state.current_session_id}"

# fine
def get_session_list() -> List[str]:
    """Get list of all active sessions with metadata."""
    try:
        sessions = chatbot_manager.list_sessions(active_only=True, limit=50)
        session_options = []
        
        for session in sessions:
            condition = (session.get('condition') or 'ไม่ระบุ')[:20]
            msg_count = session.get('total_messages', 0)
            summary_count = session.get('total_summaries', 0)
            
            display_name = f"{session['session_id']} | {msg_count}💬 {summary_count}📋 | {condition}"
            session_options.append(display_name)
        
        return session_options if session_options else ["ไม่มี Session"]
    except Exception as e:
        logger.error(f"Error getting session list: {e}")
        return ["Error loading sessions"]

def extract_session_id(dropdown_value: str) -> Optional[str]:
    """Extract session ID from dropdown display value."""
    if not dropdown_value or dropdown_value in ["ไม่มี Session", "Error loading sessions"]:
        return None
    return dropdown_value.split(" | ")[0]

def refresh_session_list() -> gr.update:
    """Refresh the session dropdown list."""
    session_list = get_session_list()
    return gr.update(choices=session_list, value=session_list[0] if session_list else None)

def create_new_session(condition: str = "") -> Tuple[List, str, str, str, str]:
    """Create a new session."""
    try:
        session_id = chatbot_manager.start_session(condition=condition or None)
        app_state.current_session_id = session_id
        app_state.message_count = 0
        
        status = get_current_session_status()
        result_msg = f"✅ **สร้าง Session ใหม่สำเร็จ!**\n\n🆔 Session ID: `{session_id}`"
        
        return [], "", result_msg, status, condition
    except Exception as e:
        logger.error(f"Error creating new session: {e}")
        error_msg = f"❌ **ไม่สามารถสร้าง Session ใหม่:** {str(e)}"
        return [], "", error_msg, get_current_session_status(), ""

def switch_session(dropdown_value: str) -> Tuple[List, str, str, str, str]:
    """Switch to selected session."""
    session_id = extract_session_id(dropdown_value)
    
    if not session_id:
        return [], "", "❌ **Session ID ไม่ถูกต้อง**", get_current_session_status(), ""
    
    try:
        session = chatbot_manager.get_session(session_id)
        if not session:
            return [], "", f"❌ **ไม่พบ Session:** {session_id}", get_current_session_status(), ""
        
        app_state.current_session_id = session_id
        app_state.message_count = session.get('total_messages', 0)
        
        # Load chat history
        chat_history = chatbot_manager._get_chat_history(session_id)
        gradio_history = []
        
        for msg in chat_history:
            if msg["role"] == "user":
                gradio_history.append({"role": "user", "content": msg["content"]})
            elif msg["role"] == "assistant":
                gradio_history.append({"role": "assistant", "content": msg["content"]})
        
        status = get_current_session_status()
        result_msg = f"✅ **เปลี่ยน Session สำเร็จ!**\n\n🆔 Session ID: `{session_id}`"
        condition = session.get('condition', '')
        
        return gradio_history, "", result_msg, status, condition
    except Exception as e:
        logger.error(f"Error switching session: {e}")
        error_msg = f"❌ **ไม่สามารถเปลี่ยน Session:** {str(e)}"
        return [], "", error_msg, get_current_session_status(), ""

def get_session_info() -> str:
    """Get detailed information about current session."""
    if not app_state.current_session_id:
        return "❌ **ไม่มี Session ที่ใช้งาน**"
    
    try:
        session_stats = chatbot_manager.get_session_stats(app_state.current_session_id)
        session_info = session_stats
        stats = session_stats.get("stats", {})
        
        # Safe formatting for potentially None values
        avg_latency = stats.get('avg_latency')
        latency_str = f"{float(avg_latency):.2f}" if avg_latency is not None else "0.00"
        
        total_tokens_in = stats.get('total_tokens_in') or 0
        total_tokens_out = stats.get('total_tokens_out') or 0
        
        condition = session_info.get('condition')
        condition_str = condition if condition else 'ไม่ได้ระบุ'
        
        summarized_history = session_info.get('summarized_history')
        summary_str = summarized_history if summarized_history else 'ยังไม่มีการสรุป'
        
        info = f"""
## 📊 ข้อมูลรายละเอียด Session: `{app_state.current_session_id}`

### 🔧 ข้อมูลพื้นฐาน
- **Session ID:** `{session_info['session_id']}`
- **สร้างเมื่อ:** {session_info.get('timestamp', 'N/A')}
- **ใช้งานล่าสุด:** {session_info.get('last_activity', 'N/A')}
- **Model:** {session_info.get('model_used', 'N/A')}
- **สถานะ:** {'🟢 Active' if session_info.get('is_active') else '🔴 Inactive'}

### 🏥 ข้อมูลสุขภาพ
- **สภาวะสุขภาพ:** {condition_str}

### 📈 สถิติการใช้งาน
- **จำนวนข้อความทั้งหมด:** {stats.get('message_count', 0) or 0} ข้อความ
- **จำนวนสรุป:** {session_info.get('total_summaries', 0) or 0} ครั้ง
- **Token Input รวม:** {total_tokens_in:,} tokens
- **Token Output รวม:** {total_tokens_out:,} tokens
- **Latency เฉลี่ย:** {latency_str} ms
- **ข้อความแรก:** {stats.get('first_message', 'N/A') or 'N/A'}
- **ข้อความล่าสุด:** {stats.get('last_message', 'N/A') or 'N/A'}

### 📋 ประวัติการสรุป
{summary_str}
        """.strip()
        
        return info
    except Exception as e:
        logger.error(f"Error getting session info: {e}")
        return f"❌ **Error:** {str(e)}"

def get_all_sessions_info() -> str:
    """Get information about all sessions."""
    try:
        sessions = chatbot_manager.list_sessions(active_only=False, limit=20)
        
        if not sessions:
            return "📭 **ไม่มี Session ในระบบ**"
        
        info_parts = ["# 📁 ข้อมูล Session ทั้งหมด\n"]
        
        for i, session in enumerate(sessions, 1):
            status_icon = "🟢" if session.get('is_active') else "🔴"
            condition = session.get('condition', 'ไม่ระบุ')[:30]
            
            session_info = f"""
## {i}. {status_icon} `{session['session_id']}`
- **สร้าง:** {session.get('timestamp', 'N/A')}
- **ใช้งานล่าสุด:** {session.get('last_activity', 'N/A')}
- **ข้อความ:** {session.get('total_messages', 0)} | **สรุป:** {session.get('total_summaries', 0)}
- **สภาวะ:** {condition}
- **Model:** {session.get('model_used', 'N/A')}
            """.strip()
            
            info_parts.append(session_info)
        
        return "\n\n".join(info_parts)
    except Exception as e:
        logger.error(f"Error getting all sessions info: {e}")
        return f"❌ **Error:** {str(e)}"

def update_medical_condition(condition: str) -> Tuple[str, str]:
    """Update medical condition for current session."""
    if not app_state.current_session_id:
        return get_current_session_status(), "❌ **ไม่มี Session ที่ใช้งาน**"
    
    if not condition.strip():
        return get_current_session_status(), "❌ **กรุณาระบุสภาวะสุขภาพ**"
    
    try:
        # Update condition in database
        with chatbot_manager._get_connection() as conn:
            conn.execute("""
                UPDATE sessions 
                SET condition = ?, last_activity = ? 
                WHERE session_id = ?
            """, (condition.strip(), datetime.now().isoformat(), app_state.current_session_id))
            conn.commit()
        
        status = get_current_session_status()
        result = f"✅ **อัปเดตสภาวะสุขภาพสำเร็จ!**\n\n📝 **ข้อมูลใหม่:** {condition.strip()}"
        
        return status, result
    except Exception as e:
        logger.error(f"Error updating medical condition: {e}")
        return get_current_session_status(), f"❌ **ไม่สามารถอัปเดตสภาวะสุขภาพ:** {str(e)}"

def process_chat_message(message: str, history: List, health_context: str) -> Tuple[List, str]:
    """Process chat message and return updated history."""
    if not app_state.current_session_id:
        history.append({"role": "assistant", "content": "❌ **กรุณาสร้าง Session ใหม่ก่อนใช้งาน**"})
        return history, ""
    
    if not message.strip():
        return history, ""
    
    try:
        # Add user message to history immediately
        history.append({"role": "user", "content": message})
        
        # Process message through chatbot manager
        bot_response = chatbot_manager.handle_message(
            session_id=app_state.current_session_id,
            user_message=message,
            health_status=health_context
        )
        
        # Add bot response to history
        history.append({"role": "assistant", "content": bot_response})
        
        # Update message count
        app_state.message_count += 2
        
        # Auto-summary check
        if app_state.message_count % app_state.auto_summary_threshold == 0:
            try:
                summary = chatbot_manager.summarize_session(app_state.current_session_id)
                history.append({
                    "role": "assistant", 
                    "content": f"📋 **สรุปอัตโนมัติ:** {summary}"
                })
            except Exception as e:
                logger.error(f"Auto-summary error: {e}")
        
        # Auto-notification check
        if app_state.message_count % app_state.auto_notification_threshold == 0:
            notification = generate_health_notification(history)
            history.append({
                "role": "assistant", 
                "content": f"🔔 **การแจ้งเตือนอัตโนมัติ:** {notification}"
            })
        
        return history, ""
    except Exception as e:
        logger.error(f"Error processing chat message: {e}")
        history.append({"role": "assistant", "content": f"❌ **ข้อผิดพลาด:** {str(e)}"})
        return history, ""

def generate_health_notification(history: List) -> str:
    """Generate health notification based on conversation history."""
    # Simple notification generator - in production, use your prompt system
    notifications = [
        "💧 อย่าลืมดื่มน้ำ 8-10 แก้วต่อวัน",
        "🚶‍♀️ ลุกขึ้นเดินและยืดเส้นยืดสายทุก 1 ชั่วโมง",
        "😴 นอนให้เพียงพอ 7-8 ชั่วโมงต่อคืน",
        "🥗 รับประทานผักและผลไม้ให้เพียงพอ",
        "🧘‍♀️ ทำสมาธิหรือผ่อนคลาย 10 นาทีต่อวัน",
        "💊 อย่าลืมทานยาตามเวลาที่แพทย์กำหนด"
    ]
    
    import random
    return random.choice(notifications)

def generate_manual_notification(history: List) -> List:
    """Generate manual health notification."""
    notification = generate_health_notification(history)
    history.append({
        "role": "assistant", 
        "content": f"🔔 **การแจ้งเตือนสุขภาพ:** {notification}"
    })
    return history

def force_update_summary() -> str:
    """Force update session summary."""
    if not app_state.current_session_id:
        return "❌ **ไม่มี Session ที่ใช้งาน**"
    
    try:
        summary = chatbot_manager.summarize_session(app_state.current_session_id)
        return f"✅ **สรุปข้อมูลสำเร็จ!**\n\n📋 **สรุป:** {summary}"
    except Exception as e:
        logger.error(f"Error forcing summary update: {e}")
        return f"❌ **ไม่สามารถสรุปข้อมูล:** {str(e)}"

def clear_session() -> Tuple[List, str, str, str, str]:
    """Clear current session."""
    if not app_state.current_session_id:
        return [], "", "❌ **ไม่มี Session ที่ใช้งาน**", get_current_session_status(), ""
    
    try:
        old_session_id = app_state.current_session_id
        chatbot_manager.delete_session(app_state.current_session_id)
        app_state.reset()
        
        result = f"✅ **ลบ Session สำเร็จ!**\n\n🗑️ **Session ที่ลบ:** `{old_session_id}`"
        return [], "", result, get_current_session_status(), ""
    except Exception as e:
        logger.error(f"Error clearing session: {e}")
        return [], "", f"❌ **ไม่สามารถลบ Session:** {str(e)}", get_current_session_status(), ""

def clear_summary() -> str:
    """Clear session summary."""
    if not app_state.current_session_id:
        return "❌ **ไม่มี Session ที่ใช้งาน**"
    
    try:
        with chatbot_manager._get_connection() as conn:
            conn.execute("""
                UPDATE sessions 
                SET summarized_history = NULL, total_summaries = 0 
                WHERE session_id = ?
            """, (app_state.current_session_id,))
            conn.commit()
        
        return f"✅ **ล้างสรุปสำเร็จ!**\n\n🗑️ **Session:** `{app_state.current_session_id}`"
    except Exception as e:
        logger.error(f"Error clearing summary: {e}")
        return f"❌ **ไม่สามารถล้างสรุป:** {str(e)}"

def export_session() -> str:
    """Export current session to JSON."""
    if not app_state.current_session_id:
        return "❌ **ไม่มี Session ที่ใช้งาน**"
    
    try:
        json_data = chatbot_manager.export_session_json(app_state.current_session_id)
        
        # Save to file
        filename = f"session_{app_state.current_session_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(json_data)
        
        return f"✅ **ส่งออกข้อมูลสำเร็จ!**\n\n📁 **ไฟล์:** `{filename}`"
    except Exception as e:
        logger.error(f"Error exporting session: {e}")
        return f"❌ **ไม่สามารถส่งออกข้อมูล:** {str(e)}"
    

def show_buttons():
    # Make the hidden buttons visible
    return gr.update(visible=True)

def hide_buttons():
    # Make the visible buttons hidden
    return gr.update(visible=False)

# UI Visual
def create_app() -> gr.Blocks:
    """Create the Gradio application."""
    custom_css = """
    :root {
        --kallam: #659435;
    }
    .gradio-container {
        max-width: 100% !important;
        width: 100% !important;
        margin: 0 auto !important;
    }
    .tab-nav {
        background: linear-gradient(90deg,rgba(111, 198, 232, 1) 0%, rgba(87, 199, 133, 1) 50%, rgba(237, 221, 83, 1) 100%);
    }
    .chat-container {
        border-radius: 10px;
        border: 1px solid #e0e0e0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }
    .summary-box {
        border-radius: 8px;
        padding: 15px;
        margin: 10px 0;
        border: 4px solid var(--kallam);
        bottom: 0;
    }
    .session-info {
        border-radius: 8px;
        padding: 15px;
        margin: 10px 0;
        background: linear-gradient(90deg,rgba(111, 198, 232, 1) 0%, rgba(87, 199, 133, 1) 50%, rgba(237, 221, 83, 1) 100%);
    }
        color: white;
        font-weight: 500;
    }
    .condition-box {
        border-radius: 8px;
        padding: 10px;
        margin: 5px 0;
        border: 1px solid #ddd;
    }
    .stat-card {
        background: white;
        border-radius: 8px;
        padding: 15px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        border-left: 4px solid #28a745;
    }
    """


    with gr.Blocks(
        title="🥬 KaLLaM",
        theme=gr.themes.Default(primary_hue="green", secondary_hue="cyan"),
        css=custom_css,
    ) as app:
        gr.HTML(f"""      
            <h1>{CABBAGE_SVG} KaLLaM - Thai Motivational Therapeutic Advisor</h1>
            """)
        with gr.Tab("Main App (Thai Ver.)"):
            with gr.Sidebar():
                # Session Status Display
                with gr.Column(
                    elem_classes=["session-info"]
                ):
                    gr.Markdown(
                        value="## ประวัติของผู้ใช้งาน",
                    )
                    with gr.Row():
                        with gr.Column():
                            session_status = gr.Markdown(
                                value=get_current_session_status(), 
                            )
                with gr.Column():
                    manage_session_btn = gr.Button("🗂️ จัดการ Session", variant="secondary")
                    edit_profile_btn = gr.Button("✏️ แก้ไขข้อมูลสุขภาพ", variant="secondary")
                # Session Details Accordion
                session_result = gr.Markdown(
                    value="**กำลังรอการอัปเดต...**", 
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
                            new_session_btn = gr.Button("➕ Session ใหม่", variant="secondary")
                            switch_btn = gr.Button("🔀 โหลด Session", variant="secondary")
                            refresh_btn = gr.Button("🔄 รีเฟรช", variant="primary")
                        with gr.Row():
                            clear_chat_btn = gr.Button("🗑️ ล้าง Session", variant="secondary")
                            clear_summary_btn = gr.Button("📝 ล้างสรุป", variant="secondary")
                        close_management_btn = gr.Button("❌ ปิดการจัดการ Session", variant="secondary")

            # Health Management Section
            with gr.Column(visible=False) as health_management:
                health_context = gr.Textbox(
                    label="🏥 ข้อมูลสุขภาพของผู้ป่วย",
                    placeholder="เช่น: ชื่อผู้ป่วย, อายุ, ความดันโลหิตสูง, เบาหวาน, ปัญหาการนอน, ระดับความเครียด...",
                    value="",
                    max_lines=5,
                    lines=3,
                    info="ข้อมูลนี้จะถูกเก็บใน session และใช้ปรับแต่งคำแนะนำ",
                    elem_classes=["condition-box"]
                )
                update_condition_btn = gr.Button("💾 อัปเดตข้อมูลสุขภาพ", variant="primary")
            
            with gr.Column() as chatbot_window:
                # Chat Interface Section
                gr.Markdown("## 💬 แชทบอทให้คำปรึกษาสุขภาพ")

                chatbot = gr.Chatbot(
                    label="💭 พูดคุยกับ KaLLaM",
                    height=300,
                    show_label=True,
                    type="messages",
                    elem_classes=["chat-container"],
                )

                with gr.Row():
                    with gr.Column(scale=4):
                        msg = gr.Textbox(
                            label="💬 พิมพ์ข้อความของคุณ",
                            placeholder="พิมพ์คำถามหรือข้อมูลสุขภาพ...",
                            lines=2,
                            max_lines=8,
                        )
                    with gr.Column(scale=1):
                        send_btn = gr.Button("📤 ส่งข้อความ", variant="primary")

            # Event Handlers
            refresh_btn.click(
                fn=refresh_session_list, 
                outputs=[session_dropdown]
            )

            switch_btn.click(
                fn=switch_session,
                inputs=[session_dropdown],
                outputs=[chatbot, msg, session_result, session_status, health_context]
            )

            new_session_btn.click(
                fn=create_new_session,
                inputs=[health_context],
                outputs=[chatbot, msg, session_result, session_status, health_context]
            )

            edit_profile_btn.click( 
                fn=hide_buttons,
                inputs=None,
                outputs=[chatbot_window]
            ).then(
                fn=hide_buttons,
                inputs=None,
                outputs=[close_management_btn]
            ).then( 
                fn=show_buttons,
                inputs=None,
                outputs=[health_management]
            ).then( 
                fn=show_buttons,
                inputs=None,
                outputs=[update_condition_btn]
            )

            manage_session_btn.click( 
                fn=hide_buttons,
                inputs=None,
                outputs=[update_condition_btn]
            ).then( 
                fn=hide_buttons,
                inputs=None,
                outputs=[health_management]
            ).then(
                fn=hide_buttons,
                inputs=None,
                outputs=[chatbot_window]
            ).then( 
                fn=show_buttons,
                inputs=None,
                outputs=[session_management]
            ).then( 
                fn=show_buttons,
                inputs=None,
                outputs=[close_management_btn]
            )

            close_management_btn.click(
                fn=hide_buttons,
                inputs=None,
                outputs=[close_management_btn]
            ).then( 
                fn=hide_buttons,
                inputs=None,
                outputs=[session_management]
            ).then(
                fn=show_buttons,
                inputs=None,
                outputs=[chatbot_window]
            )

            update_condition_btn.click(
                fn=update_medical_condition,
                inputs=[health_context],
                outputs=[session_status, session_result]
            ).then(
                fn=show_buttons,
                inputs=None,
                outputs=[chatbot_window]
            ).then(
                fn=hide_buttons,
                inputs=None,
                outputs=[update_condition_btn]
            ).then(
                fn=show_buttons,
                inputs=None,
                outputs=[edit_profile_btn]
            ).then( 
                fn=hide_buttons,
                inputs=None,
                outputs=[health_management]
            ).then(
                fn=show_buttons,
                inputs=None,
                outputs=[manage_session_btn]
            )

            send_btn.click(
                fn=process_chat_message,
                inputs=[msg, chatbot, health_context],
                outputs=[chatbot, msg]
            )

            msg.submit(
                fn=process_chat_message,
                inputs=[msg, chatbot, health_context],
                outputs=[chatbot, msg]
            )

            clear_chat_btn.click(
                fn=clear_session,
                outputs=[chatbot, msg, session_result, session_status, health_context]
            )

            clear_summary_btn.click(
                fn=clear_summary, 
                outputs=[session_result]
            )
        with gr.Tab("Main App (English Ver.)"):
            with gr.Sidebar():
                # Session Status Display
                with gr.Column(
                    elem_classes=["session-info"]
                ):
                    gr.Markdown(
                        value="## User Profile",
                    )
                    with gr.Row():
                        with gr.Column():
                            session_status = gr.Markdown(
                                value=get_current_session_status(), 
                            )
                with gr.Column():
                    manage_session_btn = gr.Button("🗂️ Manage Session", variant="secondary")
                    edit_profile_btn = gr.Button("✏️ Edit Health Profile", variant="secondary")
                # Session Details Accordion
                session_result = gr.Markdown(
                    value="**Waiting for update...**", 
                    elem_classes=["summary-box"],
                )
            with gr.Row(visible=False) as session_management:
                with gr.Column(scale=3):
                    gr.Markdown("### 🗂️ Session Management")
                    session_ids = get_session_list()
                    initial_session = session_ids[0] if session_ids else None

                    session_dropdown = gr.Dropdown(
                        choices=session_ids,
                        value=initial_session,
                        label="🗒️ Select Session",
                        info="Select the session you want to switch to",
                    )
                    with gr.Column():
                        with gr.Row():
                            new_session_btn = gr.Button("➕ New Session", variant="secondary")
                            switch_btn = gr.Button("🔀 Load Session", variant="secondary")
                            refresh_btn = gr.Button("🔄 Refresh", variant="primary")
                        with gr.Row():
                            clear_chat_btn = gr.Button("🗑️ Clear Session", variant="secondary")
                            clear_summary_btn = gr.Button("📝 Clear Summary", variant="secondary")
                        close_management_btn = gr.Button("❌ Close Session Management", variant="secondary")

            # Health Management Section
            with gr.Column(visible=False) as health_management:
                health_context = gr.Textbox(
                    label="🏥 Patient's Health Information",
                    placeholder="e.g., Patient's name, age, high blood pressure, diabetes, sleep issues, stress level...",
                    value="",
                    max_lines=5,
                    lines=3,
                    info="This information will be saved in the session and used to personalize advice",
                    elem_classes=["condition-box"]
                )
                update_condition_btn = gr.Button("💾 Update Health Information", variant="primary")
            
            with gr.Column() as chatbot_window:
                # Chat Interface Section
                gr.Markdown("## 💬 Health Consultation Chatbot")

                chatbot = gr.Chatbot(
                    label="💭 Chat with KaLLaM",
                    height=300,
                    show_label=True,
                    type="messages",
                    elem_classes=["chat-container"],
                )

                with gr.Row():
                    with gr.Column(scale=4):
                        msg = gr.Textbox(
                            label="💬 Type your message",
                            placeholder="Type your question or health information...",
                            lines=2,
                            max_lines=8,
                        )
                    with gr.Column(scale=1):
                        send_btn = gr.Button("📤 Send Message", variant="primary")
            # Event Handlers
            refresh_btn.click(
                fn=refresh_session_list, 
                outputs=[session_dropdown]
            )

            switch_btn.click(
                fn=switch_session,
                inputs=[session_dropdown],
                outputs=[chatbot, msg, session_result, session_status, health_context]
            )

            new_session_btn.click(
                fn=create_new_session,
                inputs=[health_context],
                outputs=[chatbot, msg, session_result, session_status, health_context]
            )

            edit_profile_btn.click( 
                fn=hide_buttons,
                inputs=None,
                outputs=[chatbot_window]
            ).then(
                fn=hide_buttons,
                inputs=None,
                outputs=[close_management_btn]
            ).then( 
                fn=show_buttons,
                inputs=None,
                outputs=[health_management]
            ).then( 
                fn=show_buttons,
                inputs=None,
                outputs=[update_condition_btn]
            )

            manage_session_btn.click( 
                fn=hide_buttons,
                inputs=None,
                outputs=[update_condition_btn]
            ).then( 
                fn=hide_buttons,
                inputs=None,
                outputs=[health_management]
            ).then(
                fn=hide_buttons,
                inputs=None,
                outputs=[chatbot_window]
            ).then( 
                fn=show_buttons,
                inputs=None,
                outputs=[session_management]
            ).then( 
                fn=show_buttons,
                inputs=None,
                outputs=[close_management_btn]
            )

            close_management_btn.click(
                fn=hide_buttons,
                inputs=None,
                outputs=[close_management_btn]
            ).then( 
                fn=hide_buttons,
                inputs=None,
                outputs=[session_management]
            ).then(
                fn=show_buttons,
                inputs=None,
                outputs=[chatbot_window]
            )

            update_condition_btn.click(
                fn=update_medical_condition,
                inputs=[health_context],
                outputs=[session_status, session_result]
            ).then(
                fn=show_buttons,
                inputs=None,
                outputs=[chatbot_window]
            ).then(
                fn=hide_buttons,
                inputs=None,
                outputs=[update_condition_btn]
            ).then(
                fn=show_buttons,
                inputs=None,
                outputs=[edit_profile_btn]
            ).then( 
                fn=hide_buttons,
                inputs=None,
                outputs=[health_management]
            ).then(
                fn=show_buttons,
                inputs=None,
                outputs=[manage_session_btn]
            )

            send_btn.click(
                fn=process_chat_message,
                inputs=[msg, chatbot, health_context],
                outputs=[chatbot, msg]
            )

            msg.submit(
                fn=process_chat_message,
                inputs=[msg, chatbot, health_context],
                outputs=[chatbot, msg]
            )

            clear_chat_btn.click(
                fn=clear_session,
                outputs=[chatbot, msg, session_result, session_status, health_context]
            )

            clear_summary_btn.click(
                fn=clear_summary, 
                outputs=[session_result]
            )
            

        with gr.Tab("READ ME"):
            gr.Markdown("""
            ### 🧪 **Note for Proof of Concept (POC)**
            * **Future Vision:** We aim for a fully localized Thai user interface (UI). 🇹🇭
            * **Current Version:** For this POC, the UI has two versions. The fully Thai one and the partially English one to make it easier for judges to visualize and evaluate. 👀
                        

            ### ⚙️ **Key Features**
            This system uses advanced AI to provide health advice and behavioral therapy, with the following features:
            * **🔄 Auto-Summary:** Summarizes the conversation every 5 messages.
            * **💾 Session Management:** Stores and manages conversation sessions.
            * **🏥 Medical Tracking:** Tracks health conditions across sessions.
            * **📊 Analytics:** Provides detailed usage statistics.""")

        
    return app


def main():
    app = create_app()
    app.launch(
        share=False,
        server_name="0.0.0.0",
        server_port=7860,
        debug=True,
        show_error=True,
        inbrowser=True,
    )


if __name__ == "__main__":
    main()
