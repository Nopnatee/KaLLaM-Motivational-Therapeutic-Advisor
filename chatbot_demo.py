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
    
    def reset(self):
        self.current_session_id = None
        self.message_count = 0

app_state = AppState()
# fine
def get_current_session_status() -> str:
    """Get current session status information."""
    if not app_state.current_session_id:
        return "🔴 **No Active Session** - Please create a new session"
    
    try:
        session_stats = chatbot_manager.get_session_stats(app_state.current_session_id)
        session_info = session_stats
        stats = session_stats.get("stats", {})
        
        # Safe formatting for potentially None values
        avg_latency = stats.get('avg_latency')
        latency_str = f"{float(avg_latency):.1f}" if avg_latency is not None else "0.0"
        
        condition = session_info.get('condition')
        condition_str = condition if condition else 'Not specified'
        
        return f"""
### 🏥 **Health Condition:** {condition_str}  
🟢 **Current Session:** `{app_state.current_session_id}`  
📅 **Created on:** {session_info.get('timestamp', 'N/A')}  
💬 **Messages:** {stats.get('message_count', 0) or 0} messages  
📋 **Summaries:** {session_info.get('total_summaries', 0) or 0} times  
⚡ **Average Latency:** {latency_str} ms  
🔧 **Model:** {session_info.get('model_used', 'N/A')}
        """.strip()
    except Exception as e:
        logger.error(f"Error getting session status: {e}")
        return f"❌ **Error:** Could not load session data for {app_state.current_session_id}"

# fine
def get_session_list() -> List[str]:
    """Get list of all active sessions with metadata."""
    try:
        sessions = chatbot_manager.list_sessions(active_only=True, limit=50)
        session_options = []
        
        for session in sessions:
            condition = (session.get('condition') or 'Unspecified')[:20]
            msg_count = session.get('total_messages', 0)
            summary_count = session.get('total_summaries', 0)
            
            display_name = f"{session['session_id']} | {msg_count}💬 {summary_count}📋 | {condition}"
            session_options.append(display_name)
        
        return session_options if session_options else ["No Sessions"]
    except Exception as e:
        logger.error(f"Error getting session list: {e}")
        return ["Error loading sessions"]

def extract_session_id(dropdown_value: str) -> Optional[str]:
    """Extract session ID from dropdown display value."""
    if not dropdown_value or dropdown_value in ["No Sessions", "Error loading sessions"]:
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
        result_msg = f"✅ **New session created successfully!**\n\n🆔 Session ID: `{session_id}`"
        
        return [], "", result_msg, status, condition
    except Exception as e:
        logger.error(f"Error creating new session: {e}")
        error_msg = f"❌ **Could not create new session:** {str(e)}"
        return [], "", error_msg, get_current_session_status(), ""

def switch_session(dropdown_value: str) -> Tuple[List, str, str, str, str]:
    """Switch to selected session."""
    session_id = extract_session_id(dropdown_value)
    
    if not session_id:
        return [], "", "❌ **Invalid Session ID**", get_current_session_status(), ""
    
    try:
        session = chatbot_manager.get_session(session_id)
        if not session:
            return [], "", f"❌ **Session not found:** {session_id}", get_current_session_status(), ""
        
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
        result_msg = f"✅ **Switched session successfully!**\n\n🆔 Session ID: `{session_id}`"
        condition = session.get('condition', '')
        
        return gradio_history, "", result_msg, status, condition
    except Exception as e:
        logger.error(f"Error switching session: {e}")
        error_msg = f"❌ **Could not switch session:** {str(e)}"
        return [], "", error_msg, get_current_session_status(), ""

def get_session_info() -> str:
    """Get detailed information about current session."""
    if not app_state.current_session_id:
        return "❌ **No active session**"
    
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
        condition_str = condition if condition else 'Not specified'
        
        summarized_history = session_info.get('summarized_history')
        summary_str = summarized_history if summarized_history else 'No summary yet'
        
        info = f"""
## 📊 Session Details: `{app_state.current_session_id}`

### 🔧 Basic Information
- **Session ID:** `{session_info['session_id']}`
- **Created on:** {session_info.get('timestamp', 'N/A')}
- **Last activity:** {session_info.get('last_activity', 'N/A')}
- **Model:** {session_info.get('model_used', 'N/A')}
- **Status:** {'🟢 Active' if session_info.get('is_active') else '🔴 Inactive'}

### 🏥 Health Information
- **Health Condition:** {condition_str}

### 📈 Usage Statistics
- **Total messages:** {stats.get('message_count', 0) or 0} messages
- **Summaries:** {session_info.get('total_summaries', 0) or 0} times
- **Total Input Tokens:** {total_tokens_in:,} tokens
- **Total Output Tokens:** {total_tokens_out:,} tokens
- **Average Latency:** {latency_str} ms
- **First message:** {stats.get('first_message', 'N/A') or 'N/A'}
- **Last message:** {stats.get('last_message', 'N/A') or 'N/A'}

### 📋 Summary History
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
            return "📭 **No sessions in the system**"
        
        info_parts = ["# 📁 All Session Data\n"]
        
        for i, session in enumerate(sessions, 1):
            status_icon = "🟢" if session.get('is_active') else "🔴"
            condition = session.get('condition', 'Unspecified')[:30]
            
            session_info = f"""
## {i}. {status_icon} `{session['session_id']}`
- **Created:** {session.get('timestamp', 'N/A')}
- **Last activity:** {session.get('last_activity', 'N/A')}
- **Messages:** {session.get('total_messages', 0)} | **Summaries:** {session.get('total_summaries', 0)}
- **Condition:** {condition}
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
        return get_current_session_status(), "❌ **No active session**"
    
    if not condition.strip():
        return get_current_session_status(), "❌ **Please specify the health condition**"
    
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
        result = f"✅ **Health condition updated successfully!**\n\n📝 **New data:** {condition.strip()}"
        
        return status, result
    except Exception as e:
        logger.error(f"Error updating medical condition: {e}")
        return get_current_session_status(), f"❌ **Could not update health condition:** {str(e)}"

def process_chat_message(message: str, history: List, health_context: str) -> Tuple[List, str]:
    """Process chat message and return updated history."""
    if not app_state.current_session_id:
        history.append({"role": "assistant", "content": "❌ **Please create a new session before starting**"})
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

        return history, ""
    except Exception as e:
        logger.error(f"Error processing chat message: {e}")
        history.append({"role": "assistant", "content": f"❌ **Error:** {str(e)}"})
        return history, ""

def generate_health_notification(history: List) -> str:
    """Generate health notification based on conversation history."""
    # Simple notification generator - in production, use your prompt system
    notifications = [
        "💧 Remember to drink 8-10 glasses of water a day",
        "🚶‍♀️ Get up, walk, and stretch every hour",
        "😴 Get enough sleep, 7-8 hours a night",
        "🥗 Eat plenty of fruits and vegetables",
        "🧘‍♀️ Meditate or relax for 10 minutes a day",
        "💊 Don't forget to take your medication as prescribed"
    ]
    
    import random
    return random.choice(notifications)

def generate_manual_notification(history: List) -> List:
    """Generate manual health notification."""
    notification = generate_health_notification(history)
    history.append({
        "role": "assistant", 
        "content": f"🔔 **Health Notification:** {notification}"
    })
    return history

def force_update_summary() -> str:
    """Force update session summary."""
    if not app_state.current_session_id:
        return "❌ **No active session**"
    
    try:
        summary = chatbot_manager.summarize_session(app_state.current_session_id)
        return f"✅ **Summary successful!**\n\n📋 **Summary:** {summary}"
    except Exception as e:
        logger.error(f"Error forcing summary update: {e}")
        return f"❌ **Could not summarize:** {str(e)}"

def clear_session() -> Tuple[List, str, str, str, str]:
    """Clear current session."""
    if not app_state.current_session_id:
        return [], "", "❌ **No active session**", get_current_session_status(), ""
    
    try:
        old_session_id = app_state.current_session_id
        chatbot_manager.delete_session(app_state.current_session_id)
        app_state.reset()
        
        result = f"✅ **Session deleted successfully!**\n\n🗑️ **Deleted session:** `{old_session_id}`"
        return [], "", result, get_current_session_status(), ""
    except Exception as e:
        logger.error(f"Error clearing session: {e}")
        return [], "", f"❌ **Could not delete session:** {str(e)}", get_current_session_status(), ""

def clear_summary() -> str:
    """Clear session summary."""
    if not app_state.current_session_id:
        return "❌ **No active session**"
    
    try:
        with chatbot_manager._get_connection() as conn:
            conn.execute("""
                UPDATE sessions 
                SET summarized_history = NULL, total_summaries = 0 
                WHERE session_id = ?
            """, (app_state.current_session_id,))
            conn.commit()
        
        return f"✅ **Summary cleared successfully!**\n\n🗑️ **Session:** `{app_state.current_session_id}`"
    except Exception as e:
        logger.error(f"Error clearing summary: {e}")
        return f"❌ **Could not clear summary:** {str(e)}"

def export_session() -> str:
    """Export current session to JSON."""
    if not app_state.current_session_id:
        return "❌ **No active session**"
    
    try:
        json_data = chatbot_manager.export_session_json(app_state.current_session_id)
        
        return f"✅ **Export successful!**`"
    except Exception as e:
        logger.error(f"Error exporting session: {e}")
        return f"❌ **Could not export data:** {str(e)}"

def show_buttons():
    # Make the hidden buttons visible
    return gr.update(visible=True)

def hide_buttons():
    # Make the visible buttons hidden
    return gr.update(visible=False)

def clear_all_buttons():
    
    return [
        gr.update(visible=False),  # button1
        gr.update(visible=False),  # button2
        gr.update(visible=False)   # textbox
    ]

def set_button_loading():
    return gr.update(value="⏳ ประมวลผล...", variant="stop")
def reset_send_button():
    return gr.update(value="📤 ส่งข้อความ", variant="primary")
def reset_followup_button():
    return gr.update(value="🔔 สร้างการวิเคราะห์บทสนทนา", variant="secondary")
def reset_update_summary_button():
    return gr.update(value="📋 บังคับสรุปแชท (dev)", variant="secondary")

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
        background: linear-gradient(90deg,rgba(94, 160, 189, 1) 0%, rgba(82, 171, 118, 1) 50%, rgba(184, 170, 84, 1) 100%);
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
        background: linear-gradient(90deg,rgba(94, 160, 189, 1) 0%, rgba(82, 171, 118, 1) 50%, rgba(184, 170, 84, 1) 100%);
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
            # Session Status Display
            with gr.Column(
                elem_classes=["session-info"]
            ):
                gr.Markdown(
                    value="## ข้อมูล Session",
                )
                with gr.Row():
                    with gr.Column():
                        session_status = gr.Markdown(
                            value=get_current_session_status(), 
                        )
            with gr.Sidebar():
                with gr.Column():
                    new_session_btn = gr.Button("➕ Session ใหม่", variant="primary")
                    manage_session_btn = gr.Button("🗂️ จัดการ Session", variant="secondary")
                    edit_profile_btn = gr.Button("✏️ แก้ไขข้อมูลสุขภาพ", variant="secondary")
                    update_summary_btn = gr.Button("📋 บังคับสรุปแชท (dev)", variant="secondary")
                    
                # Session Details Accordion
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
                            export_btn = gr.Button("📤 ส่งออกข้อมูล Session (dev)", variant="secondary")
                            clear_chat_btn = gr.Button("🗑️ ล้าง Session", variant="secondary")
                            clear_summary_btn = gr.Button("📝 ล้างสรุป", variant="secondary")
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
                    elem_classes=["condition-box"]
                )
                update_condition_btn = gr.Button("💾 อัปเดตข้อมูลสุขภาพ", variant="primary")
                back_btn_1 = gr.Button("⏪ กลับไปยังแชท", variant="primary")

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
                outputs=[chatbot_window,session_management,summary_page]
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
                outputs=[chatbot_window,health_management,summary_page]
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
                outputs=[health_management]
            ).then(
                fn=refresh_session_list, 
                outputs=[session_dropdown]
            )

            update_summary_btn.click(
                fn=set_button_loading,
                outputs=[update_summary_btn]
            ).then(
                fn=clear_all_buttons,
                inputs=None,
                outputs=[chatbot_window,health_management,session_management]
            ).then(
                fn=show_buttons,
                inputs=None,
                outputs=[summary_page]
            ).then(
                fn=force_update_summary, 
                outputs=[summary_result]
            ).then(
                fn=refresh_session_list, 
                outputs=[session_dropdown]
            ).then(
                fn=reset_update_summary_button,
                outputs=[update_summary_btn]
            )

            send_btn.click(
                fn=set_button_loading,
                outputs=[send_btn]
            ).then(
                fn=process_chat_message,
                inputs=[msg, chatbot, health_context],
                outputs=[chatbot, msg]
            ).then(
                fn=refresh_session_list, 
                outputs=[session_dropdown]
            ).then(
                fn=reset_send_button,
                outputs=[send_btn]
            )

            msg.submit(
                fn=process_chat_message,
                inputs=[msg, chatbot, health_context],
                outputs=[chatbot, msg]
            ).then(
                fn=refresh_session_list, 
                outputs=[session_dropdown]
            )

            clear_chat_btn.click(
                fn=clear_session,
                outputs=[chatbot, msg, session_result, session_status, health_context]
            ).then(
                fn=refresh_session_list, 
                outputs=[session_dropdown]
            )

            clear_summary_btn.click(
                fn=clear_summary, 
                outputs=[session_result]
            ).then(
                fn=refresh_session_list, 
                outputs=[session_dropdown]
            )

            export_btn.click(
                fn=export_session, 
                outputs=[session_result]
            ) 

        with gr.Tab("Main App (English Ver.)"):
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
            with gr.Sidebar():
                with gr.Column():
                    new_session_btn = gr.Button("➕ New Session", variant="primary")
                    manage_session_btn = gr.Button("🗂️ Manage Session", variant="secondary")
                    edit_profile_btn = gr.Button("✏️ Edit Health Profile", variant="secondary")
                    update_summary_btn = gr.Button("📋 Force Chat Summary (dev)", variant="secondary")
                    
                # Session Details Accordion
                session_result = gr.Markdown(
                    value="**Waiting for update...**", 
                    elem_classes=["summary-box"],
                )
            with gr.Column(visible=False) as summary_page:
                        back_btn_2 = gr.Button("⏪ Back to Chat", variant="primary")
                        summary_result = gr.Markdown(
                            value="**Waiting for summary command...**", 
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
                            switch_btn = gr.Button("🔀 Load Session", variant="secondary")
                            refresh_btn = gr.Button("🔄 Refresh", variant="primary")
                        with gr.Row():
                            export_btn = gr.Button("📤 Export Session (dev)", variant="secondary")
                            clear_chat_btn = gr.Button("🗑️ Clear Session", variant="secondary")
                            clear_summary_btn = gr.Button("📝 Clear Summary", variant="secondary")
                        close_management_btn = gr.Button("❌ Close Session Management", variant="primary")

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
                back_btn_1 = gr.Button("⏪ Back to Chat", variant="primary")
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
                outputs=[chatbot_window,session_management,summary_page]
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
                outputs=[chatbot_window,health_management,summary_page]
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
                outputs=[health_management]
            ).then(
                fn=refresh_session_list, 
                outputs=[session_dropdown]
            )

            update_summary_btn.click(
                fn=set_button_loading,
                outputs=[update_summary_btn]
            ).then(
                fn=clear_all_buttons,
                inputs=None,
                outputs=[chatbot_window,health_management,session_management]
            ).then(
                fn=show_buttons,
                inputs=None,
                outputs=[summary_page]
            ).then(
                fn=force_update_summary, 
                outputs=[summary_result]
            ).then(
                fn=refresh_session_list, 
                outputs=[session_dropdown]
            ).then(
                fn=reset_update_summary_button,
                outputs=[update_summary_btn]
            )

            send_btn.click(
                fn=set_button_loading,
                outputs=[send_btn]
            ).then(
                fn=process_chat_message,
                inputs=[msg, chatbot, health_context],
                outputs=[chatbot, msg]
            ).then(
                fn=refresh_session_list, 
                outputs=[session_dropdown]
            ).then(
                fn=reset_send_button,
                outputs=[send_btn]
            )

            msg.submit(
                fn=process_chat_message,
                inputs=[msg, chatbot, health_context],
                outputs=[chatbot, msg]
            ).then(
                fn=refresh_session_list, 
                outputs=[session_dropdown]
            )

            clear_chat_btn.click(
                fn=clear_session,
                outputs=[chatbot, msg, session_result, session_status, health_context]
            ).then(
                fn=refresh_session_list, 
                outputs=[session_dropdown]
            )

            clear_summary_btn.click(
                fn=clear_summary, 
                outputs=[session_result]
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
        with gr.Tab("POC Description"):
            gr.Markdown("""
# 🧪 **Note for Proof of Concept (POC)**

## What is KaLLaM?

**KaLLaM** (กะหล่ำ) is a warm Thai AI healthcare caring female doctor chatbot. Providing medical guidance and mental health support in Thai language, helping patients take better care of themselves.

## Main Features

### 🩺 **Thai Medical Assistant**
- Speaks naturally in Thai with appropriate medical terminology
- Behaves like a friendly Thai female doctor/psychiatrist
- Understands Thai culture and communication style
- Uses respectful Thai addressing from the according local cultural context

### 💬 **Smart Conversations**
- Remembers what you talked about before (up to 200 responses)
- Can shoot follow-up questions
- Can handle long conversations without losing main context

### 🎯 **Health Assessment**
- Asks about sleep, eating, exercise, mood, and stress
- Digs deeper to find the real cause of problems
- Uses proven questioning techniques from real doctors
- Helps stubborn patients who don't want to change their habits

### 🧠 **Two AI Options**
- **SEA-Lion**: Specialized for Southeast Asian languages and context
- **Gemini**: Google's AI as backup option

## How It Works

### Basic Usage
```python
# Create the chatbot
doctor = KaLLaMChatbot(api_provider="sea_lion")

# Get medical advice
response = doctor.get_chatbot_response(
    chat_history=previous_messages,
    user_message="ปวดหัวมาก 3 วันแล้ว",
    health_status="patient_info"
)
```

### Key Capabilities
1. **Medical Consultation**: Analyzes symptoms and gives advice
2. **Follow-up Care**: Capable of asking "How are you doing now?" as poking conversation (automatically in future)
3. **Conversation Summary**: Remembers important details from long chats
4. **Progress Tracking**: Monitors if patients are following advice

## What Makes It Special

### 🇹🇭 **Thai-First Design**
- Built specifically for Thai patients
- Understands Thai medical culture
- Uses appropriate Thai's level of formality
- Considers Thai cultural context dynamics

### 💪 **Handles Difficult Patients**
- Doesn't give up when patients resist advice
- Uses medical facts to motivate change
- Handle behavior bhange & lifestyle interventions
- Stays calm with angry or frustrated patients
- Capable of using multiple professional psychologist techniques

## Real-World Applications

### 🏥 **Primary Healthcare**
- First-line medical consultation
- Symptom assessment and triage
- Basic health screening
- Referral guidance

### 🧘 **Mental Health Support**
- Stress and anxiety management
- Depression screening
- Emotional support
- Crisis prevention

### 🔄 **Chronic Care Management**
- Medication reminders
- Lifestyle coaching
- Symptom monitoring
- Patient education

## POC Success Goals

### Patient Experience
- Patients feel understood and cared for
- High engagement and return usage
- Improved health awareness
- Better compliance with medical advice

### Clinical Impact
- Accurate behavioral and medical guidance
- Early problem detection
- Better overall health outcomes

### System Performance
- Fast responses (under 3 seconds)
- Works reliably 24/7
- Handles multiple conversations
- Graceful error recovery

## Why This POC Matters

Problem Statement
Thailand’s Universal Coverage Scheme (UCS) offers highly affordable healthcare, often requiring only a 30 THB co-payment per visit. While this ensures accessibility, it has led to overutilization, with many individuals seeking hospital care for conditions that could be managed at home. This pattern places immense strain on medical professionals, who face overwhelming patient loads while receiving relatively low compensation, prompting many to seek employment in private institutions. The problem is further exacerbated by uneven resource distribution, with rural areas experiencing a significant shortage of medical personnel compared to urban centers. Additionally, lifestyle changes in rural communities, such as increased consumption of processed foods and reduced physical activity, have contributed to a rise in chronic diseases. Many individuals in these areas lack adequate health literacy, hindering their ability to manage their health effectively and leading to poor self-care practices.

Proposing KaLLaM Chatbot
To address these multifaceted challenges, we propose KaLLaM—a Thai Motivational Therapeutic Advisor chatbot powered by a large language model (LLM). KaLLaM is designed to act as an empathetic and supportive healthcare advisor, guiding patients in managing their health while reducing unnecessary hospital visits. By collecting medical history, symptoms, and lifestyle information, the chatbot provides users with personalized, evidence-based guidance tailored to their unique situation. This approach aims to bridge the gap between patients and healthcare providers, especially in underserved rural areas.

Main Features
KaLLaM offers a comprehensive approach to patient engagement. It provides actionable recommendations for both physical and mental well-being, using motivational and empathetic prompts to encourage healthy habits. The system can communicate in Thai or English, adapting its responses to cultural and linguistic context. Beyond advice, KaLLaM educates users about their own conditions, promoting self-care, disease prevention, and early intervention strategies—all in a patient-friendly conversational format. This feature is particularly crucial in rural areas, where access to healthcare professionals is limited, and health literacy is often low.

Potential Impact
By simulating a trusted healthcare adviser, KaLLaM empowers patients to take proactive steps toward managing their health. It has the potential to reduce unnecessary hospital visits, ease the workload of medical professionals, and optimize healthcare resource allocation. Furthermore, by enhancing health literacy and encouraging preventive care, the chatbot contributes to long-term improvements in public health and supports a more sustainable, equitable healthcare system across Thailand. In rural communities, where lifestyle changes and limited health education have led to increased chronic diseases, KaLLaM can play a pivotal role in reversing these trends and promoting healthier living.
                        """)

        
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
