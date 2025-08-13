import gradio as gr
import json
import logging
from datetime import datetime
from typing import List, Tuple, Optional, Dict, Any
from data_manager import ChatbotManager

# Initialize the chatbot manager
chatbot_manager = ChatbotManager()
current_session_id = None

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global state management
class AppState:
    def __init__(self):
        self.current_session_id = None
        self.message_count = 0
        self.auto_summary_threshold = 5
        self.auto_notification_threshold = 10
    
    def reset(self):
        self.current_session_id = None
        self.message_count = 0

app_state = AppState()

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
🟢 **Session ปัจจุบัน:** `{app_state.current_session_id}`  
📅 **สร้างเมื่อ:** {session_info.get('timestamp', 'N/A')}  
💬 **จำนวนข้อความ:** {stats.get('message_count', 0) or 0} ข้อความ  
📋 **จำนวนสรุป:** {session_info.get('total_summaries', 0) or 0} ครั้ง  
🏥 **สภาวะสุขภาพ:** {condition_str}  
⚡ **Latency เฉลี่ย:** {latency_str} ms  
🔧 **Model:** {session_info.get('model_used', 'N/A')}
        """.strip()
    except Exception as e:
        logger.error(f"Error getting session status: {e}")
        return f"❌ **Error:** ไม่สามารถโหลดข้อมูล Session {app_state.current_session_id}"

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

def create_app() -> gr.Blocks:
    """Create the Gradio application."""
    custom_css = """
    .gradio-container {
        max-width: 1400px !important;
        margin: 0 auto !important;
    }
    .tab-nav {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
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
        background: #f8f9fa;
        border-left: 4px solid #667eea;
    }
    .session-info {
        border-radius: 8px;
        padding: 15px;
        margin: 10px 0;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
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
        title="🌟 ระบบการให้คำแนะนำทางสุขภาพ - Enhanced AI Chatbot",
        theme=gr.themes.Soft(),
        css=custom_css,
    ) as app:

        gr.Markdown("""
        # 🌟 ระบบการให้คำแนะนำทางสุขภาพ (Enhanced AI Chatbot)
        
        ระบบนี้ใช้เทคโนโลยี AI ขั้นสูงในการให้คำแนะนำด้านสุขภาพและการบำบัดพฤติกรรม พร้อมคุณสมบัติ:
        
        🔄 **ระบบสรุปอัตโนมัติ:** สรุปการสนทนาทุก 5 ข้อความ  
        🔔 **การแจ้งเตือนอัตโนมัติ:** แจ้งเตือนสุขภาพทุก 10 ข้อความ  
        💾 **Session Management:** จัดเก็บและจัดการ session การสนทนา  
        🏥 **Medical Tracking:** ติดตามสภาวะสุขภาพแต่ละ session  
        📊 **Analytics:** วิเคราะห์สถิติการใช้งานอย่างละเอียด
        """)

        # Session Status Display
        session_status = gr.Markdown(
            value=get_current_session_status(), 
            elem_classes=["session-info"]
        )

        # Session Management Section
        gr.Markdown("## 🗂️ การจัดการ Session")

        with gr.Row():
            with gr.Column(scale=3):
                session_ids = get_session_list()
                initial_session = session_ids[0] if session_ids else None

                session_dropdown = gr.Dropdown(
                    choices=session_ids,
                    value=initial_session,
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
            export_btn = gr.Button("📤 ส่งออกข้อมูล", variant="secondary")

        # Session Details Accordion
        with gr.Accordion("📊 ข้อมูลรายละเอียด Session", open=False):
            session_result = gr.Markdown(
                value="**กำลังรอการอัปเดต...**", 
                elem_classes=["summary-box"]
            )
            session_info_display = gr.Markdown(
                value="", 
                elem_classes=["summary-box"]
            )

        gr.Markdown("---")

        # Health Management Section
        gr.Markdown("## 🏥 การจัดการสภาวะสุขภาพ")

        with gr.Row():
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

        gr.Markdown("---")

        # Chat Interface Section
        gr.Markdown("## 💬 แชทบอทให้คำปรึกษาสุขภาพ")
        
        gr.Markdown("""
        **🔔 วิธีใช้การแจ้งเตือน:**
        - พิมพ์ "แจ้งเตือนสุขภาพ" หรือ "สร้างการแจ้งเตือน" เพื่อขอการแจ้งเตือนทันที
        - ระบบจะแจ้งเตือนอัตโนมัติทุก 10 ข้อความ
        - ใช้ปุ่ม "🔔 สร้างการแจ้งเตือน" เพื่อสร้างการแจ้งเตือนแบบ manual
        """)

        chatbot = gr.Chatbot(
            label="💭 การสนทนากับ AI",
            height=500,
            show_label=True,
            type="messages",
            elem_classes=["chat-container"],
            avatar_images=("👤", "🤖")
        )

        with gr.Row():
            with gr.Column(scale=4):
                msg = gr.Textbox(
                    label="💬 พิมพ์ข้อความของคุณ",
                    placeholder="พิมพ์คำถามหรือข้อมูลสุขภาพ... (พิมพ์ 'แจ้งเตือนสุขภาพ' เพื่อขอการแจ้งเตือน)",
                    lines=2,
                    max_lines=8,
                )
            with gr.Column(scale=1):
                send_btn = gr.Button("📤 ส่งข้อความ", variant="primary")
                notification_btn = gr.Button("🔔 สร้างการแจ้งเตือน", variant="secondary")
                update_summary_btn = gr.Button("📋 บังคับสรุปแชท", variant="secondary")

        # Control Buttons
        with gr.Row():
            clear_chat_btn = gr.Button("🗑️ ล้าง Session", variant="secondary")
            clear_summary_btn = gr.Button("📝 ล้างสรุป", variant="secondary")

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

        session_info_btn.click(
            fn=get_session_info, 
            outputs=[session_info_display]
        )

        all_sessions_btn.click(
            fn=get_all_sessions_info, 
            outputs=[session_info_display]
        )

        update_condition_btn.click(
            fn=update_medical_condition,
            inputs=[health_context],
            outputs=[session_status, session_result]
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

        notification_btn.click(
            fn=generate_manual_notification, 
            inputs=[chatbot], 
            outputs=[chatbot]
        )

        update_summary_btn.click(
            fn=force_update_summary, 
            outputs=[session_result]
        )

        clear_chat_btn.click(
            fn=clear_session,
            outputs=[chatbot, msg, session_result, session_status, health_context]
        )

        clear_summary_btn.click(
            fn=clear_summary, 
            outputs=[session_result]
        )

        export_btn.click(
            fn=export_session, 
            outputs=[session_result]
        )

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
