# chatbot_demo.py
import gradio as gr
import logging
from datetime import datetime
from typing import List, Tuple, Optional
import os

from kallam.app.chatbot_manager import ChatbotManager

mgr = ChatbotManager(log_level="INFO")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# INLINE SVG for icons
CABBAGE_SVG = """
<svg width="128" height="128" viewBox="0 0 64 64" xmlns="http://www.w3.org/2000/svg"
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

# -----------------------
# Core handlers
# -----------------------
def _session_status(session_id: str) -> str:
    """Get current session status using mgr.get_session()"""
    if not session_id:
        return "🔴 **No Active Session** - Click **New Session** to start"
    
    try:
        # Use same method as simple app
        s = mgr.get_session(session_id) or {}
        ts = s.get("timestamp", "N/A")
        model = s.get("model_used", "N/A")
        total = s.get("total_messages", 0)
        saved_memories = s.get("saved_memories") or "General consultation"
        
        return f"""
🟢 **Session:** `{session_id[:8]}...`  
🏥 **Profile:** {saved_memories[:50]}{"..." if len(saved_memories) > 50 else ""}  
📅 **Created:** {ts}  
💬 **Messages:** {total}  
🤖 **Model:** {model}
""".strip()
    except Exception as e:
        logger.error(f"Error getting session status: {e}")
        return f"❌ **Error loading session:** {session_id[:8]}..."

def start_new_session(health_profile: str = ""):
    """Create new session using mgr - same as simple app"""
    try:
        sid = mgr.start_session(saved_memories=health_profile.strip() or None)
        status = _session_status(sid)
        
        # Initial welcome message
        welcome_msg = {
            "role": "assistant", 
            "content": """Hello! I'm KaLLaM 🌿, your caring AI health advisor 💖 

I can communicate in both **Thai** and **English**. I'm here to support your health and well-being with personalized advice. How are you feeling today? 😊

สวัสดีค่ะ! ฉันชื่อกะหล่ำ 🌿 เป็นที่ปรึกษาด้านสุขภาพ AI ที่จะคอยดูแลคุณ 💖 ฉันสามารถสื่อสารได้ทั้งภาษาไทยและภาษาอังกฤษ วันนี้รู้สึกยังไงบ้างคะ? 😊"""
        }
        
        history = [welcome_msg]
        result_msg = f"✅ **New Session Created Successfully!**\n\n🆔 Session ID: `{sid}`"
        if health_profile.strip():
            result_msg += f"\n🏥 **Health Profile:** Applied successfully"
            
        return sid, history, "", status, result_msg
    except Exception as e:
        logger.error(f"Error creating new session: {e}")
        return "", [], "", "❌ **Failed to create session**", f"❌ **Error:** {e}"

def send_message(user_msg: str, history: list, session_id: str):
    """Send message using mgr - same as simple app"""
    # Defensive: auto-create session if missing (same as simple app)
    if not session_id:
        logger.warning("No session found, auto-creating...")
        sid, history, _, status, _ = start_new_session("")
        history.append({"role": "assistant", "content": "🔄 **New session created automatically.** You can now continue chatting!"})
        return history, "", sid, status

    if not user_msg.strip():
        return history, "", session_id, _session_status(session_id)

    try:
        # Add user message
        history = history + [{"role": "user", "content": user_msg}]
        
        # Get bot response using mgr (same as simple app)
        bot_response = mgr.handle_message(
            session_id=session_id,
            user_message=user_msg
        )
        
        # Add bot response
        history = history + [{"role": "assistant", "content": bot_response}]
        
        return history, "", session_id, _session_status(session_id)
    
    except Exception as e:
        logger.error(f"Error processing message: {e}")
        error_msg = {"role": "assistant", "content": f"❌ **Error:** Unable to process your message. Please try again.\n\nDetails: {e}"}
        history = history + [error_msg]
        return history, "", session_id, _session_status(session_id)

def update_health_profile(session_id: str, health_profile: str):
    """Update health profile for current session using mgr's database access"""
    if not session_id:
        return "❌ **No active session**", _session_status(session_id)
    
    if not health_profile.strip():
        return "❌ **Please provide health information**", _session_status(session_id)

    try:
        # Use mgr's database path (same pattern as simple app would use)
        from kallam.infra.db import sqlite_conn
        with sqlite_conn(str(mgr.db_path)) as conn:
            conn.execute(
                "UPDATE sessions SET saved_memories = ?, last_activity = ? WHERE session_id = ?",
                (health_profile.strip(), datetime.now().isoformat(), session_id),
            )
        
        result = f"✅ **Health Profile Updated Successfully!**\n\n📝 **Updated Information:** {health_profile.strip()[:100]}{'...' if len(health_profile.strip()) > 100 else ''}"
        return result, _session_status(session_id)
    
    except Exception as e:
        logger.error(f"Error updating health profile: {e}")
        return f"❌ **Error updating profile:** {e}", _session_status(session_id)

def clear_session(session_id: str):
    """Clear current session using mgr"""
    if not session_id:
        return "", [], "", "🔴 **No active session to clear**", "❌ **No active session**"
    
    try:
        # Check if mgr has delete_session method, otherwise handle gracefully
        if hasattr(mgr, 'delete_session'):
            mgr.delete_session(session_id)
        else:
            # Fallback: just clear the session data if method doesn't exist
            logger.warning("delete_session method not available, clearing session state only")
        
        return "", [], "", "🔴 **Session cleared - Create new session to continue**", f"✅ **Session `{session_id[:8]}...` cleared successfully**"
    except Exception as e:
        logger.error(f"Error clearing session: {e}")
        return session_id, [], "", _session_status(session_id), f"❌ **Error clearing session:** {e}"

def force_summary(session_id: str):
    """Force summary using mgr (same as simple app)"""
    if not session_id:
        return "❌ No active session."
    try:
        if hasattr(mgr, 'summarize_session'):
            s = mgr.summarize_session(session_id)
            return f"📋 Summary updated:\n\n{s}"
        else:
            return "❌ Summarize function not available."
    except Exception as e:
        return f"❌ Failed to summarize: {e}"

def lock_inputs():
    """Lock inputs during processing (same as simple app)"""
    return gr.update(interactive=False), gr.update(interactive=False)

def unlock_inputs():
    """Unlock inputs after processing (same as simple app)"""
    return gr.update(interactive=True), gr.update(interactive=True)

# -----------------------
# UI with improved architecture and greenish cream styling
# -----------------------
def create_app() -> gr.Blocks:
    # Enhanced CSS with greenish cream color scheme and fixed positioning
    custom_css = """
    :root {
        --kallam-primary: #659435;
        --kallam-secondary: #5ea0bd;
        --kallam-accent: #b8aa54;
        --kallam-light: #f8fdf5;
        --kallam-dark: #2d3748;
        --kallam-cream: #f5f7f0;
        --kallam-green-cream: #e8f4e0;
        --kallam-border-cream: #d4e8c7;
        --shadow-soft: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        --shadow-medium: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05);
        --border-radius: 12px;
        --transition: all 0.3s ease;
    }

    .dark {
        --kallam-light: #1a2332;
        --kallam-dark: #ffffff;
        --kallam-cream: #2a3a2f;
        --kallam-green-cream: #243329;
        --kallam-border-cream: #3a4d3f;
        --shadow-soft: 0 4px 6px -1px rgba(255, 255, 255, 0.1), 0 2px 4px -1px rgba(255, 255, 255, 0.06);
        --shadow-medium: 0 10px 15px -3px rgba(255, 255, 255, 0.1), 0 4px 6px -2px rgba(255, 255, 255, 0.05);
    }

    .gradio-container {
        max-width: 100% !important;
        width: 100% !important;
        margin: 0 auto !important;
        min-height: 100vh;
    }

    .main-layout {
        display: flex !important;
        min-height: calc(100vh - 2rem) !important;
        gap: 1.5rem !important;
    }

    .fixed-sidebar {
        width: 320px !important;
        min-width: 320px !important;
        max-width: 320px !important;
        background: white !important;
        backdrop-filter: blur(10px) !important;
        border-radius: var(--border-radius) !important;
        border: 3px solid var(--kallam-primary) !important;
        box-shadow: var(--shadow-soft) !important;
        padding: 1.5rem !important;
        height: fit-content !important;
        position: sticky !important;
        top: 1rem !important;
        overflow: visible !important;
    }

    .main-content {
        flex: 1 !important;
        min-width: 0 !important;
    }

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
        background: var(--background-fill-secondary) !important;
        color: var(--body-text-color) !important;
        border: 1px solid var(--border-color-primary) !important;
    }

    .chat-container {
        background: var(--kallam-green-cream) !important;
        border-radius: var(--border-radius) !important;
        border: 2px solid var(--kallam-border-cream) !important;
        box-shadow: var(--shadow-medium) !important;
        overflow: hidden !important;
    }

    .session-status-container .markdown {
        margin: 0 !important;
        padding: 0 !important;
        font-size: 0.85rem !important;
        line-height: 1.4 !important;
        overflow-wrap: break-word !important;
        word-break: break-word !important;
    }

    @media (max-width: 1200px) {
        .main-layout {
            flex-direction: column !important;
        }
        
        .fixed-sidebar {
            width: 100% !important;
            min-width: 100% !important;
            max-width: 100% !important;
            position: static !important;
        }
    }
    """

    with gr.Blocks(
        title="🥬 KaLLaM - Thai Motivational Therapeutic Advisor",
        theme=gr.themes.Soft(primary_hue="green", secondary_hue="blue", neutral_hue="slate"), # type: ignore
        css=custom_css,
    ) as app:
        
        # State management - same as simple app
        session_id = gr.State(value="")
        
        # Header
        gr.HTML(f"""      
            <div class="kallam-header">
                <div style="display: flex; align-items: center; justify-content: flex-start; gap: 2rem; padding: 0 2rem;">
                    {CABBAGE_SVG}
                    <div style="text-align: left;">
                        <h1 style="text-align: left; margin: 0;">KaLLaM</h1>
                        <p class="kallam-subtitle" style="text-align: left; margin: 0.5rem 0 0 0;">Thai Motivational Therapeutic Advisor</p>
                    </div>
                </div>
            </div>
        """)

        # Main layout
        with gr.Row(elem_classes=["main-layout"]):
            # Sidebar with enhanced styling
            with gr.Column(scale=1, elem_classes=["fixed-sidebar"]):
                gr.HTML("""
                <div style="text-align: center; padding: 0.5rem 0 1rem 0;">
                    <h3 style="color: #659435; margin: 0; font-size: 1.2rem;">Controls</h3>
                    <p style="color: #666; margin: 0.25rem 0 0 0; font-size: 0.9rem;">Manage session and health profile</p>
                </div>
                """)
                
                with gr.Group():
                    new_session_btn = gr.Button("➕ New Session", variant="primary", size="lg", elem_classes=["btn", "btn-primary"])
                    health_profile_btn = gr.Button("👤 Custom Health Profile", variant="secondary", elem_classes=["btn", "btn-secondary"])
                    clear_session_btn = gr.Button("🗑️ Clear Session", variant="secondary", elem_classes=["btn", "btn-secondary"])
                
                # Hidden health profile section
                with gr.Column(visible=False) as health_profile_section:
                    gr.HTML('<div style="margin: 1rem 0;"><hr style="border: none; border-top: 1px solid var(--border-color-primary);"></div>')
                    
                    health_context = gr.Textbox(
                        label="🏥 Patient's Health Information",
                        placeholder="e.g., Patient's name, age, medical conditions (high blood pressure, diabetes), current symptoms, medications, lifestyle factors, mental health status...",
                        lines=5,
                        max_lines=8,
                        info="This information helps KaLLaM provide more personalized and relevant health advice. All data is kept confidential within your session."
                    )
                    
                    with gr.Row():
                        update_profile_btn = gr.Button("💾 Update Health Profile", variant="primary", elem_classes=["btn", "btn-primary"])
                        back_btn = gr.Button("⏪ Back", variant="secondary", elem_classes=["btn", "btn-secondary"])
                
                gr.HTML('<div style="margin: 1rem 0;"><hr style="border: none; border-top: 1px solid var(--border-color-primary);"></div>')
                
                # Session status
                session_status = gr.Markdown(value="🔄 **Initializing...**")
                
            # Main chat area
            with gr.Column(scale=3, elem_classes=["main-content"]):
                gr.HTML("""
                <div style="text-align: center; padding: 1rem 0;">
                    <h2 style="color: #659435; margin: 0; font-size: 1.5rem;">💬 Health Consultation Chat</h2>
                    <p style="color: #666; margin: 0.5rem 0 0 0;">Chat with your AI health advisor in Thai or English</p>
                </div>
                """)

                chatbot = gr.Chatbot(
                    label="Chat with KaLLaM",
                    height=500,
                    show_label=False,
                    type="messages",
                    elem_classes=["chat-container"]
                )

                with gr.Row():
                    with gr.Column(scale=5):
                        msg = gr.Textbox(
                            label="Message",
                            placeholder="Ask about your health in Thai or English...",
                            lines=2,
                            max_lines=4,
                            show_label=False,
                            elem_classes=["chat-container"]
                        )
                    with gr.Column(scale=1, min_width=120):
                        send_btn = gr.Button("➤", variant="primary", size="lg", elem_classes=["btn", "btn-primary"])

        # Result display
        result_display = gr.Markdown(visible=True)

        # Footer
        gr.HTML("""
        <div style="
            position: fixed; bottom: 0; left: 0; right: 0;
            background: linear-gradient(135deg, var(--kallam-secondary) 0%, var(--kallam-primary) 100%);
            color: white; padding: 0.75rem 1rem; text-align: center; font-size: 0.8rem;
            box-shadow: 0 -4px 6px -1px rgba(0, 0, 0, 0.1); z-index: 1000;
            border-top: 1px solid rgba(255,255,255,0.2);
        ">
            <div style="max-width: 1400px; margin: 0 auto; display: flex; justify-content: center; align-items: center; flex-wrap: wrap; gap: 1.5rem;">
                <span style="font-weight: 600;">Built with ❤️ by:</span>
                <div style="display: flex; flex-direction: column; align-items: center; gap: 0.2rem;">
                    <span style="font-weight: 500;">👨‍💻 Nopnatee Trivoravong</span>
                    <div style="display: flex; gap: 0.5rem; font-size: 0.75rem;">
                        <span>📧 nopnatee.triv@gmail.com</span>
                        <span>•</span>
                        <a href="https://github.com/Nopnatee" target="_blank" style="color: rgba(255,255,255,0.9); text-decoration: none;">GitHub</a>
                    </div>
                </div>
                <span style="color: rgba(255,255,255,0.7);">|</span>
                <div style="display: flex; flex-direction: column; align-items: center; gap: 0.2rem;">
                    <span style="font-weight: 500;">👨‍💻 Khamic Srisutrapon</span>
                    <div style="display: flex; gap: 0.5rem; font-size: 0.75rem;">
                        <span>📧 khamic.sk@gmail.com</span>
                        <span>•</span>
                        <a href="https://github.com/Khamic672" target="_blank" style="color: rgba(255,255,255,0.9); text-decoration: none;">GitHub</a>
                    </div>
                </div>
                <span style="color: rgba(255,255,255,0.7);">|</span>
                <div style="display: flex; flex-direction: column; align-items: center; gap: 0.2rem;">
                    <span style="font-weight: 500;">👩‍💻 Napas Siripala</span>
                    <div style="display: flex; gap: 0.5rem; font-size: 0.75rem;">
                        <span>📧 millynapas@gmail.com</span>
                        <span>•</span>
                        <a href="https://github.com/kaoqueri" target="_blank" style="color: rgba(255,255,255,0.9); text-decoration: none;">GitHub</a>
                    </div>
                </div>
            </div>
        </div>
        """)

        # ====== EVENT HANDLERS - Same pattern as simple app ======
        
        # Auto-initialize on page load (same as simple app)
        def _init():
            sid, history, _, status, note = start_new_session("")
            return sid, history, status, note
        
        app.load(
            fn=_init,
            inputs=None,
            outputs=[session_id, chatbot, session_status, result_display]
        )

        # New session
        new_session_btn.click(
            fn=lambda: start_new_session(""),
            inputs=None,
            outputs=[session_id, chatbot, msg, session_status, result_display]
        )

        # Show/hide health profile section
        def show_health_profile():
            return gr.update(visible=True)
        
        def hide_health_profile():
            return gr.update(visible=False)

        health_profile_btn.click(
            fn=show_health_profile,
            outputs=[health_profile_section]
        )

        back_btn.click(
            fn=hide_health_profile,
            outputs=[health_profile_section]
        )

        # Update health profile  
        update_profile_btn.click(
            fn=update_health_profile,
            inputs=[session_id, health_context],
            outputs=[result_display, session_status]
        ).then(
            fn=hide_health_profile,
            outputs=[health_profile_section]
        )

        # Send message with lock/unlock pattern (inspired by simple app)
        send_btn.click(
            fn=lock_inputs,
            inputs=None,
            outputs=[send_btn, msg],
            queue=False,   # lock applies instantly
        ).then(
            fn=send_message,
            inputs=[msg, chatbot, session_id],
            outputs=[chatbot, msg, session_id, session_status],
        ).then(
            fn=unlock_inputs,
            inputs=None,
            outputs=[send_btn, msg],
            queue=False,
        )

        # Enter/submit flow: same treatment
        msg.submit(
            fn=lock_inputs,
            inputs=None,
            outputs=[send_btn, msg],
            queue=False,
        ).then(
            fn=send_message,
            inputs=[msg, chatbot, session_id],
            outputs=[chatbot, msg, session_id, session_status],
        ).then(
            fn=unlock_inputs,
            inputs=None,
            outputs=[send_btn, msg],
            queue=False,
        )

        # Clear session
        clear_session_btn.click(
            fn=clear_session,
            inputs=[session_id],
            outputs=[session_id, chatbot, msg, session_status, result_display]
        )

    return app

def main():
    app = create_app()
    app.launch(
        share=True,
        server_name="0.0.0.0",
        server_port=7860,
        debug=False,
        show_error=True,
        inbrowser=True,
    )

if __name__ == "__main__":
    main()