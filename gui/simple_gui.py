# simple_chat_app.py
import gradio as gr
from datetime import datetime

# Your existing manager
from kallam.app.chatbot_manager import ChatbotManager

mgr = ChatbotManager(log_level="INFO")

# -----------------------
# Core handlers
# -----------------------
def _session_status(session_id: str) -> str:
    if not session_id:
        return "🔴 No active session. Click **New Session**."
    s = mgr.get_session(session_id) or {}
    ts = s.get("timestamp", "N/A")
    model = s.get("model_used", "N/A")
    total = s.get("total_messages", 0)
    return (
        f"🟢 **Session:** `{session_id}`  \n"
        f"📅 Created: {ts}  \n"
        f"🤖 Model: {model}  \n"
        f"💬 Messages: {total}"
    )

def start_new_session(saved_memories: str = ""):
    """Create a brand-new session and return clean UI state."""
    sid = mgr.start_session(saved_memories=saved_memories or None)
    status = _session_status(sid)
    history = []  # gr.Chatbot(messages) shape: list[dict(role, content)]
    return sid, history, "", status, "✅ New session created."

def send_message(user_msg: str, history: list, session_id: str):
    """Append user turn, get bot reply, and return updated history."""
    if not session_id:
        # Defensive: if somehow no session, auto-create one
        sid, history, _, status, _ = start_new_session("")
        history.append({"role": "assistant", "content": "New session spun up automatically. Proceed."})
        return history, "", sid, status

    if not user_msg.strip():
        return history, "", session_id, _session_status(session_id)

    # User turn
    history = history + [{"role": "user", "content": user_msg}]
    # Bot turn (your manager handles persistence)
    bot = mgr.handle_message(session_id=session_id, user_message=user_msg)
    history = history + [{"role": "assistant", "content": bot}]

    return history, "", session_id, _session_status(session_id)

def force_summary(session_id: str):
    if not session_id:
        return "❌ No active session."
    try:
        s = mgr.summarize_session(session_id)
        return f"📋 Summary updated:\n\n{s}"
    except Exception as e:
        return f"❌ Failed to summarize: {e}"
    
def lock_inputs():
    # disable send + textbox
    return gr.update(interactive=False), gr.update(interactive=False)

def unlock_inputs():
    # re-enable send + textbox
    return gr.update(interactive=True), gr.update(interactive=True)

# -----------------------
# Gradio app
# -----------------------
def create_app():
    with gr.Blocks(title="Minimal Therapeutic Chat Sessions") as demo:
        gr.Markdown("# Minimal Chat Sessions • clean and boring but try me.")

        session_id = gr.State(value="")
        with gr.Row():
            status_md = gr.Markdown(value="🔄 Initializing session...")
        saved_memories = gr.Textbox(
            label="เนื้อหาเกี่ยวกับคุณ (optional)",
            placeholder="กด ➕ session ใหม่เพื่อใช้งาน e.g., อายุ, เพศ, นิสัย",
            lines=2,
        )
        new_btn = gr.Button("➕ session ใหม่", variant="primary")
        # summarize_btn = gr.Button("📋 Summarize", variant="secondary")

        chat = gr.Chatbot(label="Chat", type="messages", height=420)
        with gr.Row():
            msg = gr.Textbox(label="Message box", placeholder="พิมพ์ข้อความ", lines=1, scale=9)
            send = gr.Button("↩", variant="primary", scale=1, min_width=40)

        result_md = gr.Markdown(visible=True)

        # -------- wiring --------
        # On page load: create a fresh session
        def _init():
            sid, history, _, status, note = start_new_session("")
            return sid, history, status, note
        demo.load(_init, inputs=None, outputs=[session_id, chat, status_md, result_md])

        # New session button
        def _new(saved):
            sid, history, _, status, note = start_new_session(saved or "")
            return sid, history, "", status, note
        new_btn.click(_new, inputs=[saved_memories], outputs=[session_id, chat, msg, status_md, result_md])

        # Click flow: lock -> send -> unlock
        send.click(
            fn=lock_inputs,
            inputs=None,
            outputs=[send, msg],
            queue=False,   # lock applies instantly
        ).then(
            fn=send_message,
            inputs=[msg, chat, session_id],
            outputs=[chat, msg, session_id, status_md],
        ).then(
            fn=unlock_inputs,
            inputs=None,
            outputs=[send, msg],
            queue=False,
        )

        # Enter/submit flow: same treatment
        msg.submit(
            fn=lock_inputs,
            inputs=None,
            outputs=[send, msg],
            queue=False,
        ).then(
            fn=send_message,
            inputs=[msg, chat, session_id],
            outputs=[chat, msg, session_id, status_md],
        ).then(
            fn=unlock_inputs,
            inputs=None,
            outputs=[send, msg],
            queue=False,
        )

    return demo

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
