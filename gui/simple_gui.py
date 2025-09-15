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

# -----------------------
# Gradio app
# -----------------------
def create_app():
    with gr.Blocks(title="Minimal Chat Sessions") as demo:
        gr.Markdown("# Minimal Chat Sessions • clean and boring, just like production likes it. 🧼🧪")

        session_id = gr.State(value="")
        with gr.Row():
            status_md = gr.Markdown(value="🔄 Initializing session...")
        with gr.Row():
            saved_memories = gr.Textbox(
                label="Session context (optional)",
                placeholder="e.g., child age, focus behavior, constraints...",
                lines=2,
            )
            new_btn = gr.Button("➕ New Session", variant="primary")
            summarize_btn = gr.Button("📋 Summarize", variant="secondary")

        chat = gr.Chatbot(label="Chat", type="messages", height=420)
        with gr.Row():
            msg = gr.Textbox(placeholder="Type your message", lines=2)
            send = gr.Button("Send", variant="primary")

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

        # Send handling (button)
        send.click(
            send_message,
            inputs=[msg, chat, session_id],
            outputs=[chat, msg, session_id, status_md],
        )
        # Send handling (enter)
        msg.submit(
            send_message,
            inputs=[msg, chat, session_id],
            outputs=[chat, msg, session_id, status_md],
        )

        # Optional: force summary
        summarize_btn.click(force_summary, inputs=[session_id], outputs=[result_md])

    return demo

def main():
    app = create_app()
    app.launch(
        share=False,
        server_name="0.0.0.0",
        server_port=7860,
        debug=False,
        show_error=True,
        inbrowser=True,
    )

if __name__ == "__main__":
    main()
