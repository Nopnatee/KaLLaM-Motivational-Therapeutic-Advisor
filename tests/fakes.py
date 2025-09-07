class FakeOrchestrator:
    def __init__(self):
        self.calls = {"flags": 0, "translate": 0, "response": 0, "summarize": 0}

    def get_flags_from_supervisor(self, user_message: str):
        self.calls["flags"] += 1
        # pretend Thai if message contains "TH:"
        return {"translate": "thai" if "TH:" in user_message else False}

    def get_translation(self, message: str, flags, translation_type: str):
        self.calls["translate"] += 1
        # trivial identity “translation”
        return message.replace("TH:", "").strip()

    def get_response(self, chat_history, user_message, flags, chain_of_thoughts, summarized_histories):
        self.calls["response"] += 1
        # echo bot + minimal reasoning payload
        return {"final_output": f"BOT:{user_message}", "meta": {"len_hist": len(chat_history)}}

    def summarize_history(self, chat_history, eng_summaries):
        self.calls["summarize"] += 1
        return f"SUMMARY({len(chat_history)} msgs, {len(eng_summaries)} prev)"
