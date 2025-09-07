# infra/token_counter.py
class TokenCounter:
    def __init__(self, capacity: int = 1000):
        self.cache = {}
        self.capacity = capacity

    def count(self, text: str) -> int:
        h = hash(text)
        if h in self.cache: return self.cache[h]
        n = max(1, len(text.split()))
        if len(self.cache) >= self.capacity:
            self.cache = dict(list(self.cache.items())[self.capacity // 2 :])
        self.cache[h] = n
        return n
