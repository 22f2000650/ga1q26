from fastapi import FastAPI
import hashlib
import time
import numpy as np
from collections import OrderedDict
import threading

app = FastAPI()

CACHE_TTL = 86400
MAX_CACHE_SIZE = 1500
SIM_THRESHOLD = 0.95

TOKENS_PER_REQUEST = 300
COST_PER_MILLION = 0.40

def md5_hash(text):
    return hashlib.md5(text.encode()).hexdigest()

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def fake_embedding(text):
    np.random.seed(abs(hash(text)) % (10**8))
    return np.random.rand(128)

def fake_llm_response(query):
    time.sleep(1.2)
    return f"Moderation result for: {query}"

class IntelligentCache:
    def __init__(self):
        self.cache = OrderedDict()
        self.embeddings = {}
        self.lock = threading.Lock()

    def cleanup(self):
        now = time.time()
        expired = []
        for k, (v, t) in self.cache.items():
            if now - t > CACHE_TTL:
                expired.append(k)
        for k in expired:
            self.delete(k)

    def get(self, query):
        with self.lock:
            self.cleanup()
            key = md5_hash(query)

            if key in self.cache:
                value, timestamp = self.cache[key]
                self.cache.move_to_end(key)
                return value, True, key

            query_emb = fake_embedding(query)
            for k, emb in self.embeddings.items():
                if cosine_similarity(query_emb, emb) > SIM_THRESHOLD:
                    value, timestamp = self.cache[k]
                    self.cache.move_to_end(k)
                    return value, True, k

            return None, False, key

    def set(self, query, value):
        with self.lock:
            key = md5_hash(query)

            if len(self.cache) >= MAX_CACHE_SIZE:
                oldest = next(iter(self.cache))
                self.delete(oldest)

            self.cache[key] = (value, time.time())
            self.embeddings[key] = fake_embedding(query)

    def delete(self, key):
        if key in self.cache:
            del self.cache[key]
        if key in self.embeddings:
            del self.embeddings[key]

    def size(self):
        return len(self.cache)

class Analytics:
    def __init__(self):
        self.total = 0
        self.hits = 0
        self.misses = 0

    def record(self, hit):
        self.total += 1
        if hit:
            self.hits += 1
        else:
            self.misses += 1

    def metrics(self, cache_size):
        baseline = (self.total * TOKENS_PER_REQUEST * COST_PER_MILLION) / 1_000_000
        actual = (self.misses * TOKENS_PER_REQUEST * COST_PER_MILLION) / 1_000_000
        savings = baseline - actual
        hit_rate = self.hits / self.total if self.total else 0

        return {
            "hitRate": round(hit_rate, 2),
            "totalRequests": self.total,
            "cacheHits": self.hits,
            "cacheMisses": self.misses,
            "cacheSize": cache_size,
            "costSavings": round(savings, 2),
            "savingsPercent": round(hit_rate * 100, 2),
            "strategies": [
                "exact match",
                "semantic similarity",
                "LRU eviction",
                "TTL expiration"
            ]
        }

cache = IntelligentCache()
analytics = Analytics()

@app.post("/")
async def main_endpoint(data: dict):
    start = time.time()
    query = data.get("query")

    answer, cached, key = cache.get(query)

    if not cached:
        answer = fake_llm_response(query)
        cache.set(query, answer)

    latency = int((time.time() - start) * 1000)
    analytics.record(cached)

    return {
        "answer": answer,
        "cached": cached,
        "latency": latency,
        "cacheKey": key
    }

@app.get("/analytics")
async def analytics_endpoint():
    return analytics.metrics(cache.size())
handler = app
