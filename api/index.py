from fastapi import FastAPI
from mangum import Mangum
import hashlib
import time
import numpy as np
from collections import OrderedDict

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
    time.sleep(1)
    return f"Moderation result for: {query}"

class Cache:
    def __init__(self):
        self.store = OrderedDict()
        self.embeddings = {}

    def cleanup(self):
        now = time.time()
        expired = []
        for k, (v, t) in self.store.items():
            if now - t > CACHE_TTL:
                expired.append(k)
        for k in expired:
            del self.store[k]
            del self.embeddings[k]

    def get(self, query):
        self.cleanup()
        key = md5_hash(query)

        if key in self.store:
            value, t = self.store[key]
            self.store.move_to_end(key)
            return value, True, key

        query_emb = fake_embedding(query)
        for k, emb in self.embeddings.items():
            if cosine_similarity(query_emb, emb) > SIM_THRESHOLD:
                value, t = self.store[k]
                self.store.move_to_end(k)
                return value, True, k

        return None, False, key

    def set(self, query, value):
        key = md5_hash(query)

        if len(self.store) >= MAX_CACHE_SIZE:
            oldest = next(iter(self.store))
            del self.store[oldest]
            del self.embeddings[oldest]

        self.store[key] = (value, time.time())
        self.embeddings[key] = fake_embedding(query)

    def size(self):
        return len(self.store)

cache = Cache()

total = 0
hits = 0
misses = 0

@app.post("/")
async def main(data: dict):
    global total, hits, misses
    total += 1

    start = time.time()
    query = data.get("query")

    answer, cached, key = cache.get(query)

    if not cached:
        misses += 1
        answer = fake_llm_response(query)
        cache.set(query, answer)
    else:
        hits += 1

    latency = int((time.time() - start) * 1000)

    return {
        "answer": answer,
        "cached": cached,
        "latency": latency,
        "cacheKey": key
    }

@app.get("/analytics")
async def analytics():
    baseline = (total * TOKENS_PER_REQUEST * COST_PER_MILLION) / 1_000_000
    actual = (misses * TOKENS_PER_REQUEST * COST_PER_MILLION) / 1_000_000
    savings = baseline - actual
    hit_rate = hits / total if total else 0

    return {
        "hitRate": round(hit_rate, 2),
        "totalRequests": total,
        "cacheHits": hits,
        "cacheMisses": misses,
        "cacheSize": cache.size(),
        "costSavings": round(savings, 2),
        "savingsPercent": round(hit_rate * 100, 2),
        "strategies": [
            "exact match",
            "semantic similarity",
            "LRU eviction",
            "TTL expiration"
        ]
    }

handler = Mangum(app)
