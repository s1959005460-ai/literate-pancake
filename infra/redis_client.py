# infra/redis_client.py
"""
Redis helper: connection pool, atomic compare-and-set for seq (replay protection),
and safe list/hash operations for masked shares.
"""
from __future__ import annotations
import aioredis
import os
import logging
import base64

logger = logging.getLogger("redis_client")
logger.setLevel(logging.INFO)

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
_redis = None


async def get_redis() -> aioredis.Redis:
    global _redis
    if _redis is None:
        _redis = aioredis.from_url(REDIS_URL, encoding=None, decode_responses=False, max_connections=50)
    return _redis


async def check_and_set_seq(client_id: str, seq: int) -> bool:
    """
    Lua script atomic: if seq > cur then set and return 1 else 0
    """
    r = await get_redis()
    key = f"seq:{client_id}"
    lua = """
    local cur = tonumber(redis.call('get', KEYS[1]) or '-1')
    local new = tonumber(ARGV[1])
    if new > cur then
        redis.call('set', KEYS[1], ARGV[1])
        return 1
    end
    return 0
    """
    res = await r.eval(lua, keys=[key], args=[str(seq)])
    return res == 1


async def store_masked_share(round_id: int, client_id: str, payload: bytes, ttl: int = 3600):
    r = await get_redis()
    key = f"masked:{round_id}"
    await r.hset(key, client_id, base64.b64encode(payload))
    await r.expire(key, ttl)


async def fetch_round_shares(round_id: int):
    r = await get_redis()
    key = f"masked:{round_id}"
    items = await r.hgetall(key)
    out = {}
    for k, v in items.items():
        if isinstance(k, bytes):
            k = k.decode()
        if isinstance(v, bytes):
            v = base64.b64decode(v)
        out[k] = v
    return out
