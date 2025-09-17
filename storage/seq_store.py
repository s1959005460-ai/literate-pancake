# File: storage/seq_store.py
"""
Redis-backed sequence store for replay protection and masked share persistence.

Functions:
 - check_and_set_seq(client_id, seq): atomically set seq if greater than previous
 - store_masked_share(round_id, client_id, payload): store base64 payload in hash
 - fetch_round_shares(round_id): get all shares for round (decode at consumer)
"""
from __future__ import annotations

import aioredis
import os
import base64
import logging
from typing import Dict, Optional

logger = logging.getLogger(__name__)

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
redis = None  # will be initialized with init_redis


async def init_redis() -> None:
    global redis
    if redis is None:
        redis = await aioredis.from_url(REDIS_URL, decode_responses=False)


async def check_and_set_seq(client_id: str, seq: int) -> bool:
    """
    Atomically sets seq if seq > existing value. Returns True if set, False if replay or stale.
    Uses Lua script for atomic compare-and-set.
    """
    await init_redis()
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
    res = await redis.eval(lua, keys=[key], args=[str(seq)])
    return res == 1


async def store_masked_share(round_id: int, client_id: str, payload: bytes, ttl_seconds: int = 3600) -> None:
    """
    Store payload (raw bytes) under redis hash for the round.
    """
    await init_redis()
    key = f"masked:{round_id}"
    # store base64 to preserve binary
    await redis.hset(key, client_id, base64.b64encode(payload))
    await redis.expire(key, ttl_seconds)


async def fetch_round_shares(round_id: int) -> Dict[str, bytes]:
    await init_redis()
    key = f"masked:{round_id}"
    items = await redis.hgetall(key)
    # hgetall returns bytes for both key & value; decode accordingly
    out = {}
    for k, v in items.items():
        ck = k.decode() if isinstance(k, bytes) else k
        cv = v
        if isinstance(cv, bytes):
            try:
                import base64 as _b64
                out[ck] = _b64.b64decode(cv)
            except Exception:
                out[ck] = cv
        else:
            out[ck] = cv
    return out
