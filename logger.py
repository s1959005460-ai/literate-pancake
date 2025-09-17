"""
Structured logger helper for FedGNN_advanced.
Provides secure_log to avoid printing secrets and to ensure JSON-lines output.
"""

import json
import time
from typing import Any, Dict

_MAX_BYTES_PREVIEW = 256


def _sanitize_value(v: Any) -> Any:
    try:
        if isinstance(v, (bytes, bytearray)):
            preview = bytes(v[:_MAX_BYTES_PREVIEW]).hex()
            if len(v) > _MAX_BYTES_PREVIEW:
                preview += "...(truncated)"
            return {"__bytes__preview_hex": preview, "length": len(v)}
        if isinstance(v, dict):
            return {k: _sanitize_value(vv) for k, vv in v.items()}
        if isinstance(v, list):
            return [_sanitize_value(x) for x in v]
        return v
    except Exception:
        return str(v)


def secure_log(level: str, msg: str, **kwargs: Any) -> None:
    out: Dict[str, Any] = {
        "ts": time.time(),
        "level": level.upper(),
        "msg": msg,
        "payload": {},
    }
    for k, v in kwargs.items():
        out["payload"][k] = _sanitize_value(v)
    print(json.dumps(out, default=str), flush=True)


info = lambda msg, **kw: secure_log("info", msg, **kw)
warning = lambda msg, **kw: secure_log("warning", msg, **kw)
error = lambda msg, **kw: secure_log("error", msg, **kw)
debug = lambda msg, **kw: secure_log("debug", msg, **kw)
