# File: FedGNN_advanced/privacy/bonawitz_server.py
"""
Bonawitz-style secure aggregation server helpers (simplified).

Features:
 - Per-share HMAC verification using X25519-derived keys (crypto_utils.derive_shared_key)
 - Sequence numbers for replay protection (simple in-memory store; production: use persistent KV)
 - Process masked shares and dispatch to aggregator

Rationale:
 - Audit found placeholder MAC relying on TLS (unsafe). Implement application-level MAC/AEAD.
 - Use server's X25519 private key fetched from KMS (he_wrapper.kms_client_factory).
 - Keep code robust to tampered payloads and ensure structured logging (no secrets in logs).

Notes:
 - This module focuses on message verification + orchestration, not the full Bonawitz spec (e.g., Shamir share reconstruction).
 - For Shamir reconstruction and full dropout handling, integrate with secure_agg.py helpers.
"""
from __future__ import annotations

import base64
import logging
from typing import Dict, Tuple, Any

from FedGNN_advanced.crypto.crypto_utils import derive_shared_key, hmac_verify
from FedGNN_advanced.crypto.he_wrapper import kms_client_factory

logger = logging.getLogger(__name__)
if not logger.handlers:
    import sys
    h = logging.StreamHandler(sys.stdout)
    h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    logger.addHandler(h)
logger.setLevel("INFO")

# Simple in-memory store for last seen seq per client (production: persist to KV)
_last_seen_seq: Dict[str, int] = {}

# Simple store for accumulated masked shares per round (round -> list of bytes)
_masked_store: Dict[int, Dict[str, bytes]] = {}


class InvalidMAC(Exception):
    pass


class ReplayDetected(Exception):
    pass


def get_server_x25519_priv() -> bytes:
    """
    Fetch server's X25519 private key bytes from KMS.
    This code expects he_wrapper.kms_client_factory to return a client with get_private_key_bytes.
    In production, this should not expose raw private bytes unless policy allows; preferred approach:
    - Use KMS to perform sign/derive operations or provide secure handle.
    """
    kms = kms_client_factory()
    # 'bonawitz-server-x25519' is a recommended key_id; adjust to your deployment.
    key_id = "bonawitz-server-x25519"
    priv = kms.get_private_key_bytes(key_id)
    if not isinstance(priv, (bytes, bytearray)):
        raise RuntimeError("KMS did not return raw private key bytes for X25519")
    return bytes(priv)


def is_replay(client_id: str, seq: int) -> bool:
    last = _last_seen_seq.get(client_id)
    if last is None:
        return False
    return seq <= last


def update_seq(client_id: str, seq: int) -> None:
    _last_seen_seq[client_id] = seq


def verify_and_store_masked_share(client_id: str, round_id: int, payload: Dict[str, Any]) -> None:
    """
    Verify HMAC on payload and store the masked share.

    payload expected fields:
      - seq: int
      - masked_share: base64 str
      - sender_pub: base64 str (client's x25519 pubkey)
      - mac: base64 str (HMAC-SHA256 over seq||masked_share)
    """
    seq = int(payload.get("seq", -1))
    if seq < 0:
        logger.warning("Invalid seq for client %s", client_id)
        raise ValueError("invalid seq")

    if is_replay(client_id, seq):
        logger.warning("Replay detected for client %s seq=%d", client_id, seq)
        raise ReplayDetected("replay detected")

    try:
        masked_share_b64 = payload["masked_share"]
        sender_pub_b64 = payload["sender_pub"]
        mac_b64 = payload["mac"]
    except KeyError:
        raise ValueError("Missing required payload fields")

    masked_share = base64.b64decode(masked_share_b64)
    sender_pub = base64.b64decode(sender_pub_b64)
    mac = base64.b64decode(mac_b64)

    # derive shared key between server and this client
    server_priv = get_server_x25519_priv()
    shared_key = derive_shared_key(server_priv, sender_pub, info=b"bonawitz-mac")

    # message to MAC: seq (8 bytes big endian) + masked_share
    seq_bytes = int(seq).to_bytes(8, "big")
    msg = seq_bytes + masked_share

    if not hmac_verify(shared_key, msg, mac):
        logger.warning("MAC verification failed for client %s", client_id)
        raise InvalidMAC("mac verification failed")

    # store masked share for the round
    round_store = _masked_store.setdefault(round_id, {})
    if client_id in round_store:
        logger.warning("Duplicate share for client %s in round %s", client_id, round_id)
        # it's okay to ignore duplicate, but log
    round_store[client_id] = masked_share

    # update last seen seq for replay protection
    update_seq(client_id, seq)
    logger.info("Stored masked share for client=%s round=%s seq=%s", client_id, round_id, seq)


def get_masked_shares_for_round(round_id: int) -> Dict[str, bytes]:
    return _masked_store.get(round_id, {})


def clear_round(round_id: int) -> None:
    _masked_store.pop(round_id, None)
