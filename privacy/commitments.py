"""
Commitment helpers: Merkle root and simple Pedersen-like commitment stubs.

These utilities are intentionally small; for production use audited implementations.
"""

import hashlib
from typing import List, Tuple


def sha256(b: bytes) -> bytes:
    return hashlib.sha256(b).digest()


def merkle_root(leaves: List[bytes]) -> bytes:
    if not leaves:
        return sha256(b"")
    nodes = [sha256(l) for l in leaves]
    while len(nodes) > 1:
        next_nodes = []
        for i in range(0, len(nodes), 2):
            if i + 1 < len(nodes):
                next_nodes.append(sha256(nodes[i] + nodes[i + 1]))
            else:
                next_nodes.append(nodes[i])
        nodes = next_nodes
    return nodes[0]
