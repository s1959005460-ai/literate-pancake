"""
Lightweight message classes for in-process async protocol simulation.
"""

from dataclasses import dataclass
from typing import Any, Dict, Tuple, List


@dataclass
class RegisterClient:
    client_id: str
    param_shapes: Dict[str, Tuple[int, ...]]


@dataclass
class SharePackage:
    sender: str
    recipient: str
    share: Tuple[int, int]


@dataclass
class MaskedUpdate:
    sender: str
    masked_params: Dict[str, Any]


@dataclass
class UnmaskRequest:
    requester: str
    missing_clients: List[str]


@dataclass
class UnmaskShare:
    sender: str
    missing_client: str
    share: Tuple[int, int]
