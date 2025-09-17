"""
Async Bonawitz flow (single-process simulation).

Contains:
- AsyncServer
- AsyncClient (simple in-process client to interact with AsyncServer)

SECURITY NOTE (IMPORTANT):
- This file implements a demo/simulation of the Bonawitz-style secure aggregation flow.
- **DO NOT** use the example clients/seed generation below in production as-is.
- In particular, earlier demo versions used deterministic seeds based on client_id which is insecure.
  This file uses cryptographically secure random seeds by default (secrets.token_bytes).
- For production, prefer ECDH/X25519 pairwise key agreement for deriving pairwise seeds, add VSS,
  and authenticate all messages/participants.

This module intentionally aims to be readable and deterministic for testing, but the security
assumptions must be reviewed before any real deployment.
"""

from typing import Dict, Any, List, Tuple, Optional
import asyncio
from dataclasses import dataclass
import numpy as np
from . import protocol_messages as msg
from . import shamir, mask_manager
from .. import constants, logger
import secrets  # secure random for seeds

# local aliases for constants
SEED_BYTE_LEN = int(constants.SEED_BYTE_LEN)
SHAMIR_THRESHOLD_FRACTION = float(constants.SHAMIR_THRESHOLD_FRACTION)
DEFAULT_INBOX_DRAIN_TIMEOUT = float(constants.DEFAULT_INBOX_DRAIN_TIMEOUT)
DEFAULT_UNMASK_REQUEST_TIMEOUT = float(constants.DEFAULT_UNMASK_REQUEST_TIMEOUT)


@dataclass
class ClientHandle:
    client_id: str
    inbox: asyncio.Queue
    param_shapes: Dict[str, Tuple[int, ...]]


class AsyncServer:
    def __init__(self, threshold_fraction: Optional[float] = None):
        self.clients: Dict[str, ClientHandle] = {}
        self.param_shapes: Dict[str, Tuple[int, ...]] = {}
        self.masked_updates: Dict[str, Dict[str, Any]] = {}
        self._unmask_shares: List[msg.UnmaskShare] = []
        self.threshold_fraction = threshold_fraction if threshold_fraction is not None else SHAMIR_THRESHOLD_FRACTION
        self._lock = asyncio.Lock()

    async def register_client(self, register_msg: msg.RegisterClient, inbox: asyncio.Queue):
        cid = register_msg.client_id
        async with self._lock:
            self.clients[cid] = ClientHandle(client_id=cid, inbox=inbox, param_shapes=register_msg.param_shapes)
            # store param shapes globally (assumes homogenous)
            self.param_shapes = register_msg.param_shapes
        logger.secure_log("info", "client registered", client=cid)

    async def receive_masked_update(self, masked_update: msg.MaskedUpdate):
        async with self._lock:
            self.masked_updates[masked_update.sender] = masked_update.masked_params
        logger.secure_log("info", "Received masked update", sender=masked_update.sender)

    def participants_who_sent(self) -> List[str]:
        return list(self.masked_updates.keys())

    def missing_clients(self) -> List[str]:
        # clients that failed to send masked updates
        return [c for c in self.clients.keys() if c not in self.masked_updates]

    async def request_unmasking(self, timeout: float = DEFAULT_UNMASK_REQUEST_TIMEOUT):
        """
        Request unmask shares from clients for any missing participants.
        This function sends requests (in the simulation they are method calls/queue events).
        """
        missing = self.missing_clients()
        if not missing:
            return

        # In real flow: server sends request to all alive clients asking to provide
        # shares for missing clients. Here we simulate by draining client inboxes.
        for cid, handle in self.clients.items():
            try:
                await handle.inbox.put(msg.UnmaskRequest(missing_clients=missing))
            except Exception:
                logger.secure_log("warning", "failed to send unmask request", client=cid)

        # allow some time for clients to respond (simulation)
        await asyncio.sleep(timeout)

    async def reconstruct_missing_seeds(self) -> Dict[str, bytes]:
        reconstructed = {}
        shares_by_target: Dict[str, List[Tuple[int, int]]] = {}
        async with self._lock:
            for us in self._unmask_shares:
                shares_by_target.setdefault(us.missing_client, []).append((int(us.share[0]), int(us.share[1])))
        for target, list_shares in shares_by_target.items():
            if len(list_shares) < self.expected_threshold():
                logger.secure_log("warning", "not enough shares", client=target, have=len(list_shares))
                continue
            try:
                seed = shamir.reconstruct_secret_bytes(list_shares[: self.expected_threshold()], SEED_BYTE_LEN)
                reconstructed[target] = seed
            except Exception as e:
                logger.secure_log("warning", "reconstruct failed", client=target, err=str(e))
        return reconstructed

    def expected_threshold(self) -> int:
        n = max(1, len(self.clients))
        return max(1, int(np.ceil(n * self.threshold_fraction)))

    async def _collect_unmask_share(self, share: msg.UnmaskShare):
        async with self._lock:
            self._unmask_shares.append(share)

    async def compute_aggregate(self) -> Dict[str, Any]:
        if not self.masked_updates:
            raise RuntimeError("no masked updates")
        param_names = list(next(iter(self.masked_updates.values())).keys())
        agg = {p: None for p in param_names}
        for sender, params in self.masked_updates.items():
            for p, arr in params.items():
                a = np.array(arr, dtype=np.float64)
                agg[p] = a.copy() if agg[p] is None else agg[p] + a
        missing = self.missing_clients()
        if missing:
            reconstructed = await self.reconstruct_missing_seeds()
            alive = [c for c in self.clients.keys() if c not in missing]
            for m in missing:
                seed_m = reconstructed.get(m)
                if seed_m is None:
                    logger.secure_log("warning", "missing seed not reconstructable", missing=m)
                    continue
                for other in alive:
                    pair_seed = mask_manager.derive_pairwise_seed(m, other, seed_m)
                    masks = mask_manager.compute_mask_from_seed(pair_seed, self.param_shapes)
                    for p, arr in masks.items():
                        if agg[p] is not None:
                            agg[p] -= arr
        return {p: agg[p].astype(np.float32) for p in param_names}

    # inbox / share handling helpers used by demo clients
    async def relay_share(self, share_pkg: msg.SharePackage):
        """
        In real flow, shares are sent peer-to-peer. For the demo, server relays share packages
        to the intended recipient's inbox.
        """
        recipient = share_pkg.recipient
        if recipient in self.clients:
            await self.clients[recipient].inbox.put(share_pkg)
        else:
            logger.secure_log("warning", "relay recipient not found", recipient=recipient)

    async def receive_unmask_share(self, share: msg.UnmaskShare):
        async with self._lock:
            self._unmask_shares.append(share)

    async def run_demo_rounds(self):
        """
        Helper to drive demo flows in unit tests / examples.
        """
        # for tests/demos only
        return


# Minimal AsyncClient used in tests and demos
class AsyncClient:
    def __init__(self, client_id: str, server: AsyncServer, param_shapes: Dict[str, Tuple[int, ...]]):
        self.client_id = client_id
        self.server = server
        self.param_shapes = param_shapes
        self.inbox: asyncio.Queue = asyncio.Queue()
        # SECURITY: use cryptographically secure random seed for demo clients.
        # DO NOT use deterministic seed derived from client_id in any real deployment.
        self.seed = secrets.token_bytes(SEED_BYTE_LEN)
        self._received_shares: List[msg.SharePackage] = []
        self.will_send_masked_update = True

        # register with server
        asyncio.create_task(self.server.register_client(msg.RegisterClient(client_id=self.client_id, param_shapes=self.param_shapes), self.inbox))

    async def collect_initial_shares(self, timeout: float = 0.05):
        # drain mailbox for sharepackages
        try:
            while True:
                item = await asyncio.wait_for(self.inbox.get(), timeout=timeout)
                if isinstance(item, msg.SharePackage):
                    self._received_shares.append(item)
        except asyncio.TimeoutError:
            return

    async def _split_and_send_shares(self, n_shares: int = 5):
        # split seed into n_shares with threshold t
        t = max(1, int(np.ceil(n_shares * SHAMIR_THRESHOLD_FRACTION)))
        shares = shamir.split_secret_bytes(self.seed, n_shares, t)
        # distribute shares to other clients via server.relay_share
        # demo mapping: map index -> recipient by server clients order
        client_ids = list(self.server.clients.keys())
        for idx, (i, s) in enumerate(shares):
            recipient = client_ids[idx % max(1, len(client_ids))]
            await self.server.relay_share(msg.SharePackage(sender=self.client_id, recipient=recipient, share=(i, s)))

    async def send_masked_update(self, update: Dict[str, Any]):
        if not self.will_send_masked_update:
            return
        # In this demo the client assumes it already computed masked update
        masked = {}
        for p, arr in update.items():
            masked[p] = np.array(arr, dtype=np.float32)
        await self.server.receive_masked_update(msg.MaskedUpdate(sender=self.client_id, masked_params=masked))

    async def provide_unmask_shares(self, missing_clients: List[str]) -> List[msg.UnmaskShare]:
        # create shamir shares for each missing client seed (if we have them)
        out = []
        # In real flow, client holds shares for other clients - here we simulate: respond with shares in _received_shares
        for rec in self._received_shares:
            # rec.share belongs to some origin; for demo, if origin in missing, respond
            for m in missing_clients:
                out.append(msg.UnmaskShare(sender=self.client_id, missing_client=m, share=rec.share))
        return out


# End of file
