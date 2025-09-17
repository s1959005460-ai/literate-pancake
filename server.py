# server.py
import asyncio
import logging
import random
from typing import List, Dict, Any, Optional

logger = logging.getLogger("FedServer")
logger.setLevel(logging.INFO)

class AsyncFedServer:
    def __init__(self, protocol_server, aggregator, client_rpc,
                 per_client_timeout: float = 30.0, max_retries: int = 3,
                 blacklist_threshold: int = 5, share_wait_seconds: float = 2.0,
                 backoff_base: float = 1.0, backoff_factor: float = 2.0, backoff_jitter: float = 0.1):
        self.protocol_server = protocol_server
        self.aggregator = aggregator
        self.client_rpc = client_rpc
        self.per_client_timeout = float(per_client_timeout)
        self.max_retries = int(max_retries)
        self.blacklist_threshold = int(blacklist_threshold)
        self.share_wait_seconds = float(share_wait_seconds)
        self.backoff_base = float(backoff_base)
        self.backoff_factor = float(backoff_factor)
        self.backoff_jitter = float(backoff_jitter)

    async def _call_with_retry(self, client_id: str):
        attempts = 0
        while attempts <= self.max_retries:
            attempts += 1
            try:
                coro = self.client_rpc.send_train(client_id)
                result = await asyncio.wait_for(coro, timeout=self.per_client_timeout)
                return result
            except asyncio.TimeoutError:
                logger.warning("Timeout calling %s (attempt %d/%d)", client_id, attempts, self.max_retries+1)
            except Exception as e:
                logger.exception("Error calling %s: %s", client_id, e)
            # backoff before next attempt
            if attempts <= self.max_retries:
                backoff = self.backoff_base * (self.backoff_factor ** (attempts - 1))
                jitter = backoff * self.backoff_jitter * (random.random() * 2 - 1)
                wait = max(0.0, backoff + jitter)
                await asyncio.sleep(wait)
        logger.error("Exhausted retries for %s", client_id)
        try:
            ci = self.protocol_server.clients.get(client_id)
            if ci:
                ci.blacklist_score += 1
                if ci.blacklist_score >= self.blacklist_threshold:
                    ci.approved = False
                    logger.warning("Client %s blacklisted", client_id)
        except Exception:
            logger.exception("Failed update blacklist for %s", client_id)
        return None

    async def run_round(self, round_id: int, participants: List[str], validation_loader=None, server_noise_std: float = 0.0, aggregator_topo_kwargs: Optional[dict] = None):
        logger.info("Starting round %s participants=%d", round_id, len(participants))
        tasks = [self._call_with_retry(p) for p in participants]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        received = []
        for res in results:
            if res is None:
                continue
            if isinstance(res, Exception):
                logger.exception("Task error: %s", res)
                continue
            try:
                client_id, masked_params, mac, nonce, fmt = res
            except Exception:
                logger.warning("Unexpected RPC result: %r", res)
                continue
            try:
                ok = self.protocol_server.submit_masked_update(client_id, masked_params, mac, nonce, format=fmt)
                if ok:
                    received.append(client_id)
            except Exception:
                logger.exception("submit_masked_update failed for %s", client_id)

        missing = set(participants) - set(received)
        if missing:
            logger.warning("Missing after collection: %s", missing)
        if self.share_wait_seconds > 0 and missing:
            logger.info("Waiting %.2fs for unmask shares", self.share_wait_seconds)
            await asyncio.sleep(self.share_wait_seconds)

        reconstructed = []
        for t in list(missing):
            try:
                secret = self.protocol_server._reconstruct_secret(t)
                if secret is not None:
                    reconstructed.append(t)
                    missing.discard(t)
            except Exception:
                logger.exception("reconstruct failed for %s", t)

        server_agg = self.protocol_server.compute_aggregate(server_add_noise=(server_noise_std>0.0), noise_std=server_noise_std)

        try:
            agg_result = self.aggregator.apply_aggregate(self.aggregator.global_state or {}, {"server": server_agg}, topo_kwargs=aggregator_topo_kwargs)
        except Exception:
            logger.exception("aggregator.apply_aggregate failed")
            raise

        eval_metrics = None
        if validation_loader is not None:
            eval_metrics = await asyncio.get_event_loop().run_in_executor(None, self.evaluate, agg_result, validation_loader)
            logger.info("Eval metrics: %s", eval_metrics)

        logger.info("Round %s finished: received=%d missing=%d reconstructed=%d", round_id, len(received), len(missing), len(reconstructed))
        return {"round_id": round_id, "aggregate": agg_result, "missing": list(missing), "reconstructed": reconstructed, "eval_metrics": eval_metrics}

    def evaluate(self, aggregate: Dict[str, Any], validation_loader):
        """
        Default evaluation is not implemented — must be overridden by caller.
        """
        raise NotImplementedError("Override AsyncFedServer.evaluate to perform model evaluation")
