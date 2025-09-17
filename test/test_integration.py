# tests/test_integration.py
"""
Minimal integration: orchestrator round with two clients.
"""
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import json
from FedGNN_advanced.orchestrator import Orchestrator
from FedGNN_advanced.privacy.auditor import PrivacyAuditor

def test_orchestrator_round():
    os.environ["AUDIT_SECRET"] = "integration_secret"
    td = TemporaryDirectory()
    run_dir = Path(td.name)
    auditor = PrivacyAuditor(run_dir / "audit")
    orch = Orchestrator(run_dir, run_dir / "store.db", expected_shapes={"w": (2,2)}, auditor=auditor)
    # fake clients
    import numpy as np
    client_keys = {"c1": b"key000000000000000000000000000000", "c2": b"key111111111111111111111111111111"}
    clients = []
    for cid in ["c1","c2"]:
        params = {"w": np.array([[1.0,1.0],[1.0,1.0]]).tolist()}
        payload = json.dumps({"params": params, "batch_size": 5, "local_dataset_size": 100}).encode("utf-8")
        from FedGNN_advanced.server.receiver import Receiver, PersistedStore
        mac = Receiver(PersistedStore(run_dir / "store.db")).compute_hmac(client_keys[cid], payload, 1)
        clients.append({"client_id": cid, "payload_bytes": payload, "mac_bytes": mac, "nonce": 1})
    res = orch.start_round(1, clients, client_keys)
    assert res["status"] == "ok"

if __name__ == "__main__":
    test_orchestrator_round()
    print("integration test passed")
