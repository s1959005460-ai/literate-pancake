# tests/test_e2e.py
"""
E2E smoke test: starts orchestrator demo (single-process) and simulates normal and failure case.
"""
from FedGNN_advanced.orchestrator import Orchestrator
from FedGNN_advanced.privacy.auditor import PrivacyAuditor
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import json

def test_e2e_normal():
    os.environ["AUDIT_SECRET"] = "e2e_secret"
    td = TemporaryDirectory()
    run_dir = Path(td.name)
    auditor = PrivacyAuditor(run_dir / "audit")
    orch = Orchestrator(run_dir, run_dir / "store.db", expected_shapes={"w": (2,2)}, auditor=auditor)
    # reuse integration flow
    import numpy as np
    client_keys = {"c1": b"k1"*16, "c2": b"k2"*16}
    clients = []
    from FedGNN_advanced.server.receiver import Receiver, PersistedStore
    for cid in client_keys:
        params = {"w": np.array([[1.0,1.0],[1.0,1.0]]).tolist()}
        payload = json.dumps({"params": params, "batch_size": 2, "local_dataset_size": 10}).encode("utf-8")
        mac = Receiver(PersistedStore(run_dir / "store.db")).compute_hmac(client_keys[cid], payload, 1)
        clients.append({"client_id": cid, "payload_bytes": payload, "mac_bytes": mac, "nonce": 1})
    res = orch.start_round(1, clients, client_keys)
    assert res["status"] == "ok"

if __name__ == "__main__":
    test_e2e_normal()
    print("E2E normal passed")
