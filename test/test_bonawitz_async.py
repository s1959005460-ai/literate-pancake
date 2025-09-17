import asyncio
import numpy as np
import pytest
from FedGNN_advanced.privacy.bonawitz_async import AsyncServer, AsyncClient
from FedGNN_advanced import constants

@pytest.mark.asyncio
async def test_bonawitz_async_all_online_sum_equals_expected():
    server = AsyncServer(threshold_fraction=float(constants.DEFAULTS.get("TEST_BONAWITZ_THRESHOLD", 0.6)))
    param_shapes = {"w": (2, 2), "b": (2,)}

    clients = []
    for i in range(4):
        cid = f"client_{i}"
        c = AsyncClient(cid, server, param_shapes)
        await c.register()
        clients.append(c)

    for c in clients:
        await c.prepare_and_send_shares()

    for c in clients:
        await c.collect_initial_shares(timeout=0.05)

    updates = {}
    for idx, c in enumerate(clients):
        upd = {"w": np.ones((2, 2), dtype=np.float32) * (idx + 1), "b": np.ones((2,), dtype=np.float32) * (idx + 1)}
        updates[c.client_id] = upd
        await c.send_masked_update(upd)

    agg = await server.compute_aggregate()

    expected = {}
    for p in updates[clients[0].client_id].keys():
        s = None
        for u in updates.values():
            if s is None:
                s = np.array(u[p], dtype=np.float64)
            else:
                s += np.array(u[p], dtype=np.float64)
        expected[p] = s.astype(np.float32)

    for k in expected.keys():
        assert np.allclose(agg[k], expected[k], atol=1e-5)


@pytest.mark.asyncio
async def test_bonawitz_async_with_dropout_and_reconstruction():
    server = AsyncServer(threshold_fraction=float(constants.DEFAULTS.get("TEST_BONAWITZ_THRESHOLD", 0.6)))
    param_shapes = {"w": (2, 2), "b": (2,)}

    clients = []
    for i in range(5):
        cid = f"client_{i}"
        c = AsyncClient(cid, server, param_shapes)
        await c.register()
        clients.append(c)

    for c in clients:
        await c.prepare_and_send_shares()

    for c in clients:
        await c.collect_initial_shares(timeout=0.05)

    for c in clients:
        if c.client_id == "client_3":
            c.will_send_masked_update = False

    updates = {}
    for idx, c in enumerate(clients):
        upd = {"w": np.ones((2, 2), dtype=np.float32) * (idx + 1), "b": np.ones((2,), dtype=np.float32) * (idx + 1)}
        updates[c.client_id] = upd
        await c.send_masked_update(upd)

    missing = server.missing_clients()
    assert "client_3" in missing

    for c in clients:
        us = await c.provide_unmask_shares(missing)
        for item in us:
            await server.collect_unmask_share(item)

    agg = await server.compute_aggregate()

    expected = None
    for cid, u in updates.items():
        if cid == "client_3":
            continue
        if expected is None:
            expected = {p: np.array(v, dtype=np.float64) for p, v in u.items()}
        else:
            for p, v in u.items():
                expected[p] += np.array(v, dtype=np.float64)
    for p in expected:
        assert np.allclose(agg[p], expected[p].astype(np.float32), atol=1e-5)
