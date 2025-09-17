import pytest
import asyncio
import numpy as np
import torch

from FedGNN_advanced.privacy import bonawitz_protocol as protocol
from FedGNN_advanced.learner import Learner
from FedGNN_advanced.models.gcn_lora import GCNWithLoRA
from FedGNN_advanced.privacy import dp as dp_module

# small deterministic synthetic dataset
def make_simple_dataset(seed=0):
    import torch
    from torch_geometric.data import Data
    torch.manual_seed(seed)
    x = torch.randn(10, 4)
    row = np.arange(0, 9)
    col = row + 1
    edge_index = np.stack([np.concatenate([row, col]), np.concatenate([col, row])], axis=0)
    edge_index = torch.from_numpy(edge_index).long()
    y = torch.randint(0, 2, (10,), dtype=torch.long)
    return [Data(x=x, edge_index=edge_index, y=y)]

@pytest.mark.parametrize("dropout_client", [None, "client_3"])
def test_bonawitz_all_online_or_with_dropout(dropout_client):
    proto = protocol.ProtocolServer(threshold_fraction=0.6)
    model_template = GCNWithLoRA(in_channels=4, hidden_channels=8, out_channels=2, num_layers=2, dropout=0.0, lora_r=4, lora_alpha=1.0, task="node")
    # create 4 clients
    clients = []
    for i in range(4):
        cid = f"client_{i+1}"
        ds = make_simple_dataset(seed=i)
        cl = Learner(client_id=cid, model_template=model_template, dataset=ds, server_adapter=proto, device="cpu", local_epochs=1, batch_size=4, lr=0.01, dp_config={"noise_multiplier":0.0})
        cl.prepare_privacy_engine()
        cl.prepare_and_send_shares(n_shares=4)
        clients.append(cl)
    proto.drain_all_inboxes_and_collect_shares()
    # construct a trivial global state
    global_state = {k: v.cpu().clone() for k, v in model_template.state_dict().items()}
    # simulate clients sending masked updates; optionally make one client dropout
    for cl in clients:
        if dropout_client is not None and cl.id == dropout_client:
            continue
        cl.local_train_and_mask(global_state)
    missing = proto.missing_clients()
    if dropout_client is not None:
        assert dropout_client in missing
        # alive clients produce unmask shares from their received shares
        for alive in [c for c in clients if c.id != dropout_client]:
            recs = getattr(alive, "_received_shares", [])
            for sp in recs:
                us = protocol.UnmaskShare(sender=alive.id, missing_client=sp.sender, share=sp.share)
                proto.collect_unmask_share(us)
    # compute aggregate (should work; if dropout and insufficient shares exist, some won't be reconstructable)
    agg = proto.compute_aggregate()
    assert isinstance(agg, dict)
    # basic property: aggregated entries are numpy arrays
    for v in agg.values():
        assert isinstance(v, np.ndarray)
