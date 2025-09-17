"""
Hugging Face style integration example.

Provides a minimal FedGNNTransformer wrapper integrating a GNN + HF transformer.
This module uses lazy imports and will raise informative errors when dependencies missing.
"""

from __future__ import annotations
from typing import Any

try:
    from transformers import PreTrainedModel, AutoModel  # type: ignore
    _HF_AVAILABLE = True
except Exception:
    _HF_AVAILABLE = False

from .. import logger

if _HF_AVAILABLE:
    class FedGNNTransformer(PreTrainedModel):
        def __init__(self, config, gnn_model: Any):
            # PreTrainedModel requires __init__ signature; keep minimal to illustrate integration
            super().__init__(config)
            self.gnn = gnn_model
            self.transformer = AutoModel.from_config(config)  # or AutoModel.from_pretrained
            # fusion layer: simple linear for demo
            import torch.nn as nn
            hidden_dim = getattr(config, "hidden_size", 256)
            self.fusion_layer = nn.Linear(hidden_dim + hidden_dim, hidden_dim)

        def federated_forward(self, graph_data, text_data):
            # graph_data -> embeddings (user supplies appropriate preprocessing)
            graph_embeddings = self.gnn(graph_data)  # expects tensor
            text_embeddings = self.transformer(**text_data).last_hidden_state.mean(dim=1)
            combined = torch.cat([graph_embeddings, text_embeddings], dim=-1)
            return self.fusion_layer(combined)
else:
    class FedGNNTransformer:
        def __init__(self, *args, **kwargs):
            logger.secure_log("warning", "Transformers not installed; FedGNNTransformer is a stub")
            raise RuntimeError("transformers package required for FedGNNTransformer")

