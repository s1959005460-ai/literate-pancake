"""
FedGNLLightningModule

Lightweight PyTorch Lightning integration helper.
This file uses lazy imports so your repo can still be used without Lightning installed.
"""

from __future__ import annotations
from typing import Any, Optional

try:
    import pytorch_lightning as pl  # type: ignore
    _PL_AVAILABLE = True
except Exception:
    _PL_AVAILABLE = False

from .. import logger


if _PL_AVAILABLE:
    class FedGNLLightningModule(pl.LightningModule):
        def __init__(self, fedgnn_model: Any, privacy_engine: Optional[Any] = None):
            super().__init__()
            self.model = fedgnn_model
            self.privacy_engine = privacy_engine

        def training_step(self, batch, batch_idx):
            if self.privacy_engine:
                with self.privacy_engine:
                    loss = self.model.training_step(batch)
                    # privacy_engine.step() or equivalent update should be invoked by the privacy_engine's context manager
            else:
                loss = self.model.training_step(batch)
            return loss

        def configure_optimizers(self):
            return self.model.configure_optimizers()
else:
    # fallback stub to avoid import errors in environments without pytorch_lightning
    class FedGNLLightningModule:
        def __init__(self, *args, **kwargs):
            logger.secure_log("warning", "PyTorch Lightning is not installed; FedGNLLightningModule is a stub")
            raise RuntimeError("pytorch_lightning is required for FedGNLLightningModule")
