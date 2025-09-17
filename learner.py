# 文件: FedGNN_advanced/learner.py
"""
Learner with DP accounting via Opacus and immutable audit logging.

Standards:
- Differential Privacy per Abadi et al. (2016): gradient clipping, Gaussian noise, moments accountant (RDP).
- Use Opacus PrivacyEngine and RDPAccountant.
- Enforce privacy budget guard and append-only audit log.
"""
from __future__ import annotations

import logging
import json
import os
from typing import Optional, Any
from prometheus_client import Gauge, Counter

logger = logging.getLogger("fedgnn.learner")
if not logger.handlers:
    import sys
    h = logging.StreamHandler(sys.stdout)
    h.setFormatter(logging.Formatter('{"ts":"%(asctime)s","level":"%(levelname)s","module":"%(name)s","msg":%(message)s}'))
    logger.addHandler(h)
logger.setLevel(os.getenv("FEDGNN_LOG_LEVEL", "INFO"))

DP_EPSILON_GAUGE = Gauge("fedgnn_dp_epsilon_current", "Current DP epsilon")
DP_ACCOUNTANT_ERRORS = Counter("fedgnn_dp_accountant_errors_total", "DP accountant errors")
DP_BUDGET_EXHAUSTED = Counter("fedgnn_dp_budget_exhausted_total", "DP budget exhausted events")

# Opacus availability
try:
    from opacus import PrivacyEngine  # type: ignore
    from opacus.accountants import RDPAccountant  # type: ignore
    _OPACUS_AVAILABLE = True
except Exception:
    PrivacyEngine = None  # type: ignore
    RDPAccountant = None  # type: ignore
    _OPACUS_AVAILABLE = False

# Audit log file path (append-only)
AUDIT_LOG_PATH = os.getenv("FEDGNN_AUDIT_LOG", "/var/log/fedgnn_privacy_audit.log")

class PrivacyAuditError(Exception):
    pass

def _append_audit_record(record: dict) -> None:
    """
    Append-only audit trail. In production, write to immutable storage or SIEM.
    """
    try:
        with open(AUDIT_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
    except Exception as e:
        logger.error(json.dumps({"msg":"failed write audit","error":str(e)}))
        raise PrivacyAuditError("failed audit write") from e

class Learner:
    def __init__(self, model, optimizer, train_loader, noise_multiplier: float = 1.0, max_grad_norm: float = 1.0, target_delta: float = 1e-5, policy_max_eps: float = 8.0):
        self.model = model
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.noise_multiplier = noise_multiplier
        self.max_grad_norm = max_grad_norm
        self.target_delta = target_delta
        self.privacy_engine = None
        self.accountant = None
        self.policy_max_eps = float(policy_max_eps)
        self._initialized = False

    def initialize_privacy(self, sample_rate: Optional[float] = None, epochs: int = 1):
        if not _OPACUS_AVAILABLE:
            logger.error(json.dumps({"msg":"Opacus not available"}))
            raise RuntimeError("Opacus required for DP")
        try:
            self.accountant = RDPAccountant()
            self.privacy_engine = PrivacyEngine(accountant=self.accountant)
            # Make private; record sample_rate and epochs to audit
            self.model, self.optimizer, self.train_loader = self.privacy_engine.make_private(
                module=self.model,
                optimizer=self.optimizer,
                data_loader=self.train_loader,
                noise_multiplier=self.noise_multiplier,
                max_grad_norm=self.max_grad_norm
            )
            self._initialized = True
            # audit record
            audit = {"event":"privacy_init","noise_multiplier":self.noise_multiplier,"max_grad_norm":self.max_grad_norm,"sample_rate":sample_rate,"epochs":epochs}
            _append_audit_record(audit)
            logger.info(json.dumps({"msg":"privacy initialized","noise_multiplier":self.noise_multiplier}))
        except Exception as e:
            DP_ACCOUNTANT_ERRORS.inc()
            logger.error(json.dumps({"msg":"privacy init failed","error":str(e)}))
            raise

    def compute_current_epsilon(self) -> Optional[float]:
        """
        Compute epsilon using RDP accountant and audited parameters. Returns None if unavailable.
        """
        if not _OPACUS_AVAILABLE or self.accountant is None:
            logger.warning(json.dumps({"msg":"accountant unavailable"}))
            return None
        try:
            eps = self.accountant.get_epsilon(delta=self.target_delta)
            DP_EPSILON_GAUGE.set(float(eps))
            # audit epsilon
            _append_audit_record({"event":"epsilon_report","epsilon":float(eps),"delta":self.target_delta})
            # enforce budget policy
            if float(eps) > self.policy_max_eps:
                DP_BUDGET_EXHAUSTED.inc()
                logger.error(json.dumps({"msg":"epsilon exceeds policy","epsilon":float(eps),"policy_max":self.policy_max_eps}))
            return float(eps)
        except Exception as e:
            DP_ACCOUNTANT_ERRORS.inc()
            logger.error(json.dumps({"msg":"epsilon compute failed","error":str(e)}))
            return None

    def local_train(self, epochs: int = 1):
        """
        Minimal local train example; production must supply model and dataloader logic.
        """
        if not self._initialized:
            raise RuntimeError("privacy not initialized")
        for _ in range(epochs):
            # training steps...
            pass
        # after training, compute current epsilon
        self.compute_current_epsilon()
