# FedGNN_advanced/dp.py
import math
import logging
logger = logging.getLogger("fedgnn.dp")

try:
    from opacus.accountants.rdp_accountant import compute_eps_from_rdp as _opacus_compute_eps
    _HAS_OPACUS = True
except Exception:
    _HAS_OPACUS = False

def compute_epsilon_from_rdp(orders, rdp, target_delta):
    """
    Convert lists orders & rdp to epsilon via Mironov formula; use opacus if available.
    """
    if _HAS_OPACUS:
        try:
            return float(_opacus_compute_eps(orders, rdp, target_delta))
        except Exception as e:
            logger.warning("opacus accountant failed: %s; falling back", e)
    best = float("inf")
    for a, r in zip(orders, rdp):
        if a <= 1:
            continue
        eps = r + math.log(1.0 / target_delta) / (a - 1.0)
        if eps < best:
            best = eps
    return best
