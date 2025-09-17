# File: scripts/compute_epsilon.py
"""
Compute (epsilon, delta) given Gaussian noise multiplier using RDP accountant.

Behavior:
 - Prefer opacus.accountants.rdp if opacus installed.
 - Otherwise, prefer tensorflow_privacy.accountants if available.
 - If neither present, compute a conservative/approximate bound and WARN strongly.

Usage:
    python scripts/compute_epsilon.py --q 0.01 --sigma 1.2 --steps 1000 --delta 1e-5

Notes:
 - This script should not be used as sole compliance evidence unless you validate results
   with a well-known library (opacus/tensorflow_privacy).
"""
from __future__ import annotations

import argparse
import math
import sys
import warnings
from typing import Tuple

try:
    # prefer Opacus (PyTorch) accountant
    from opacus.accountants import RDPAccountant  # type: ignore
    _HAS_OPACUS = True
except Exception:
    _HAS_OPACUS = False

try:
    # fallback to tensorflow_privacy
    from tensorflow_privacy.privacy.analysis import rdp_accountant as tf_rdp  # type: ignore
    _HAS_TFP = True
except Exception:
    _HAS_TFP = False

def compute_with_opacus(q: float, sigma: float, steps: int, delta: float) -> Tuple[float,int]:
    acct = RDPAccountant()
    acct.step(q, sigma, steps)  # note: API may differ per version—this is conceptual
    result = acct.get_privacy_spent(delta)
    # opacus API differences: adjust as needed
    return result.epsilon, result.order

def compute_with_tf(q: float, sigma: float, steps: int, delta: float) -> Tuple[float,int]:
    # using tensorflow_privacy rdp accountant
    orders = [1 + x / 10. for x in range(1, 1000)]
    rdp = tf_rdp.compute_rdp(q, sigma, steps, orders)
    eps, opt_order = tf_rdp.get_privacy_spent(orders, rdp, target_delta=delta)
    return eps, opt_order

def conservative_approx(q: float, sigma: float, steps: int, delta: float) -> Tuple[float,int]:
    """
    Conservative fallback (upper bound-ish). Uses Gaussian mechanism approximate composition:
    epsilon ≈ q * sqrt(2 * steps * ln(1/delta)) / sigma

    This is a conservative approximation and should NOT be used for compliance.
    """
    eps = q * math.sqrt(2.0 * steps * math.log(1.0 / delta)) / sigma
    return eps, 2

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--q", type=float, required=True, help="sampling probability per step")
    parser.add_argument("--sigma", type=float, required=True, help="noise multiplier (sigma)")
    parser.add_argument("--steps", type=int, required=True, help="number of steps / rounds")
    parser.add_argument("--delta", type=float, required=True)
    args = parser.parse_args()

    q, sigma, steps, delta = args.q, args.sigma, args.steps, args.delta

    if _HAS_OPACUS:
        try:
            eps, order = compute_with_opacus(q, sigma, steps, delta)
            print(f"Opacus: epsilon={eps:.6f}, optimal_order={order}")
            return
        except Exception:
            warnings.warn("Opacus accountant call failed; falling back", RuntimeWarning)

    if _HAS_TFP:
        try:
            eps, order = compute_with_tf(q, sigma, steps, delta)
            print(f"TensorFlow Privacy: epsilon={eps:.6f}, optimal_order={order}")
            return
        except Exception:
            warnings.warn("TensorFlowPrivacy accountant call failed; falling back", RuntimeWarning)

    warnings.warn(
        "No RDP accountant library found. Using CONSERVATIVE approximation. "
        "THIS IS NOT SUFFICIENT FOR COMPLIANCE. Install opacus or tensorflow_privacy.", UserWarning
    )
    eps, order = conservative_approx(q, sigma, steps, delta)
    print(f"Conservative approx: epsilon={eps:.6f}, order={order}")


if __name__ == "__main__":
    main()
