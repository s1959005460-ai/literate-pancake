# tests/test_secure_agg.py
import numpy as np
from FedGNN_advanced.privacy.secure_agg import generate_mask_for_param, aggregate_masked_updates, unmask_total

def test_mask_roundtrip():
    shape = (4,4)
    # create two updates
    u1 = np.ones(shape, dtype=np.float32)
    u2 = np.full(shape, 2.0, dtype=np.float32)
    m1 = generate_mask_for_param(shape)
    m2 = generate_mask_for_param(shape)
    masked1 = u1 + m1
    masked2 = u2 + m2
    total_masked = aggregate_masked_updates([masked1, masked2])
    # reconstruct masks: in toy test reconstruct as m1+m2
    reconstructed = [m1 + m2]
    total_plain = unmask_total(total_masked, reconstructed)
    assert np.allclose(total_plain, u1 + u2)
