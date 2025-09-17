# run_experiment.py
import logging
from typing import Dict, Any, Callable, Optional
import numpy as np
import torch

from FedGNN_advanced import constants
from FedGNN_advanced.protocols import UpdateFormat

logger = logging.getLogger("run_experiment")
logger.setLevel(logging.INFO)

FINITE_FIELD_PRIME = constants.FINITE_FIELD_PRIME
FLOAT_TO_INT_SCALE = constants.FLOAT_TO_INT_SCALE

def _finite_field_ints_to_float(arr_int: np.ndarray) -> np.ndarray:
    arr_int = np.asarray(arr_int, dtype=np.int64)
    p = int(FINITE_FIELD_PRIME)
    arr_mod = np.mod(arr_int, p).astype(np.int64)
    half = p // 2
    signed = np.where(arr_mod > half, arr_mod - p, arr_mod)
    return signed.astype(np.float32) / float(FLOAT_TO_INT_SCALE)

def _ensure_numpy_array(x) -> np.ndarray:
    if isinstance(x, np.ndarray):
        return x
    try:
        return np.array(x)
    except Exception as e:
        raise TypeError("Unable to convert update entry to numpy array") from e

def apply_aggregate_to_model(model: torch.nn.Module,
                             aggregate: Dict[str, Any],
                             device: str = "cpu",
                             fmt: UpdateFormat = UpdateFormat.FLOAT32,
                             strict: bool = True,
                             name_mapping: Optional[Callable[[str], str]] = None) -> None:
    if not isinstance(aggregate, dict):
        raise TypeError("aggregate must be a dict")

    processed = {}
    for k, v in aggregate.items():
        arr = _ensure_numpy_array(v)
        if fmt == UpdateFormat.FINITE_FIELD_INT64:
            arr = _finite_field_ints_to_float(arr)
        if arr.dtype != np.float32:
            arr = arr.astype(np.float32)
        processed[k] = arr

    param_dict = dict(model.named_parameters())

    with torch.no_grad():
        for agg_key, arr in processed.items():
            model_param_name = name_mapping(agg_key) if name_mapping else agg_key
            if model_param_name not in param_dict:
                msg = f"Aggregate key '{agg_key}' -> '{model_param_name}' not found in model"
                if strict:
                    logger.error(msg)
                    raise KeyError(msg)
                else:
                    logger.warning(msg)
                    continue
            param = param_dict[model_param_name]
            if tuple(arr.shape) != tuple(param.data.shape):
                msg = f"Shape mismatch '{model_param_name}': model {tuple(param.data.shape)}, agg {arr.shape}"
                if strict:
                    logger.error(msg)
                    raise ValueError(msg)
                else:
                    logger.warning(msg)
                    continue
            update_tensor = torch.tensor(arr, device=device, dtype=param.data.dtype)
            param.data.copy_(update_tensor)
    logger.info("Applied aggregate to model (fmt=%s, strict=%s)", fmt.name, strict)
