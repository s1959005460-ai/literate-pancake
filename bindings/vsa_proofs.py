# bindings/vsa_proofs.py
import ctypes
import os

# 假设 vsa_proofs 编译后生成 libvsa_proofs.so
LIB_PATH = os.path.join(os.path.dirname(__file__), "..", "vsa_proofs", "target", "release", "libvsa_proofs.so")

_vsa = ctypes.CDLL(LIB_PATH)

# C 函数签名
_vsa.prove_and_verify.argtypes = [ctypes.c_ulonglong]
_vsa.prove_and_verify.restype = ctypes.c_bool

def prove_and_verify(x: int) -> bool:
    return _vsa.prove_and_verify(x)
