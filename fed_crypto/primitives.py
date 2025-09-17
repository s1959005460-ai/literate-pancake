# fed_crypto/primitives.py
import random
from fed_crypto.constants import DEFAULT_PRIME, DEFAULT_KEY_BITS

def modinv(a: int, p: int = DEFAULT_PRIME) -> int:
    """计算 a 在模 p 下的逆元，要求 gcd(a, p) = 1"""
    # 使用扩展欧几里得算法
    t, new_t = 0, 1
    r, new_r = p, a
    while new_r != 0:
        quotient = r // new_r
        t, new_t = new_t, t - quotient * new_t
        r, new_r = new_r, r - quotient * new_r
    if r > 1:
        raise ValueError(f"{a} has no inverse mod {p}")
    if t < 0:
        t += p
    return t

def modexp(base: int, exp: int, p: int = DEFAULT_PRIME) -> int:
    """快速幂模运算"""
    return pow(base, exp, p)

def rand_scalar(bits: int = DEFAULT_KEY_BITS) -> int:
    """生成随机数（模拟私钥/nonce）"""
    return random.getrandbits(bits) % DEFAULT_PRIME

def add_mod(a: int, b: int, p: int = DEFAULT_PRIME) -> int:
    return (a + b) % p

def sub_mod(a: int, b: int, p: int = DEFAULT_PRIME) -> int:
    return (a - b) % p

def mul_mod(a: int, b: int, p: int = DEFAULT_PRIME) -> int:
    return (a * b) % p
