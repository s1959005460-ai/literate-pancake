# fed_crypto/constants.py

# 全局安全常量，避免魔法数字

# 大素数（用于 Zp，有需要可替换）
DEFAULT_PRIME = 2**61 - 1

# 默认比特长度（例如生成安全随机数时用）
DEFAULT_KEY_BITS = 256

# 超时时间（网络通信等）
DEFAULT_TIMEOUT = 30
