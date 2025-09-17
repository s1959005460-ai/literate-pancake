# utils/local_dp.py
import torch
import numpy as np

def add_local_dp_noise(tensor, epsilon, delta=1e-5, sensitivity=1.0):
    """
    向张量添加高斯噪声，实现 (epsilon, delta)-差分隐私。
    默认计算给定敏感度和隐私预算的噪声方差。
    """
    sigma = sensitivity * np.sqrt(2 * np.log(1.25 / delta)) / epsilon
    noise = torch.normal(mean=0, std=sigma, size=tensor.size(), device=tensor.device)
    return tensor + noise
