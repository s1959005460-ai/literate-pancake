# models/lora_adapter.py
import torch
import torch.nn as nn

class LoRALinear(nn.Module):
    """
    为任意线性层添加 LoRA 低秩适配器。
    W_eff = W + alpha * (B @ A)  (rank = r)
    """
    def __init__(self, in_features, out_features, r=4, alpha=1.0, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.r = r
        self.alpha = alpha
        # 原始权重
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        # LoRA 参数
        self.A = nn.Parameter(torch.Tensor(r, in_features))
        self.B = nn.Parameter(torch.Tensor(out_features, r))
        self.reset_parameters()

    def reset_parameters(self):
        # 初始化原始权重和 LoRA 参数
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        nn.init.zeros_(self.B)

    def forward(self, x):
        # 原始线性变换
        W_orig = self.weight
        # LoRA 低秩适配：B @ A (维度 out x in)
        W_lora = (self.B @ self.A) * (self.alpha / self.r)
        return torch.nn.functional.linear(x, W_orig + W_lora, self.bias)
