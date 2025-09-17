# clients/client_utils.py
import torch

def compute_fisher_matrix(model, data_loader, device):
    """
    计算模型的费舍尔信息矩阵，可用于衡量参数重要性等。
    这里只给出伪代码示例，实际应根据模型梯度方差计算。
    """
    model.to(device)
    model.eval()
    fisher = {n: torch.zeros_like(p) for n, p in model.named_parameters()}
    for data in data_loader:
        data = data.to(device)
        model.zero_grad()
        output = model(data)
        # 假设使用标签的交叉熵
        loss = torch.nn.functional.nll_loss(output, data.y)
        loss.backward()
        for n, p in model.named_parameters():
            if p.grad is not None:
                fisher[n] += p.grad.data.pow(2)
    # 平均梯度平方
    fisher = {n: f / len(data_loader) for n, f in fisher.items()}
    return fisher

def compute_rule_utility(model, rule, data_loader, device):
    """
    计算单个规则的效用，例如规则应用后的性能提升。
    这里返回示例值，实际可根据数据集和模型性能计算。
    """
    # 示例：模型在应用 rule 后在数据集上的准确率
    accuracy = evaluate_model_with_rule(model, rule, data_loader, device)
    return accuracy
