# utils/spectral_utils.py
import numpy as np

def compute_spectral_score(rule_embeddings):
    """
    计算规则的谱因果打分 C(s)，例如基于低秩表示的相似度或谱聚类方法。
    这里给出示例计算，实际可根据论文/需求调整。
    """
    # 假设 rule_embeddings 是形如 (num_rules, r)
    # 可以计算协方差矩阵的特征值总和作为谱度量
    cov_matrix = np.cov(rule_embeddings, rowvar=False)
    eigvals = np.linalg.eigvalsh(cov_matrix)
    return np.sum(eigvals)  # 谱总能量作为示例得分

def promote_rules(global_rules, aggregated_scores, threshold):
    """
    根据聚合后的因果得分 C(s)，对全局规则集进行推广。
    例如，选择得分高于阈值的规则为活跃规则。
    """
    promoted = [rule for rule, score in zip(global_rules, aggregated_scores) if score >= threshold]
    return promoted

