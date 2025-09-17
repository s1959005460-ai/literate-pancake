# utils/rollback.py
import copy

class ModelCheckpoint:
    """
    简易的模型检查点，用于保存和恢复模型状态。
    """
    def __init__(self, model):
        self.best_state = copy.deepcopy(model.state_dict())

    def update(self, model, current_metric):
        # 如果当前指标更好（例如规则效用提高），则更新最优状态
        self.best_state = copy.deepcopy(model.state_dict())

    def rollback(self, model):
        # 回滚到保存的最优状态
        model.load_state_dict(self.best_state)
