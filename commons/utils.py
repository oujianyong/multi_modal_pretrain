import torch 
import random
import numpy as np
import pandas as pd
import torch.distributed as dist
from sklearn.metrics import roc_auc_score, precision_recall_curve, roc_curve


# 设置随机种子
def fix_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


# 模型参数初始化
def init_weights(module):
    if isinstance(module, torch.nn.Linear):
        torch.nn.init.xavier_uniform_(module.weight)
        module.bias.data.fill_(0.01)


# -------------------评估指标计算------------------------------#
# 单机模式计算auc、ks、f1、top_recall
def compute_auc_f1_ks_toprecall(y_true, y_pred) -> tuple[float, float, float, float, float, float]:

    # auc计算
    def _auc_score(y_true: np.array, y_pred: np.array) -> float:
        auc = roc_auc_score(y_true, y_pred)
        return auc

    # f1计算
    def _f1_score(y_true: np.array, y_pred: np.array) -> float:
        precisions, recalls, _ = precision_recall_curve(y_true, y_pred)
        f1_scores = (2 * precisions * recalls) / (precisions + recalls)
        return np.max(f1_scores[np.isfinite(f1_scores)])

    # ks计算
    def _ks_score(y_true: np.array, y_pred: np.array) -> float:
        fpr, tpr, _ = roc_curve(y_true, y_pred)
        ks = np.max(fpr - tpr)
        return ks

    # k1、k5、b1的recall
    def _top_recall(y_true: np.array, y_pred: np.array) -> float:
        # [1] 构建数据
        data = np.concatenate([y_true, y_pred], axis=-1)
        data = pd.DataFrame(data, columns=['label', 'score'])
        n_data, n_positive = len(data), data.label.sum()
        # [2] 排序
        data = data.sort_values(by=['score'], ascending=False).reset_index(drop=True)
        # [3] 计算k1、k5、b1 的recall
        k1_recall = round(data.head(int(n_data*0.001)).label.sum() / n_positive, 3)
        k5_recall = round(data.head(int(n_data*0.005)).label.sum() / n_positive, 3)
        b1_recall = round(data.head(int(n_data*0.010)).label.sum() / n_positive, 3)
        return k1_recall, k5_recall, b1_recall
    
    # 数据结构变化
    if torch.is_tensor(y_true):
        y_true = y_true.cup().numpy()
    if torch.is_tensor(y_pred):
        y_pred = y_pred.cup().numpy()
    
    k1_recall, k5_recall, b1_recall = _top_recall(y_true, y_pred)
    y_true, y_pred = y_true.reshape(-1), y_pred.reshape(-1)
    auc, k1, ks = _auc_score(y_true, y_pred), _f1_score(y_true, y_pred), _ks_score(y_true, y_pred)
    return auc, k1, ks, k1_recall, k5_recall, b1_recall


# 分布式环境下计算全局的auc、ks、f1、top_recall
def compute_global_auc_f1_ks_toprecall(y_true: torch.Tensor, y_pred: torch.Tensor) -> tuple[float, float, float, float, float, float]:
    # [1]确保输入张量在正确设备上
    y_true, y_pred = y_true.view(-1).cuda(), y_pred.view(-1).cuda()
    gathered_y_true = [torch.zeros_like(y_true) for _ in range(dist.get_world_size())]
    gathered_y_pred = [torch.zeros_like(y_pred) for _ in range(dist.get_world_size())]
    # [2]收集标签与打分
    dist.barrier()
    dist.all_gather(gathered_y_true, y_true)
    dist.barrier()
    dist.all_gather(gathered_y_pred, y_pred)
    # [3]在主进程计算全局指标
    auc, k1, ks, k1_recall, k5_recall, b1_recall = None, None, None, None,None, None
    if dist.get_rank() == 0:
        all_y_true, all_y_pred = torch.cat(gathered_y_true), torch.cat(gathered_y_pred)
        auc, k1, ks, k1_recall, k5_recall, b1_recall = compute_auc_f1_ks_toprecall(all_y_true, all_y_pred)
    return auc, k1, ks, k1_recall, k5_recall, b1_recall

    
