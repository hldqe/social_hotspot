"""
训练工具函数，包含损失函数和阈值搜索
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import f1_score

class FocalLoss(nn.Module):
    """Focal Loss实现，用于处理类别不平衡问题
    
    参考论文: "Focal Loss for Dense Object Detection"
    """
    def __init__(self, alpha=0.25, gamma=1.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        # 使用BCE损失作为基础
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        # 应用focal loss公式
        loss = self.alpha * (1-pt)**self.gamma * BCE_loss
        return loss.mean()

class RobustFocalLoss(nn.Module):
    """带类别均衡的稳健Focal Loss"""
    def __init__(self, alpha=0.25, gamma=1.0, class_weights=None, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.class_weights = class_weights  # [neg_weight, pos_weight]
        
    def forward(self, inputs, targets):
        # 计算BCE损失
        BCE_loss = F.binary_cross_entropy_with_logits(
            inputs, targets, reduction='none'
        )
        
        # 添加类别权重
        if self.class_weights is not None:
            weight_mask = torch.ones_like(targets)
            weight_mask[targets > 0] = self.class_weights[1]  # 正类权重
            weight_mask[targets <= 0] = self.class_weights[0]  # 负类权重
            BCE_loss = BCE_loss * weight_mask
        
        # 应用Focal Loss
        pt = torch.exp(-BCE_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * BCE_loss
        
        # 应用标签平滑 - 减少过拟合
        smoothed_targets = targets * 0.9 + 0.05  # 将1变为0.95，将0变为0.05
        label_smooth_loss = F.binary_cross_entropy_with_logits(
            inputs, smoothed_targets, reduction='none'
        )
        
        # 组合损失
        combined_loss = 0.8 * focal_loss + 0.2 * label_smooth_loss
        
        # 应用reduction
        if self.reduction == 'mean':
            return combined_loss.mean()
        elif self.reduction == 'sum':
            return combined_loss.sum()
        else:
            return combined_loss

def find_optimal_threshold(y_true, y_score):
    """寻找最优的分类阈值"""
    # 先检查是否有正样本，如果没有则返回默认阈值
    if sum(y_true) == 0:
        return 0.5, 0.0
        
    # 使用均匀分布的阈值点
    thresholds = np.linspace(0.1, 0.9, 50)  # 避免极端值0和1
    best_f1 = 0
    best_th = 0.5  # 默认阈值
    
    for th in thresholds:
        y_pred = (y_score > th).astype(int)
        # 确保存在预测的正样本，否则F1为0
        if sum(y_pred) > 0:
            f1 = f1_score(y_true, y_pred, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_th = th
    
    return best_th, best_f1
