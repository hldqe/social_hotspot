import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    """改进的Focal Loss"""
    def __init__(self, alpha=0.8, gamma=2.0, reduction='mean', class_weight=10.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.class_weight = class_weight  # 正类权重
        
    def forward(self, inputs, targets):
        # 计算BCE损失
        bce_loss = F.binary_cross_entropy_with_logits(
            inputs, targets, reduction='none'
        )
        
        # 计算权重
        weights = torch.ones_like(targets)
        weights[targets > 0.5] = self.class_weight  # 正类加权
        
        # 计算focal权重
        pt = torch.exp(-bce_loss)
        focal_weight = self.alpha * (1-pt)**self.gamma
        
        # 加权损失
        loss = weights * focal_weight * bce_loss
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss

class DiceLoss(nn.Module):
    """Dice损失，适用于类别不平衡问题"""
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth
        
    def forward(self, inputs, targets):
        # Sigmoid激活
        inputs = torch.sigmoid(inputs)
        
        # 平滑处理
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        # 计算交集
        intersection = (inputs * targets).sum()
        
        # 计算Dice系数
        dice = (2. * intersection + self.smooth) / (
            inputs.sum() + targets.sum() + self.smooth
        )
        
        return 1 - dice

class CombinedLoss(nn.Module):
    """组合损失函数: Focal Loss + Dice Loss"""
    def __init__(self, focal_weight=0.5, dice_weight=0.5, class_weight=10.0):
        super().__init__()
        self.focal_loss = FocalLoss(class_weight=class_weight)
        self.dice_loss = DiceLoss()
        self.focal_weight = focal_weight
        self.dice_weight = dice_weight
        
    def forward(self, inputs, targets):
        focal = self.focal_loss(inputs, targets)
        dice = self.dice_loss(inputs, targets)
        return self.focal_weight * focal + self.dice_weight * dice 