"""
图神经网络模型训练脚本
"""

import os
import logging
import argparse
import time
from datetime import datetime
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Union
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import torch.nn.functional as F
from torch.optim.lr_scheduler import OneCycleLR
from utils.graph_enhancer import enhance_graph, add_feature_noise, enhance_graph_structure
from utils.training_utils import FocalLoss, find_optimal_threshold
from models.enhanced_gnn import EnhancedGNN  # 导入新模型
from models.advanced_gnn import AdvancedGNN
from utils.feature_engineering import extract_node_features, add_virtual_connections, enhance_features
from utils.loss_functions import CombinedLoss, FocalLoss, DiceLoss
from utils.early_stopping import EarlyStopping

from config import *
from utils.simple_graph_builder import SimpleGraphBuilder
from models.gnn_model import create_model, SimpleGNN  # 修正：从正确的模块导入
from models.hybrid_gnn import HybridGNN  # 导入HybridGNN模型

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE, encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

print("=== 训练脚本开始执行 ===")

def train_model(model: nn.Module, 
              node_features: torch.Tensor, 
              edge_index: torch.Tensor, 
              edge_weight: torch.Tensor, 
              node_labels: torch.Tensor,
              train_mask: torch.Tensor,
              val_mask: torch.Tensor,
              test_mask: torch.Tensor,
              device: torch.device,
              num_epochs: int = NUM_EPOCHS,
              lr: float = LEARNING_RATE,
              weight_decay: float = WEIGHT_DECAY,
              early_stopping: int = EARLY_STOPPING) -> Tuple[Dict[str, List[float]], Dict[str, float]]:
    """
    训练GNN模型
    
    Args:
        model: GNN模型
        node_features: 节点特征
        edge_index: 边索引
        edge_weight: 边权重
        node_labels: 节点标签
        train_mask: 训练节点掩码
        val_mask: 验证节点掩码
        test_mask: 测试节点掩码
        device: 训练设备
        num_epochs: 训练轮数
        lr: 学习率
        weight_decay: 权重衰减
        early_stopping: 早停轮数
        
    Returns:
        训练历史和最佳验证指标
    """
    # 初始化
    model = model.to(device)
    node_features = node_features.to(device)
    edge_index = edge_index.to(device)
    edge_weight = edge_weight.to(device)
    node_labels = node_labels.to(device)
    
    # 确认数据类型
    node_features = node_features.float()
    edge_weight = edge_weight.float()
    node_labels = node_labels.long()
    
    # 计算权重
    pos_weight = torch.tensor([(len(node_labels) - node_labels.sum()) / max(node_labels.sum(), 1)])
    # 使用带权重的交叉熵
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))
    
    # 优化器
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    # 记录历史
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_acc': [],
        'val_acc': [],
        'train_f1': [],
        'val_f1': []
    }
    
    # 早停设置
    best_val_f1 = 0
    best_val_metrics = {}
    patience_counter = 0
    best_state = None
    
    # 训练循环
    for epoch in range(num_epochs):
        # 训练模式
        model.train()
        
        # 前向传播
        logits = model(node_features, edge_index, edge_weight)
        
        # 计算损失
        loss = criterion(logits[train_mask], node_labels[train_mask])
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # 评估训练集
        train_metrics = evaluate_node_classification(
            model, node_features, edge_index, edge_weight, node_labels, train_mask
        )
        
        # 评估验证集
        val_metrics = evaluate_node_classification(
            model, node_features, edge_index, edge_weight, node_labels, val_mask
        )
        
        # 记录历史
        history['train_loss'].append(loss.item())
        history['val_loss'].append(criterion(logits[val_mask], node_labels[val_mask]).item())
        history['train_acc'].append(train_metrics['accuracy'])
        history['val_acc'].append(val_metrics['accuracy'])
        history['train_f1'].append(train_metrics['f1'])
        history['val_f1'].append(val_metrics['f1'])
        
        # 输出进度
        if (epoch + 1) % 10 == 0 or epoch == 0:
            logger.info(f"轮次 {epoch+1}/{num_epochs} - "
                      f"训练损失: {loss.item():.4f}, "
                      f"训练准确率: {train_metrics['accuracy']:.4f}, "
                      f"验证准确率: {val_metrics['accuracy']:.4f}, "
                      f"训练F1: {train_metrics['f1']:.4f}, "
                      f"验证F1: {val_metrics['f1']:.4f}")
        
        # 早停检查
        if val_metrics['f1'] > best_val_f1:
            best_val_f1 = val_metrics['f1']
            best_val_metrics = val_metrics.copy()
            best_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= early_stopping:
                logger.info(f"早停触发！在 {epoch+1} 轮次停止训练")
                break
    
    # 加载最佳模型
    if best_state is not None:
        model.load_state_dict(best_state)
    
    # 在测试集上评估
    test_metrics = evaluate_node_classification(
        model, node_features, edge_index, edge_weight, node_labels, test_mask
    )
    
    # 记录最佳结果
    logger.info("训练完成！")
    log_metrics(best_val_metrics, "最佳验证集")
    log_metrics(test_metrics, "测试集")
    
    return history, test_metrics

def create_train_val_test_masks(num_nodes: int, val_ratio: float = 0.2, test_ratio: float = 0.2) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    创建训练、验证和测试掩码
    
    Args:
        num_nodes: 节点数量
        val_ratio: 验证集比例
        test_ratio: 测试集比例
        
    Returns:
        (训练掩码, 验证掩码, 测试掩码)
    """
    indices = torch.randperm(num_nodes)
    
    test_size = int(num_nodes * test_ratio)
    val_size = int(num_nodes * val_ratio)
    train_size = num_nodes - test_size - val_size
    
    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size+val_size]
    test_indices = indices[train_size+val_size:]
    
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    
    train_mask[train_indices] = True
    val_mask[val_indices] = True
    test_mask[test_indices] = True
    
    return train_mask, val_mask, test_mask

def save_model(model: nn.Module, save_dir: str, model_name: str) -> None:
    """
    保存模型
    
    Args:
        model: 要保存的模型
        save_dir: 保存目录
        model_name: 模型名称
    """
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{model_name}.pth")
    torch.save(model.state_dict(), save_path)
    logger.info(f"模型已保存到 {save_path}")

def load_model(model: nn.Module, model_path: str) -> nn.Module:
    """
    加载模型
    
    Args:
        model: 模型实例
        model_path: 模型路径
        
    Returns:
        加载参数后的模型
    """
    model.load_state_dict(torch.load(model_path))
    logger.info(f"模型已加载: {model_path}")
    return model

# 添加缺失的评估函数
def evaluate_node_classification(model, node_features, edge_index, edge_weight, node_labels, mask):
    """
    评估节点分类任务
    
    Args:
        model: 模型
        node_features: 节点特征
        edge_index: 边索引
        edge_weight: 边权重
        node_labels: 节点标签
        mask: 评估节点掩码
        
    Returns:
        评估指标字典
    """
    model.eval()
    with torch.no_grad():
        logits = model(node_features, edge_index, edge_weight)
        logits = logits.squeeze(-1)  # 压缩维度
        probs = torch.sigmoid(logits[mask])
        preds = (probs > 0.5).float()
        
        # 计算指标
        y_true = node_labels[mask].cpu().numpy()
        y_pred = preds.cpu().numpy()
        
        # 如果只有一个类，sklearn.metrics可能会出错
        if len(np.unique(y_true)) == 1 or len(np.unique(y_pred)) == 1:
            # 全部预测正确
            if np.array_equal(y_true, y_pred):
                accuracy = 1.0
                precision = 1.0 if np.sum(y_pred) > 0 else 0.0
                recall = 1.0 if np.sum(y_true) > 0 else 0.0
                f1 = 1.0 if precision > 0 and recall > 0 else 0.0
            # 全部预测错误
            else:
                accuracy = 0.0
                precision = 0.0
                recall = 0.0
                f1 = 0.0
        else:
            # 计算常规指标
            accuracy = accuracy_score(y_true, y_pred)
            precision, recall, f1, _ = precision_recall_fscore_support(
                y_true, y_pred, average='binary', zero_division=0)
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }

def log_metrics(metrics, dataset_name=""):
    """记录评估指标"""
    logger.info(f"{dataset_name} 评估结果:")
    logger.info(f"- 准确率: {metrics['accuracy']:.4f}")
    logger.info(f"- 精确率: {metrics['precision']:.4f}")
    logger.info(f"- 召回率: {metrics['recall']:.4f}")
    logger.info(f"- F1分数: {metrics['f1']:.4f}")

def plot_confusion_matrix(y_true, y_pred, classes, save_path=None):
    """画出混淆矩阵"""
    pass  # 简化版本不需要实现

def plot_metrics_history(history, save_path=None):
    """画出训练历史指标"""
    pass  # 简化版本不需要实现

class Trainer:
    def __init__(self, model_type: str, data_prefix: str, epochs: int = 100):
        """
        初始化训练器
        
        Args:
            model_type: 模型类型 ('SimpleGNN' or 'EnhancedGNN' or 'HybridGNN' or 'AdvancedGNN')
            data_prefix: 数据前缀
            epochs: 训练轮数
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"使用设备: {self.device}")
        
        # 加载数据
        self.load_data(data_prefix)
        
        # 计算类别权重（处理样本不平衡）
        total_samples = len(self.labels)
        hot_samples = self.labels.sum().item()
        non_hot_samples = total_samples - hot_samples
        
        logger.info(f"热点节点比例: {hot_samples/total_samples:.4f} ({hot_samples}/{total_samples})")
        
        # 创建模型
        if model_type == 'AdvancedGNN':
            # 增强特征
            logger.info("增强节点特征...")
            self.original_features = self.features.clone()
            self.features = extract_node_features(self.features, self.edge_index, self.labels)
            
            # 增强图结构
            logger.info("增强图结构...")
            try:
                import networkx as nx
                edges = [(self.edge_index[0, i].item(), self.edge_index[1, i].item()) 
                        for i in range(self.edge_index.size(1))]
                G = nx.Graph()
                G.add_edges_from(edges)
                G = enhance_graph_structure(G)
                
                edges = list(G.edges())
                new_edge_index = torch.tensor([[u, v] for u, v in edges], dtype=torch.long).t()
                self.edge_index = new_edge_index
            except Exception as e:
                logger.warning(f"图增强失败: {str(e)}")
            
            # 创建模型
            self.model = AdvancedGNN(
                input_dim=self.features.size(1),
                hidden_dim=256  # 降低复杂度
            ).to(self.device)
            
            # 计算正类权重
            total_samples = len(self.labels)
            hot_samples = self.labels.sum().item()
            non_hot_samples = total_samples - hot_samples
            pos_weight = torch.tensor(non_hot_samples / hot_samples)
            
            logger.info(f"正类权重: {pos_weight.item():.2f} (非热点数/热点数: {non_hot_samples}/{hot_samples})")
            
            # 使用权重化的BCEWithLogitsLoss
            self.criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(self.device))
            
            # 优化器设置
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=5e-5,  # 降低学习率提高稳定性
                weight_decay=1e-4  # 更强的L2正则化
            )
            
            # 学习率调度器
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='max',
                factor=0.5,
                patience=5,
                verbose=True,
                min_lr=1e-6
            )
            
            # 提高早停的耐心值
            self.early_stopping = EarlyStopping(
                patience=30,  # 增加耐心值避免过早停止
                min_delta=0.001,
                verbose=True
            )
        elif model_type == 'HybridGNN':
            self.model = HybridGNN(
                input_dim=INPUT_DIM,
                hidden_dim=HIDDEN_DIM
            ).to(self.device)
            
            # 添加损失函数 - 这行很重要！
            pos_weight = torch.tensor(9.0)  # 约为非热点/热点比例
            self.criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(self.device))
            
            # 使用分层学习率
            gnn_params = list(self.model.gnn1.parameters()) + list(self.model.gnn2.parameters())
            other_params = list(self.model.fc_raw.parameters()) + list(self.model.classifier.parameters())
            
            self.optimizer = optim.AdamW([
                {'params': gnn_params, 'lr': 0.002},
                {'params': other_params, 'lr': 0.005}
            ], weight_decay=1e-4)
            
            # 使用reduce on plateau调度器
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode='max', factor=0.5, patience=5, verbose=True
            )
        elif model_type == 'EnhancedGNN':
            self.model = EnhancedGNN(
                input_dim=INPUT_DIM,
                hidden_dim=HIDDEN_DIM, 
                num_layers=3,  # 使用更深的网络
                dropout=0.4    # 增加dropout
            ).to(self.device)
            
            # 使用改进的损失函数
            self.criterion = FocalLoss(alpha=0.8, gamma=2)
            
            # 使用AdamW优化器
            self.optimizer = optim.AdamW(self.model.parameters(), lr=0.005, weight_decay=1e-4)
            
            # 添加学习率调度器
            self.scheduler = OneCycleLR(
                self.optimizer, 
                max_lr=0.005,
                steps_per_epoch=1, 
                epochs=epochs,
                pct_start=0.3
            )
        else:
            self.model = SimpleGNN(
                input_dim=INPUT_DIM,
                hidden_dim=HIDDEN_DIM,
                num_layers=NUM_LAYERS,
                dropout=DROPOUT
            ).to(self.device)
            
            # 使用更合理的权重值 - 接近热点节点与非热点节点的比例
            pos_weight = torch.tensor(9.0)  # 约为非热点/热点比例
            self.criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(self.device))
            
            # 优化器
            self.optimizer = optim.Adam(self.model.parameters(), lr=0.002)
            self.scheduler = None
        
        # 创建TensorBoard写入器
        self.writer = SummaryWriter(os.path.join(LOG_DIR, f"{model_type}_{data_prefix}"))
        
        logger.info(f"模型参数总量: {sum(p.numel() for p in self.model.parameters())}")
    
    def load_data(self, data_prefix: str):
        """加载图数据，优化数据分割"""
        data_path = os.path.join(PROCESSED_DATA_DIR, f"{data_prefix}_graph_data.pt")
        logger.info(f"从 {data_path} 加载数据")
        
        data = torch.load(data_path)
        self.features = data['node_features'].to(self.device)
        self.edge_index = data['edge_index'].to(self.device)
        self.edge_weight = data['edge_weight'].to(self.device)
        self.labels = data['node_labels'].float().to(self.device)
        
        # 创建训练/验证/测试掩码 - 使用分层抽样
        num_nodes = self.features.size(0)
        
        # 标签转换为NumPy数组用于分层抽样
        all_labels = self.labels.cpu().numpy()
        
        # 使用分层抽样确保标签分布一致
        from sklearn.model_selection import train_test_split
        
        # 先分离测试集
        train_val_idx, test_idx = train_test_split(
            np.arange(num_nodes), 
            test_size=0.15,         # 15%作为测试集 
            random_state=42, 
            stratify=all_labels     # 确保分层抽样
        )
        
        # 再分离训练集和验证集
        train_idx, val_idx = train_test_split(
            train_val_idx,
            test_size=0.18,         # 约15%作为验证集
            random_state=42,
            stratify=all_labels[train_val_idx]  # 确保分层抽样
        )
        
        # 创建掩码
        self.train_mask = torch.zeros(num_nodes, dtype=torch.bool)
        self.val_mask = torch.zeros(num_nodes, dtype=torch.bool)
        self.test_mask = torch.zeros(num_nodes, dtype=torch.bool)
        
        self.train_mask[train_idx] = True
        self.val_mask[val_idx] = True
        self.test_mask[test_idx] = True
        
        # 统计类别分布，确认分层抽样效果
        train_hot_ratio = self.labels[self.train_mask].mean().item()
        val_hot_ratio = self.labels[self.val_mask].mean().item()
        test_hot_ratio = self.labels[self.test_mask].mean().item()
        
        logger.info(f"数据加载完成:")
        logger.info(f"- 训练集: {self.train_mask.sum().item()} 个节点, 热点比例: {train_hot_ratio:.4f}")
        logger.info(f"- 验证集: {self.val_mask.sum().item()} 个节点, 热点比例: {val_hot_ratio:.4f}")
        logger.info(f"- 测试集: {self.test_mask.sum().item()} 个节点, 热点比例: {test_hot_ratio:.4f}")
    
    def train_epoch(self):
        self.model.train()
        
        # 前向传播
        self.optimizer.zero_grad()
        logits = self.model(self.features, self.edge_index, self.edge_weight)
        
        # 确保维度匹配
        logits = logits.squeeze(-1)
        
        # 计算损失
        loss = self.criterion(logits[self.train_mask], self.labels[self.train_mask])
        
        # 反向传播
        loss.backward()
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        return loss.item()
    
    def evaluate(self, mask) -> tuple:
        """评估模型，使用动态阈值优化"""
        self.model.eval()
        with torch.no_grad():
            logits = self.model(self.features, self.edge_index, self.edge_weight)
            logits = logits.squeeze(-1)
            probs = torch.sigmoid(logits[mask])
            
            # 计算指标
            y_true = self.labels[mask].cpu().numpy()
            y_score = probs.cpu().numpy()
            
            # 对验证集使用阈值搜索
            if mask is self.val_mask:
                best_threshold = 0.5
                best_f1 = 0.0  # 确保初始值是数值
                
                # 搜索最佳阈值
                for threshold in np.arange(0.3, 0.8, 0.05):
                    y_pred = (y_score > threshold).astype(int)
                    
                    # 确保安全计算F1分数
                    try:
                        if len(np.unique(y_pred)) == 1:
                            # 处理全0或全1预测的情况
                            if np.unique(y_pred)[0] == 0:  # 全预测为0
                                precision = 0.0
                                recall = 0.0
                                f1 = 0.0
                            else:  # 全预测为1
                                tp = np.sum((y_pred == 1) & (y_true == 1))
                                precision = tp / np.sum(y_pred == 1) if np.sum(y_pred == 1) > 0 else 0.0
                                recall = tp / np.sum(y_true == 1) if np.sum(y_true == 1) > 0 else 0.0
                                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
                        else:
                            # 正常情况下计算
                            precision, recall, f1, _ = precision_recall_fscore_support(
                                y_true, y_pred, average='binary', zero_division=0)
                            
                            # 确保f1不是None
                            f1 = 0.0 if f1 is None else f1
                    except Exception as e:
                        logger.warning(f"阈值{threshold}计算F1分数时出错: {str(e)}")
                        precision, recall, f1 = 0.0, 0.0, 0.0
                    
                    if f1 > best_f1:
                        best_f1 = f1
                        best_threshold = threshold
                
                # 记录最佳阈值
                self.best_threshold = best_threshold
                logger.info(f"验证集上的最佳阈值: {best_threshold:.3f}, F1: {best_f1:.4f}")
            else:
                # 训练集和测试集使用验证集上的最佳阈值
                if hasattr(self, 'best_threshold'):
                    best_threshold = self.best_threshold
                else:
                    best_threshold = 0.5
            
            # 使用阈值预测
            y_pred = (y_score > best_threshold).astype(int)
            
            # 计算最终指标，确保异常处理
            try:
                if len(np.unique(y_pred)) == 1:
                    # 处理全0或全1预测的情况
                    if np.unique(y_pred)[0] == 0:  # 全预测为0
                        accuracy = np.mean(y_true == 0)
                        precision = 0.0
                        recall = 0.0
                        f1 = 0.0
                    else:  # 全预测为1
                        accuracy = np.mean(y_true == 1)
                        tp = np.sum((y_pred == 1) & (y_true == 1))
                        precision = tp / np.sum(y_pred == 1) if np.sum(y_pred == 1) > 0 else 0.0
                        recall = tp / np.sum(y_true == 1) if np.sum(y_true == 1) > 0 else 0.0
                        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
                else:
                    # 正常情况下计算
                    accuracy = accuracy_score(y_true, y_pred)
                    precision, recall, f1, _ = precision_recall_fscore_support(
                        y_true, y_pred, average='binary', zero_division=0)
            except Exception as e:
                logger.warning(f"计算最终指标时出错: {str(e)}")
                accuracy, precision, recall, f1 = 0.0, 0.0, 0.0, 0.0
            
            return accuracy, precision, recall, f1
    
    def train(self, epochs: int):
        """训练模型"""
        # 确保模型保存目录存在
        os.makedirs(MODEL_DIR, exist_ok=True)
        
        best_val_f1 = 0
        
        for epoch in range(epochs):
            # 训练一个epoch
            train_loss = self.train_epoch()
            
            # 评估
            train_acc, train_prec, train_recall, train_f1 = self.evaluate(self.train_mask)
            val_acc, val_prec, val_recall, val_f1 = self.evaluate(self.val_mask)
            
            # 更新学习率 - 修改这里，确保传入正确的参数
            if self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_f1)
                else:
                    self.scheduler.step()
            
            # 记录日志
            self.writer.add_scalar('Loss/train', train_loss, epoch)
            self.writer.add_scalar('Accuracy/train', train_acc, epoch)
            self.writer.add_scalar('F1/train', train_f1, epoch)
            self.writer.add_scalar('Accuracy/val', val_acc, epoch)
            self.writer.add_scalar('F1/val', val_f1, epoch)
            
            # 打印进度
            if epoch % 10 == 0:
                logger.info(
                    f"Epoch {epoch:03d}: "
                    f"Train Loss: {train_loss:.4f}, "
                    f"Train F1: {train_f1:.4f}, "
                    f"Val F1: {val_f1:.4f}"
                )
            
            # 保存最佳模型
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                self.save_model()
            
            # 早停检查
            if self.early_stopping(val_f1):
                logger.info(f'Early stopping at epoch {epoch}, 最佳验证F1={best_val_f1:.4f}')
                break
        
        # 在测试集上评估
        test_acc, test_prec, test_recall, test_f1 = self.evaluate(self.test_mask)
        
        logger.info("训练完成!")
        logger.info(f"测试集结果:")
        logger.info(f"- 准确率: {test_acc:.4f}")
        logger.info(f"- 精确率: {test_prec:.4f}")
        logger.info(f"- 召回率: {test_recall:.4f}")
        logger.info(f"- F1分数: {test_f1:.4f}")

    def save_model(self):
        """保存模型"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        model_name = f"{timestamp}_best_model.pth"
        model_path = os.path.join(MODEL_DIR, model_name)
        torch.save(self.model.state_dict(), model_path)
        logger.info(f"保存新的最佳模型，F1={self.evaluate(self.val_mask)[3]:.4f}，路径={model_path}")

    def train_with_cv(self, epochs: int, k_folds=5):
        """使用交叉验证训练模型"""
        # 保存原始掩码
        all_indices = torch.arange(self.features.size(0))
        train_indices = all_indices[self.train_mask | self.val_mask]  # 合并训练和验证集
        test_indices = all_indices[self.test_mask]
        
        # 创建K折交叉验证
        kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
        
        # 存储每折的最佳模型
        fold_models = []
        fold_f1_scores = []
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(train_indices)):
            logger.info(f"开始第 {fold+1}/{k_folds} 折训练")
            
            # 创建当前折的掩码
            curr_train_mask = torch.zeros(self.features.size(0), dtype=torch.bool)
            curr_val_mask = torch.zeros(self.features.size(0), dtype=torch.bool)
            
            curr_train_mask[train_indices[train_idx]] = True
            curr_val_mask[train_indices[val_idx]] = True
            
            # 重置模型
            if fold > 0:  # 第一折已经初始化过了
                self._reset_model()
            
            # 训练当前折
            best_val_f1, best_state = self._train_single_fold(
                epochs, curr_train_mask, curr_val_mask
            )
            
            fold_models.append(best_state)
            fold_f1_scores.append(best_val_f1)
            logger.info(f"第 {fold+1} 折完成，最佳验证F1: {best_val_f1:.4f}")
        
        # 使用最佳折模型预测
        best_fold = np.argmax(fold_f1_scores)
        logger.info(f"最佳模型来自第 {best_fold+1} 折，F1分数: {fold_f1_scores[best_fold]:.4f}")
        
        # 加载最佳模型评估
        self.model.load_state_dict(fold_models[best_fold])
        test_metrics = self._evaluate(self.test_mask)
        
        logger.info("交叉验证训练完成!")
        logger.info(f"测试集结果:")
        for metric, value in test_metrics.items():
            logger.info(f"- {metric}: {value:.4f}")
        
        return test_metrics

    def prepare_batches(self):
        """准备节点批次，用于大规模图训练"""
        if self.labels is None:
            return None
        
        num_nodes = self.labels.size(0)
        indices = torch.randperm(num_nodes)
        
        # 创建批次
        batches = []
        for i in range(0, num_nodes, BATCH_SIZE):
            end = min(i + BATCH_SIZE, num_nodes)
            batch_indices = indices[i:end]
            batches.append(batch_indices)
        
        return batches
    
    def train_batch(self, batch_indices):
        """训练单个批次"""
        self.model.train()
        self.optimizer.zero_grad()
        
        # 如果使用节点采样
        if NODE_SAMPLING:
            # 实现节点采样逻辑
            sampled_features, sampled_edge_index, sampled_labels = self.sample_subgraph(batch_indices)
            output = self.model(sampled_features, sampled_edge_index)
            loss = self.criterion(output, sampled_labels.float())
        else:
            # 常规批次训练
            output = self.model(self.features, self.edge_index)[batch_indices]
            loss = self.criterion(output, self.labels[batch_indices].float())
        
        loss.backward()
        # 添加梯度裁剪
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        return loss.item()

    def sample_subgraph(self, batch_indices):
        """采样子图，用于节点采样算法"""
        # 这是一个简化版本，实际上可能需要更复杂的采样算法
        # 如GraphSAGE的邻居采样
        
        # 示例：一阶邻居采样
        neighbors = set(batch_indices.tolist())
        edge_mask = []
        
        # 找出这些节点的边
        for i in range(self.edge_index.size(1)):
            src, dst = self.edge_index[0, i].item(), self.edge_index[1, i].item()
            if src in neighbors and dst in neighbors:
                edge_mask.append(i)
        
        # 创建子图
        sub_edge_index = self.edge_index[:, edge_mask]
        
        # 节点重新映射
        node_map = {old_idx: new_idx for new_idx, old_idx in enumerate(neighbors)}
        
        # 重映射边索引
        new_edge_index = torch.zeros_like(sub_edge_index)
        for i in range(sub_edge_index.size(1)):
            src, dst = sub_edge_index[0, i].item(), sub_edge_index[1, i].item()
            new_edge_index[0, i] = node_map[src]
            new_edge_index[1, i] = node_map[dst]
        
        # 提取节点特征和标签
        nodes = list(neighbors)
        sub_features = self.features[nodes]
        sub_labels = self.labels[batch_indices]
        
        return sub_features, new_edge_index, sub_labels

if __name__ == "__main__":
    try:
        print("解析命令行参数...")
        parser = argparse.ArgumentParser(description="社交热点预测模型训练器")
        parser.add_argument("--model_type", type=str, required=True, help="模型类型")
        parser.add_argument("--data_prefix", type=str, required=True, help="数据前缀")
        parser.add_argument("--epochs", type=int, required=True, help="训练轮数")
        args = parser.parse_args()
        
        print("开始训练...")
        trainer = Trainer(args.model_type, args.data_prefix, args.epochs)
        trainer.train(args.epochs)
    except Exception as e:
        print(f"发生错误: {e}")
        import traceback
        traceback.print_exc()