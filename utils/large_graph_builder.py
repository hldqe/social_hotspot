"""
优化的大规模图构建模块
"""

import os
import pandas as pd
import numpy as np
import json
import logging
from typing import Dict, List, Tuple, Optional, Set
from collections import defaultdict
import networkx as nx
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import gc  # 垃圾回收

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import *

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class LargeGraphBuilder:
    """针对大规模数据优化的图构建类"""
    
    def __init__(self, entity_dict: Optional[Dict[str, int]] = None,
                data_dir: str = PROCESSED_DATA_DIR,
                data_prefix: str = "mind_full"):
        """
        初始化图构建器
        
        Args:
            entity_dict: 实体词典 {实体: 索引}
            data_dir: 数据目录
            data_prefix: 数据前缀
        """
        self.data_dir = data_dir
        self.data_prefix = data_prefix
        
        # 加载实体词典
        if entity_dict is None:
            entity_dict_path = os.path.join(data_dir, f"{data_prefix}_entity_dict.json")
            if os.path.exists(entity_dict_path):
                with open(entity_dict_path, 'r') as f:
                    self.entity_dict = json.load(f)
            else:
                raise FileNotFoundError(f"实体词典不存在: {entity_dict_path}")
        else:
            self.entity_dict = entity_dict
            
        self.idx_to_entity = {idx: entity for entity, idx in self.entity_dict.items()}
        self.num_entities = len(self.entity_dict)
        
        logger.info(f"大规模图构建器初始化完成，共 {self.num_entities} 个实体")
    
    def build_cooccurrence_graph(self, df: pd.DataFrame, batch_size: int = 10000) -> nx.Graph:
        """
        批处理方式构建共现关系图
        
        Args:
            df: 包含实体列表的DataFrame
            batch_size: 每批处理的数据量
            
        Returns:
            NetworkX图对象
        """
        logger.info("构建共现关系图")
        
        # 创建无向图
        G = nx.Graph()
        
        # 添加所有实体作为节点
        for entity_idx in range(self.num_entities):
            G.add_node(entity_idx, name=self.idx_to_entity[entity_idx])
        
        # 统计实体共现次数 - 使用批处理
        cooccurrence = defaultdict(int)
        
        # 计算总批次数
        num_batches = (len(df) + batch_size - 1) // batch_size
        
        for batch_idx in tqdm(range(num_batches), desc="构建共现关系"):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, len(df))
            batch_df = df.iloc[start_idx:end_idx]
            
            # 处理当前批次
            for _, row in batch_df.iterrows():
                entities = row['entities']
                
                # 处理不同类型的实体列表
                if not isinstance(entities, list):
                    try:
                        # 尝试将字符串转换为列表
                        entities = eval(entities)
                    except:
                        entities = []
                        
                if not entities or len(entities) < 2:
                    continue
                    
                # 获取实体索引
                entity_indices = [self.entity_dict.get(e) for e in entities if e in self.entity_dict]
                entity_indices = [idx for idx in entity_indices if idx is not None]
                
                # 统计所有实体对的共现次数
                for i in range(len(entity_indices)):
                    for j in range(i+1, len(entity_indices)):
                        pair = tuple(sorted([entity_indices[i], entity_indices[j]]))
                        cooccurrence[pair] += 1
            
            # 每10个批次清理内存
            if batch_idx % 10 == 0:
                gc.collect()
        
        # 添加边，权重为共现次数
        logger.info("添加边到图...")
        for (entity_i, entity_j), weight in tqdm(cooccurrence.items(), desc="添加边"):
            if weight >= MIN_EDGE_WEIGHT:
                G.add_edge(entity_i, entity_j, weight=weight)
        
        # 清理内存
        del cooccurrence
        gc.collect()
        
        logger.info(f"共现关系图构建完成，共 {G.number_of_nodes()} 个节点，{G.number_of_edges()} 条边")
        return G
    
    def convert_graph_to_pytorch(self, nx_graph: nx.Graph) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        将NetworkX图转换为PyTorch张量，针对大图优化
        
        Args:
            nx_graph: NetworkX图对象
            
        Returns:
            (节点特征张量, 边索引张量, 边权重张量)
        """
        logger.info("将图转换为PyTorch张量")
        
        # 获取边列表和权重
        src_nodes = []
        dst_nodes = []
        edge_weights = []
        
        # 分批处理边
        edges = list(nx_graph.edges(data=True))
        for i, (u, v, data) in enumerate(tqdm(edges, desc="处理边")):
            src_nodes.append(u)
            dst_nodes.append(v)
            edge_weights.append(data['weight'])
            # 对于无向图，添加反向边
            src_nodes.append(v)
            dst_nodes.append(u)
            edge_weights.append(data['weight'])
            
            # 每处理10000条边清理一次内存
            if i % 10000 == 0:
                gc.collect()
        
        # 转换为PyTorch张量
        edge_index = torch.tensor([src_nodes, dst_nodes], dtype=torch.long)
        edge_weight = torch.tensor(edge_weights, dtype=torch.float)
        
        # 清理临时变量
        del src_nodes, dst_nodes, edge_weights
        gc.collect()
        
        # 创建节点特征（随机初始化）
        logger.info(f"创建节点特征 (维度: {INPUT_DIM})...")
        node_features = torch.randn(self.num_entities, INPUT_DIM)
        
        logger.info(f"转换完成，边数: {len(edges)} (双向: {edge_index.size(1)})")
        return node_features, edge_index, edge_weight
    
    def compute_node_labels(self, df: pd.DataFrame, batch_size: int = 10000) -> torch.Tensor:
        """
        计算节点标签（热点/非热点），针对大数据集优化
        
        Args:
            df: 包含热点标签的DataFrame
            batch_size: 每批处理的数据量
            
        Returns:
            节点标签张量
        """
        logger.info("计算节点标签")
        
        # 初始化所有节点标签为0（非热点）
        node_labels = torch.zeros(self.num_entities, dtype=torch.long)
        
        # 统计每个实体在热点文档中的出现次数
        entity_hot_count = defaultdict(int)
        entity_total_count = defaultdict(int)
        
        # 计算总批次数
        num_batches = (len(df) + batch_size - 1) // batch_size
        
        for batch_idx in tqdm(range(num_batches), desc="计算节点标签"):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, len(df))
            batch_df = df.iloc[start_idx:end_idx]
            
            for _, row in batch_df.iterrows():
                entities = row['entities']
                if not isinstance(entities, list):
                    try:
                        entities = eval(entities)
                    except:
                        entities = []
                        
                is_hot = row['is_hot']
                
                for entity in entities:
                    if entity in self.entity_dict:
                        entity_idx = self.entity_dict[entity]
                        entity_total_count[entity_idx] += 1
                        if is_hot:
                            entity_hot_count[entity_idx] += 1
            
            # 每10个批次清理内存
            if batch_idx % 10 == 0:
                gc.collect()
        
        # 计算所有实体的热点率
        hot_ratios = []
        for entity_idx in range(self.num_entities):
            if entity_total_count[entity_idx] > 0:
                hot_ratio = entity_hot_count[entity_idx] / entity_total_count[entity_idx]
                hot_ratios.append((entity_idx, hot_ratio))
        
        # 排序
        hot_ratios.sort(key=lambda x: x[1], reverse=True)
        
        # 取前30%的实体作为热点
        num_hot = max(1, int(self.num_entities * 0.3))  # 使用硬编码0.3
        logger.info(f"选择前 {num_hot} 个实体作为热点 (比例: {0.3:.2f})")
        
        for i in range(num_hot):
            if i < len(hot_ratios):
                entity_idx, _ = hot_ratios[i]
                node_labels[entity_idx] = 1
        
        hot_count = node_labels.sum().item()
        logger.info(f"节点标签计算完成，共 {hot_count} 个热点节点，{self.num_entities - hot_count} 个非热点节点")
        return node_labels
    
    def process_data_to_graph(self, batch_size: int = 10000) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        从处理好的数据构建图，针对大数据集优化
        
        Args:
            batch_size: 每批处理的数据量
            
        Returns:
            (节点特征, 边索引, 边权重, 节点标签)
        """
        # 加载处理后的数据
        train_file = os.path.join(self.data_dir, f"{self.data_prefix}_train.csv")
        val_file = os.path.join(self.data_dir, f"{self.data_prefix}_val.csv")
        test_file = os.path.join(self.data_dir, f"{self.data_prefix}_test.csv")
        
        logger.info(f"加载训练数据: {train_file}")
        train_df = pd.read_csv(train_file)
        
        logger.info(f"加载验证数据: {val_file}")
        val_df = pd.read_csv(val_file)
        
        logger.info(f"加载测试数据: {test_file}")
        test_df = pd.read_csv(test_file)
        
        # 合并所有数据用于构建图
        logger.info("合并数据...")
        all_df = pd.concat([train_df, val_df, test_df])
        
        # 构建共现关系图
        nx_graph = self.build_cooccurrence_graph(all_df, batch_size)
        
        # 转换为PyTorch张量
        node_features, edge_index, edge_weight = self.convert_graph_to_pytorch(nx_graph)
        
        # 添加节点标签
        node_labels = self.compute_node_labels(all_df, batch_size)
        
        # 保存图数据
        logger.info("保存图数据...")
        self.save_graph_data(node_features, edge_index, edge_weight, node_labels, 
                             os.path.join(self.data_dir, f"{self.data_prefix}_graph_data.pt"))
        
        logger.info("保存NetworkX图...")
        # 可选：保存NetworkX图(用于可视化)
        self.save_networkx_graph(nx_graph, os.path.join(self.data_dir, f"{self.data_prefix}_nx_graph.gpickle"))
        
        return node_features, edge_index, edge_weight, node_labels
    
    def save_graph_data(self, node_features: torch.Tensor, edge_index: torch.Tensor, 
                        edge_weight: torch.Tensor, node_labels: torch.Tensor, path: str) -> None:
        """
        保存图数据
        
        Args:
            node_features: 节点特征
            edge_index: 边索引
            edge_weight: 边权重
            node_labels: 节点标签
            path: 保存路径
        """
        logger.info(f"保存图数据到 {path}")
        torch.save({
            'node_features': node_features,
            'edge_index': edge_index,
            'edge_weight': edge_weight,
            'node_labels': node_labels,
        }, path)
    
    def save_networkx_graph(self, G: nx.Graph, path: str) -> None:
        """
        保存NetworkX图
        
        Args:
            G: NetworkX图对象
            path: 保存路径
        """
        logger.info(f"保存NetworkX图到 {path}")
        # 使用pickle模块直接保存
        import pickle
        with open(path, 'wb') as f:
            pickle.dump(G, f)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="大规模图构建工具")
    parser.add_argument("--batch_size", type=int, default=10000,
                        help="每批处理的数据量")
    parser.add_argument("--data_prefix", type=str, default="mind_full",
                        help="数据前缀")
    
    args = parser.parse_args()
    
    # 创建图构建器
    builder = LargeGraphBuilder(data_prefix=args.data_prefix)
    
    # 处理数据并构建图
    node_features, edge_index, edge_weight, node_labels = builder.process_data_to_graph(args.batch_size)
    
    # 统计信息
    logger.info(f"节点数: {node_features.shape[0]}")
    logger.info(f"边数: {edge_index.shape[1]//2} (双向: {edge_index.shape[1]})")
    logger.info(f"热点节点比例: {node_labels.sum().item() / len(node_labels):.2f}")
    
    logger.info("图构建完成!")
