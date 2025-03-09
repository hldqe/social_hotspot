"""
简化版图构建模块，仅使用NetworkX和PyTorch
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

class SimpleGraphBuilder:
    """简化图构建类，使用NetworkX和PyTorch"""
    
    def __init__(self, entity_dict: Optional[Dict[str, int]] = None,
                data_dir: str = PROCESSED_DATA_DIR):
        """
        初始化图构建器
        
        Args:
            entity_dict: 实体词典 {实体: 索引}
            data_dir: 数据目录
        """
        self.data_dir = data_dir
        
        # 加载实体词典
        if entity_dict is None:
            entity_dict_path = os.path.join(data_dir, "entity_dict.json")
            if os.path.exists(entity_dict_path):
                with open(entity_dict_path, 'r') as f:
                    self.entity_dict = json.load(f)
            else:
                raise FileNotFoundError(f"实体词典不存在: {entity_dict_path}")
        else:
            self.entity_dict = entity_dict
            
        self.idx_to_entity = {idx: entity for entity, idx in self.entity_dict.items()}
        self.num_entities = len(self.entity_dict)
        
        logger.info(f"图构建器初始化完成，共 {self.num_entities} 个实体")
    
    def build_cooccurrence_graph(self, df: pd.DataFrame) -> nx.Graph:
        """
        构建共现关系图
        
        Args:
            df: 包含实体列表的DataFrame
            
        Returns:
            NetworkX图对象
        """
        logger.info("构建共现关系图")
        
        # 创建无向图
        G = nx.Graph()
        
        # 添加所有实体作为节点
        for entity_idx in range(self.num_entities):
            G.add_node(entity_idx, name=self.idx_to_entity[entity_idx])
        
        # 统计实体共现次数
        cooccurrence = defaultdict(int)
        
        # 遍历每篇文档的实体列表
        for _, row in tqdm(df.iterrows(), total=len(df), desc="构建共现关系"):
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
        
        # 添加边，权重为共现次数
        for (entity_i, entity_j), weight in cooccurrence.items():
            if weight >= MIN_EDGE_WEIGHT:
                G.add_edge(entity_i, entity_j, weight=weight)
        
        logger.info(f"共现关系图构建完成，共 {G.number_of_nodes()} 个节点，{G.number_of_edges()} 条边")
        return G
    
    def convert_graph_to_pytorch(self, nx_graph: nx.Graph) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        将NetworkX图转换为PyTorch张量
        
        Args:
            nx_graph: NetworkX图对象
            
        Returns:
            (边源节点张量, 边目标节点张量, 边权重张量)
        """
        logger.info("将图转换为PyTorch张量")
        
        # 获取边列表和权重
        src_nodes = []
        dst_nodes = []
        edge_weights = []
        
        for u, v, data in nx_graph.edges(data=True):
            src_nodes.append(u)
            dst_nodes.append(v)
            edge_weights.append(data['weight'])
            # 对于无向图，添加反向边
            src_nodes.append(v)
            dst_nodes.append(u)
            edge_weights.append(data['weight'])
        
        # 转换为PyTorch张量
        edge_index = torch.tensor([src_nodes, dst_nodes], dtype=torch.long)
        edge_weight = torch.tensor(edge_weights, dtype=torch.float)
        
        # 创建节点特征（随机初始化）
        node_features = torch.randn(self.num_entities, INPUT_DIM)
        
        logger.info(f"转换完成，边数: {len(edge_weights)//2} (双向: {len(edge_weights)})")
        return node_features, edge_index, edge_weight
    
    def compute_node_labels(self, df: pd.DataFrame) -> torch.Tensor:
        """
        计算节点标签（热点/非热点）
        
        Args:
            df: 包含热点标签的DataFrame
            
        Returns:
            节点标签张量
        """
        logger.info("计算节点标签")
        
        # 初始化所有节点标签为0（非热点）
        node_labels = torch.zeros(self.num_entities, dtype=torch.long)
        
        # 统计每个实体在热点文档中的出现次数
        entity_hot_count = defaultdict(int)
        entity_total_count = defaultdict(int)
        
        for _, row in df.iterrows():
            entities = row['entities']
            if not isinstance(entities, list):
                try:
                    # 尝试将字符串转换为列表
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
        
        # 计算所有实体的热点率
        hot_ratios = []
        for entity_idx in range(self.num_entities):
            if entity_total_count[entity_idx] > 0:
                hot_ratio = entity_hot_count[entity_idx] / entity_total_count[entity_idx]
                hot_ratios.append((entity_idx, hot_ratio))
        
        # 排序
        hot_ratios.sort(key=lambda x: x[1], reverse=True)
        
        # 取前30%的实体作为热点
        num_hot = max(1, int(self.num_entities * 0.3))  # 至少1个热点
        for i in range(num_hot):
            if i < len(hot_ratios):
                entity_idx, _ = hot_ratios[i]
                node_labels[entity_idx] = 1
        
        hot_count = node_labels.sum().item()
        logger.info(f"节点标签计算完成，共 {hot_count} 个热点节点，{self.num_entities - hot_count} 个非热点节点")
        return node_labels
    
    def normalize_features(self, features: torch.Tensor) -> torch.Tensor:
        """
        归一化节点特征
        
        Args:
            features: 节点特征张量
            
        Returns:
            归一化后的特征
        """
        logger.info("归一化节点特征")
        
        # 计算均值和标准差
        mean = torch.mean(features, dim=0)
        std = torch.std(features, dim=0)
        
        # 防止除零
        std = torch.where(std > 0, std, torch.ones_like(std))
        
        # 归一化
        normalized_features = (features - mean) / std
        
        return normalized_features
    
    def process_data_to_graph(self, data_prefix: str = "mind") -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        从处理好的数据构建图
        
        Args:
            data_prefix: 数据文件前缀
            
        Returns:
            (节点特征, 边索引, 边权重, 节点标签)
        """
        # 加载处理后的数据
        train_file = os.path.join(self.data_dir, f"{data_prefix}_train.csv")
        val_file = os.path.join(self.data_dir, f"{data_prefix}_val.csv")
        test_file = os.path.join(self.data_dir, f"{data_prefix}_test.csv")
        
        train_df = pd.read_csv(train_file)
        val_df = pd.read_csv(val_file)
        test_df = pd.read_csv(test_file)
        
        # 合并所有数据用于构建图
        all_df = pd.concat([train_df, val_df, test_df])
        
        # 构建共现关系图
        nx_graph = self.build_cooccurrence_graph(all_df)
        
        # 转换为PyTorch张量
        node_features, edge_index, edge_weight = self.convert_graph_to_pytorch(nx_graph)
        
        # 添加节点标签
        node_labels = self.compute_node_labels(all_df)
        
        # 归一化特征
        node_features = self.normalize_features(node_features)
        
        # 保存图数据
        self.save_graph_data(node_features, edge_index, edge_weight, node_labels, 
                            os.path.join(self.data_dir, f"{data_prefix}_graph_data.pt"))
        
        # 可选：保存NetworkX图(用于可视化)
        self.save_networkx_graph(nx_graph, os.path.join(self.data_dir, f"{data_prefix}_nx_graph.gpickle"))
        
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
    
    def load_graph_data(self, path: str) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        加载图数据
        
        Args:
            path: 图数据文件路径
            
        Returns:
            (节点特征, 边索引, 边权重, 节点标签)
        """
        logger.info(f"从 {path} 加载图数据")
        data = torch.load(path)
        return data['node_features'], data['edge_index'], data['edge_weight'], data['node_labels']
    
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
    
    def load_networkx_graph(self, path: str) -> nx.Graph:
        """
        加载NetworkX图
        
        Args:
            path: 图文件路径
            
        Returns:
            NetworkX图对象
        """
        logger.info(f"从 {path} 加载NetworkX图")
        # 使用pickle模块直接加载
        import pickle
        with open(path, 'rb') as f:
            return pickle.load(f)
    
    def visualize_graph(self, G: nx.Graph, output_path: str, max_nodes: int = VIZ_MAX_NODES) -> None:
        """
        可视化图结构
        
        Args:
            G: NetworkX图对象
            output_path: 输出图像路径
            max_nodes: 最大可视化节点数
        """
        logger.info("可视化图结构")
        
        # 获取图中的热点标签
        # 通过加载已保存的标签或重新计算
        try:
            _, _, _, node_labels = self.load_graph_data(
                os.path.join(self.data_dir, f"mind_graph_data.pt"))
            is_hot = {i: label.item() == 1 for i, label in enumerate(node_labels)}
        except:
            # 如果无法加载标签，将所有节点视为非热点
            is_hot = {i: False for i in range(self.num_entities)}
        
        # 如果节点太多，只可视化一部分
        if G.number_of_nodes() > max_nodes:
            # 优先选择热点节点
            hot_nodes = [n for n in G.nodes() if is_hot.get(n, False)]
            
            # 限制热点节点数量
            hot_nodes = hot_nodes[:max_nodes//2]
            
            # 如果热点节点不足，选择度最大的节点补充
            if len(hot_nodes) < max_nodes//2:
                non_hot_nodes = [n for n in G.nodes() if not is_hot.get(n, False)]
                # 按度数排序
                non_hot_nodes = sorted(non_hot_nodes, key=lambda n: G.degree(n), reverse=True)
                # 补充节点
                nodes_to_add = non_hot_nodes[:(max_nodes//2 - len(hot_nodes))]
                hot_nodes.extend(nodes_to_add)
            
            # 获取这些节点的一阶邻居
            neighbors = set()
            for node in hot_nodes:
                neighbors.update(G.neighbors(node))
            
            # 限制邻居节点数量
            neighbors = list(neighbors)[:max_nodes - len(hot_nodes)]
            
            # 合并热点节点和邻居节点
            nodes_to_viz = list(set(hot_nodes + neighbors))
            
            # 创建子图
            G = G.subgraph(nodes_to_viz)
        
        # 获取边权重用于线宽
        edge_weights = [G[u][v]['weight'] * VIZ_EDGE_WIDTH_FACTOR for u, v in G.edges()]
        
        # 获取节点标签用于颜色
        node_colors = ['red' if is_hot.get(n, False) else 'blue' for n in G.nodes()]
        
        # 获取节点度用于大小
        node_sizes = [G.degree(node) * VIZ_NODE_SIZE_FACTOR + 100 for node in G.nodes()]
        
        # 创建图形
        plt.figure(figsize=(12, 12))
        
        # 使用spring_layout来布局
        pos = nx.spring_layout(G, seed=42)  # 使用固定随机种子42替代RANDOM_SEED
        
        # 绘制图
        nx.draw_networkx(
            G, pos=pos,
            node_color=node_colors,
            node_size=node_sizes,
            edge_color='gray',
            width=edge_weights,
            alpha=0.8,
            with_labels=False,
        )
        
        # 添加图例
        plt.plot([0], [0], 'o', color='red', label='热点实体')
        plt.plot([0], [0], 'o', color='blue', label='非热点实体')
        plt.legend(fontsize=12)
        
        # 添加标题
        plt.title(f'实体关系图（展示 {G.number_of_nodes()} 个节点, {G.number_of_edges()} 条边）', fontsize=15)
        
        # 保存图像
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"图结构可视化已保存到 {output_path}")

    def save_pytorch_graph_data(self, prefix, graph, node_labels, node_features):
        """
        将图数据保存为PyTorch格式
        
        Args:
            prefix: 数据前缀
            graph: NetworkX图
            node_labels: 节点标签
            node_features: 节点特征
        """
        # 创建边索引和边权重张量
        edges = list(graph.edges(data=True))
        num_edges = len(edges)
        edge_index = torch.zeros((2, num_edges * 2), dtype=torch.long)  # 双向边
        edge_weight = torch.zeros(num_edges * 2, dtype=torch.float)
        
        for i, (src, dst, data) in enumerate(edges):
            # 正向边
            edge_index[0, i] = src
            edge_index[1, i] = dst
            edge_weight[i] = data.get('weight', 1.0)
            
            # 反向边
            edge_index[0, i + num_edges] = dst
            edge_index[1, i + num_edges] = src
            edge_weight[i + num_edges] = data.get('weight', 1.0)
        
        # 保存为PyTorch张量
        graph_data = {
            'node_features': node_features,
            'edge_index': edge_index,
            'edge_weight': edge_weight,
            'node_labels': node_labels
        }
        
        save_path = os.path.join(PROCESSED_DATA_DIR, f"{prefix}_graph_data.pt")
        torch.save(graph_data, save_path)
        logger.info(f"PyTorch图数据已保存: {save_path}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="实体关系图构建工具")
    parser.add_argument("--data_prefix", type=str, default="mind",
                        help="数据文件前缀")
    parser.add_argument("--visualize", action="store_true",
                        help="是否可视化图")
    
    args = parser.parse_args()
    
    # 创建图构建器
    builder = SimpleGraphBuilder()
    
    # 处理数据并构建图
    node_features, edge_index, edge_weight, node_labels = builder.process_data_to_graph(args.data_prefix)
    
    # 统计信息
    logger.info(f"节点数: {node_features.shape[0]}")
    logger.info(f"边数: {edge_index.shape[1]//2} (双向: {edge_index.shape[1]})")
    logger.info(f"热点节点比例: {node_labels.sum().item() / len(node_labels):.2f}")
    
    # 可视化图结构
    if args.visualize:
        os.makedirs(RESULT_DIR, exist_ok=True)
        # 加载NetworkX图
        nx_graph = builder.load_networkx_graph(os.path.join(builder.data_dir, f"{args.data_prefix}_nx_graph.gpickle"))
        # 可视化
        builder.visualize_graph(nx_graph, os.path.join(RESULT_DIR, f"{args.data_prefix}_graph.png"))
    
    logger.info("图构建完成!")
