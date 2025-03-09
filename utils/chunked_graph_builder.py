"""
分片数据图构建脚本 - 处理多个CSV文件
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
import gc

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

class ChunkedGraphBuilder:
    """处理分片数据的图构建类"""
    
    def __init__(self, entity_dict: Optional[Dict[str, int]] = None,
                data_dir: str = PROCESSED_DATA_DIR,
                data_prefix: str = "mind_full"):
        """初始化图构建器"""
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
        
        logger.info(f"分片数据图构建器初始化完成，共 {self.num_entities} 个实体")
    
    def get_chunk_files(self, split_name: str) -> List[str]:
        """获取分片文件列表"""
        index_file = os.path.join(self.data_dir, f"{self.data_prefix}_{split_name}.txt")
        
        if os.path.exists(index_file):
            # 从索引文件读取
            with open(index_file, 'r') as f:
                chunk_files = [line.strip() for line in f]
            return chunk_files
        else:
            # 直接查找匹配的文件
            all_files = os.listdir(self.data_dir)
            chunk_files = [f for f in all_files if f.startswith(f"{self.data_prefix}_{split_name}_") and f.endswith(".csv")]
            chunk_files.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))  # 按数字顺序排序
            return chunk_files
    
    def build_cooccurrence_graph(self, chunks_info: List[Tuple[str, str]], batch_size: int = 5000) -> nx.Graph:
        """构建共现关系图，处理多个CSV文件"""
        logger.info("构建共现关系图")
        
        # 创建无向图
        G = nx.Graph()
        
        # 添加所有实体作为节点
        for entity_idx in range(self.num_entities):
            G.add_node(entity_idx, name=self.idx_to_entity[entity_idx])
        
        # 统计实体共现次数
        cooccurrence = defaultdict(int)
        total_processed = 0
        
        # 处理每个分片文件
        for split_name, chunk_file in tqdm(chunks_info, desc="处理数据分片"):
            file_path = os.path.join(self.data_dir, chunk_file)
            
            # 分批读取CSV文件
            for chunk_df in pd.read_csv(file_path, chunksize=batch_size):
                for _, row in chunk_df.iterrows():
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
                
                total_processed += len(chunk_df)
                
                # 清理内存
                del chunk_df
                gc.collect()
            
            # 每处理完一个文件清理一次内存
            gc.collect()
            logger.info(f"处理了 {total_processed} 条记录")
        
        # 添加边，权重为共现次数
        logger.info("添加边到图...")
        edge_count = 0
        for (entity_i, entity_j), weight in tqdm(cooccurrence.items(), desc="添加边"):
            if weight >= MIN_EDGE_WEIGHT:
                G.add_edge(entity_i, entity_j, weight=weight)
                edge_count += 1
                
                # 每添加1000条边清理一次内存
                if edge_count % 1000 == 0:
                    gc.collect()
        
        # 清理内存
        del cooccurrence
        gc.collect()
        
        logger.info(f"共现关系图构建完成，共 {G.number_of_nodes()} 个节点，{G.number_of_edges()} 条边")
        return G
    
    def convert_graph_to_pytorch(self, nx_graph: nx.Graph) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """将NetworkX图转换为PyTorch张量"""
        logger.info("将图转换为PyTorch张量")
        
        # 获取边列表和权重
        edge_count = nx_graph.number_of_edges()
        src_nodes = torch.zeros(edge_count * 2, dtype=torch.long)
        dst_nodes = torch.zeros(edge_count * 2, dtype=torch.long)
        edge_weights = torch.zeros(edge_count * 2, dtype=torch.float)
        
        # 分批处理边以节省内存
        for i, (u, v, data) in enumerate(tqdm(nx_graph.edges(data=True), desc="处理边")):
            # 正向边
            src_nodes[i] = u
            dst_nodes[i] = v
            edge_weights[i] = data['weight']
            
            # 反向边
            src_nodes[i + edge_count] = v
            dst_nodes[i + edge_count] = u
            edge_weights[i + edge_count] = data['weight']
            
            # 每处理1000条边清理一次内存
            if i % 1000 == 0:
                gc.collect()
        
        # 构建边索引
        edge_index = torch.stack([src_nodes, dst_nodes], dim=0)
        
        # 创建节点特征（随机初始化）
        logger.info(f"创建节点特征 (维度: {INPUT_DIM})...")
        node_features = torch.randn(self.num_entities, INPUT_DIM)
        
        logger.info(f"转换完成，边数: {edge_count} (双向: {edge_count*2})")
        return node_features, edge_index, edge_weights
    
    def compute_node_labels(self, chunks_info: List[Tuple[str, str]], batch_size: int = 5000) -> torch.Tensor:
        """计算节点标签，处理多个CSV文件"""
        logger.info("计算节点标签")
        
        # 初始化所有节点标签为0（非热点）
        node_labels = torch.zeros(self.num_entities, dtype=torch.long)
        
        # 统计每个实体在热点文档中的出现次数
        entity_hot_count = defaultdict(int)
        entity_total_count = defaultdict(int)
        total_processed = 0
        
        # 处理每个分片文件
        for split_name, chunk_file in tqdm(chunks_info, desc="处理数据分片"):
            file_path = os.path.join(self.data_dir, chunk_file)
            
            # 分批读取CSV文件
            for chunk_df in pd.read_csv(file_path, chunksize=batch_size):
                for _, row in chunk_df.iterrows():
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
                
                total_processed += len(chunk_df)
                
                # 清理内存
                del chunk_df
                gc.collect()
            
            # 每处理完一个文件清理一次内存
            gc.collect()
            logger.info(f"处理了 {total_processed} 条记录")
        
        # 计算所有实体的热点率
        hot_ratios = []
        for entity_idx in range(self.num_entities):
            if entity_total_count[entity_idx] > 0:
                hot_ratio = entity_hot_count[entity_idx] / entity_total_count[entity_idx]
                hot_ratios.append((entity_idx, hot_ratio))
        
        # 排序
        hot_ratios.sort(key=lambda x: x[1], reverse=True)
        
        # 取前30%的实体作为热点
        num_hot = max(1, int(self.num_entities * 0.3))
        logger.info(f"选择前 {num_hot} 个实体作为热点 (比例: {0.3:.2f})")
        
        for i in range(num_hot):
            if i < len(hot_ratios):
                entity_idx, _ = hot_ratios[i]
                node_labels[entity_idx] = 1
        
        hot_count = node_labels.sum().item()
        logger.info(f"节点标签计算完成，共 {hot_count} 个热点节点，{self.num_entities - hot_count} 个非热点节点")
        return node_labels
    
    def process_data_to_graph(self, batch_size: int = 5000, max_files: int = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """处理数据并构建图，支持分片文件"""
        # 获取所有分片文件
        train_chunks = self.get_chunk_files("train")
        val_chunks = self.get_chunk_files("val")
        test_chunks = self.get_chunk_files("test")
        
        logger.info(f"找到训练集分片: {len(train_chunks)} 个文件")
        logger.info(f"找到验证集分片: {len(val_chunks)} 个文件")
        logger.info(f"找到测试集分片: {len(test_chunks)} 个文件")
        
        # 限制文件数量以减少处理时间（可选）
        if max_files:
            train_chunks = train_chunks[:max_files]
            val_chunks = val_chunks[:min(max_files//4, len(val_chunks))]
            test_chunks = test_chunks[:min(max_files//4, len(test_chunks))]
            logger.info(f"限制处理文件数: 训练集 {len(train_chunks)}, 验证集 {len(val_chunks)}, 测试集 {len(test_chunks)}")
        
        # 组合所有分片信息
        all_chunks = [("train", chunk) for chunk in train_chunks] + \
                     [("val", chunk) for chunk in val_chunks] + \
                     [("test", chunk) for chunk in test_chunks]
        
        # 构建共现关系图
        logger.info(f"开始构建图，使用 {len(all_chunks)} 个数据分片...")
        nx_graph = self.build_cooccurrence_graph(all_chunks, batch_size)
        
        # 转换为PyTorch张量
        node_features, edge_index, edge_weight = self.convert_graph_to_pytorch(nx_graph)
        
        # 计算节点标签
        node_labels = self.compute_node_labels(all_chunks, batch_size)
        
        # 保存图数据
        logger.info("保存图数据...")
        self.save_graph_data(node_features, edge_index, edge_weight, node_labels, 
                             os.path.join(self.data_dir, f"{self.data_prefix}_graph_data.pt"))
        
        # 保存NetworkX图
        logger.info("保存NetworkX图...")
        self.save_networkx_graph(nx_graph, os.path.join(self.data_dir, f"{self.data_prefix}_nx_graph.gpickle"))
        
        return node_features, edge_index, edge_weight, node_labels
    
    def save_graph_data(self, node_features: torch.Tensor, edge_index: torch.Tensor, 
                        edge_weight: torch.Tensor, node_labels: torch.Tensor, path: str) -> None:
        """保存图数据"""
        logger.info(f"保存图数据到 {path}")
        torch.save({
            'node_features': node_features,
            'edge_index': edge_index,
            'edge_weight': edge_weight,
            'node_labels': node_labels,
        }, path)
    
    def save_networkx_graph(self, G: nx.Graph, path: str) -> None:
        """保存NetworkX图"""
        logger.info(f"保存NetworkX图到 {path}")
        # 使用pickle模块直接保存
        import pickle
        with open(path, 'wb') as f:
            pickle.dump(G, f)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="分片数据图构建工具")
    parser.add_argument("--batch_size", type=int, default=5000,
                        help="每批处理的数据量")
    parser.add_argument("--data_prefix", type=str, default="mind_full",
                        help="数据前缀")
    parser.add_argument("--max_files", type=int, default=None,
                        help="每个数据集最多处理的文件数，用于快速测试")
    
    args = parser.parse_args()
    
    # 创建图构建器
    builder = ChunkedGraphBuilder(data_prefix=args.data_prefix)
    
    # 处理数据并构建图
    node_features, edge_index, edge_weight, node_labels = builder.process_data_to_graph(
        args.batch_size, args.max_files)
    
    # 统计信息
    logger.info(f"节点数: {node_features.shape[0]}")
    logger.info(f"边数: {edge_index.shape[1]//2} (双向: {edge_index.shape[1]})")
    logger.info(f"热点节点比例: {node_labels.sum().item() / len(node_labels):.2f}")
    
    logger.info("图构建完成!") 