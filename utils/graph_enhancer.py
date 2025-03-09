"""
图数据增强工具
"""

import torch
import numpy as np
import logging

logger = logging.getLogger(__name__)

def enhance_graph(edge_index, num_nodes):
    """增强图结构，添加自环和虚拟边
    
    Args:
        edge_index: 原始边索引张量 [2, num_edges]
        num_nodes: 节点数量
        
    Returns:
        增强后的边索引张量
    """
    # 添加自环
    loop_index = torch.arange(num_nodes, device=edge_index.device)
    loop_index = loop_index.repeat(2, 1)  # [2, num_nodes]
    
    # 合并原始边和自环
    enhanced_edge_index = torch.cat([edge_index, loop_index], dim=1)
    
    # 减少虚拟边的比例从5%到2%
    num_edges = edge_index.size(1)
    num_virtual = int(num_edges * 0.02)  # 从0.05改为0.02
    
    # 生成随机边
    virtual_src = torch.randint(0, num_nodes, (num_virtual,), device=edge_index.device)
    virtual_dst = torch.randint(0, num_nodes, (num_virtual,), device=edge_index.device)
    virtual_edges = torch.stack([virtual_src, virtual_dst], dim=0)
    
    # 合并所有边
    enhanced_edge_index = torch.cat([enhanced_edge_index, virtual_edges], dim=1)
    
    logger.info(f"图增强: 原始边数={num_edges}, 添加自环={num_nodes}, "
               f"添加虚拟边={num_virtual}, 总边数={enhanced_edge_index.size(1)}")
    
    return enhanced_edge_index

def add_feature_noise(features, noise_level=0.05):
    """添加特征噪声进行数据增强
    
    Args:
        features: 节点特征张量
        noise_level: 噪声强度
        
    Returns:
        添加噪声后的特征张量
    """
    noise = torch.randn_like(features) * noise_level
    return features + noise

def enhance_graph_structure(G):
    """增强图结构，增加有意义的连接"""
    # 计算节点相似度，添加更多边
    import numpy as np
    
    # 获取所有节点的度
    degrees = dict(G.degree())
    
    # 为低度节点添加更多连接
    low_degree_nodes = [n for n, d in degrees.items() if d < 3]
    
    # 基于共同邻居添加边
    for node in low_degree_nodes:
        neighbors = set(G.neighbors(node))
        potential_connections = []
        
        for other in G.nodes():
            if other != node and other not in neighbors:
                other_neighbors = set(G.neighbors(other))
                common = len(neighbors.intersection(other_neighbors))
                if common > 0:
                    potential_connections.append((other, common))
        
        # 添加最相似的节点作为边
        potential_connections.sort(key=lambda x: x[1], reverse=True)
        for other, _ in potential_connections[:3]:  # 每个低度节点最多添加3条边
            G.add_edge(node, other, weight=1.0)
    
    return G