import torch
import numpy as np
from torch_geometric.utils import degree
from torch_geometric.utils import add_self_loops
import networkx as nx
from tqdm import tqdm
import logging

logger = logging.getLogger(__name__)

def extract_node_features(features, edge_index, labels=None):
    """
    增强节点特征：添加图结构特征和节点中心性特征
    
    Args:
        features: 原始节点特征 [num_nodes, feat_dim]
        edge_index: 边索引 [2, num_edges]
        labels: 节点标签（用于统计聚合特征） [num_nodes]
        
    Returns:
        增强后的节点特征 [num_nodes, new_feat_dim]
    """
    logger.info("开始增强节点特征...")
    num_nodes = features.size(0)
    device = features.device
    
    # 转换为NetworkX图
    edges = [(edge_index[0, i].item(), edge_index[1, i].item()) 
             for i in range(edge_index.size(1))]
    G = nx.Graph(edges)
    # 确保图包含所有节点
    G.add_nodes_from(range(num_nodes))
    
    logger.info("计算图拓扑特征...")
    # 计算度中心性
    degree_cent = torch.tensor([G.degree(i) for i in range(num_nodes)], dtype=torch.float)
    degree_cent = degree_cent.view(-1, 1)
    
    logger.info("计算邻居聚合特征...")
    # 计算聚类系数
    clustering = torch.zeros(num_nodes, 1)
    nx_clustering = nx.clustering(G)
    for i in range(num_nodes):
        clustering[i, 0] = nx_clustering.get(i, 0.0)
    
    logger.info("计算PageRank特征...")
    # 计算PageRank
    pagerank = torch.zeros(num_nodes, 1)
    nx_pagerank = nx.pagerank(G, alpha=0.85)
    for i in range(num_nodes):
        pagerank[i, 0] = nx_pagerank.get(i, 0.0)
    
    logger.info("计算中介中心性特征...")
    # 计算中介中心性 - 使用直接计算而非近似算法
    betweenness = torch.zeros(num_nodes, 1)
    try:
        # 直接使用NetworkX的betweenness_centrality函数
        if num_nodes < 500:  # 小图直接计算
            nx_betweenness = nx.betweenness_centrality(G)
        else:  # 大图使用抽样
            nx_betweenness = nx.betweenness_centrality(G, k=min(20, num_nodes-1))
            
        for i in range(num_nodes):
            betweenness[i, 0] = nx_betweenness.get(i, 0.0)
        logger.info("中介中心性计算成功")
    except Exception as e:
        logger.warning(f"中介中心性计算失败: {str(e)}，使用节点度作为替代")
        # 使用节点度作为替代特征
        nx_betweenness = dict(G.degree())
        max_degree = max(nx_betweenness.values()) if nx_betweenness else 1
        for i in range(num_nodes):
            betweenness[i, 0] = nx_betweenness.get(i, 0.0) / max(max_degree, 1)
    
    # 添加k-core分解特征
    kcore = torch.zeros(num_nodes, 1)
    try:
        nx_kcore = nx.core_number(G)
        for i in range(num_nodes):
            kcore[i, 0] = nx_kcore.get(i, 0.0)
        logger.info("K-core分解计算成功")
    except Exception as e:
        logger.warning(f"K-core分解计算失败: {str(e)}")
    
    logger.info("组合所有特征...")
    # 组合所有特征
    enhanced_features = torch.cat([
        features,        # 原始特征
        degree_cent,     # 度中心性
        clustering,      # 聚类系数 
        pagerank,        # PageRank
        betweenness,     # 中介中心性
        kcore            # K-core分解
    ], dim=1)
    
    logger.info(f"特征增强完成, 原始维度: {features.size(1)}, 增强后维度: {enhanced_features.size(1)}")
    
    return enhanced_features

def add_virtual_connections(edge_index, labels, num_nodes, ratio=0.1):
    """
    添加虚拟连接，优先连接同类节点
    
    Args:
        edge_index: 原始边索引
        labels: 节点标签
        num_nodes: 节点数量
        ratio: 添加边的比例
        
    Returns:
        增强后的边索引
    """
    logger.info("开始添加虚拟连接...")
    
    # 转为CPU计算
    edge_index_cpu = edge_index.cpu()
    labels_cpu = labels.cpu()
    
    # 获取现有边的集合，用于避免重复
    existing_edges = set()
    for i in range(edge_index_cpu.size(1)):
        src, dst = edge_index_cpu[0, i].item(), edge_index_cpu[1, i].item()
        existing_edges.add((src, dst))
        existing_edges.add((dst, src))  # 考虑无向图
    
    # 计算要添加的边数
    num_edges = edge_index_cpu.size(1)
    num_new_edges = int(num_edges * ratio)
    
    # 找出所有热点节点
    hotspot_nodes = [i for i in range(num_nodes) if labels_cpu[i] == 1]
    num_hotspots = len(hotspot_nodes)
    
    if num_hotspots > 0:
        logger.info(f"找到 {num_hotspots} 个热点节点，添加优先连接")
        
        # 创建新边列表
        new_edges = []
        
        # 策略1: 连接热点节点和其它热点节点
        added = 0
        attempts = 0
        max_attempts = num_new_edges * 10  # 防止无限循环
        
        while added < num_new_edges * 0.6 and attempts < max_attempts:
            src = hotspot_nodes[np.random.randint(0, num_hotspots)]
            dst = hotspot_nodes[np.random.randint(0, num_hotspots)]
            
            if src != dst and (src, dst) not in existing_edges:
                new_edges.append((src, dst))
                existing_edges.add((src, dst))
                existing_edges.add((dst, src))
                added += 1
            
            attempts += 1
        
        # 策略2: 随机添加其余的边
        remaining = num_new_edges - added
        attempts = 0
        
        while len(new_edges) < num_new_edges and attempts < max_attempts:
            src = np.random.randint(0, num_nodes)
            dst = np.random.randint(0, num_nodes)
            
            if src != dst and (src, dst) not in existing_edges:
                new_edges.append((src, dst))
                existing_edges.add((src, dst))
                existing_edges.add((dst, src))
            
            attempts += 1
        
        # 转换为张量并合并
        if new_edges:
            new_edges_tensor = torch.tensor(new_edges, dtype=torch.long).t()
            enhanced_edge_index = torch.cat([edge_index_cpu, new_edges_tensor], dim=1)
            logger.info(f"添加了 {len(new_edges)} 条虚拟连接, 总边数: {enhanced_edge_index.size(1)}")
            return enhanced_edge_index.to(edge_index.device)
    
    logger.info("未添加虚拟连接")
    return edge_index 

def extract_topic_features(features, edge_index, labels=None, topic_data=None):
    """增加主题相关特征"""
    # 基础特征提取
    enhanced_features = extract_node_features(features, edge_index, labels)
    
    # 如果有主题数据，添加主题相关特征
    if topic_data is not None:
        # 提取主题流行度特征
        topic_popularity = calculate_topic_popularity(topic_data)
        # 提取主题传播速度特征
        topic_velocity = calculate_topic_velocity(topic_data)
        
        # 合并主题特征
        topic_features = torch.cat([
            topic_popularity.unsqueeze(1),
            topic_velocity.unsqueeze(1)
        ], dim=1)
        
        # 合并到现有特征
        enhanced_features = torch.cat([enhanced_features, topic_features], dim=1)
    
    return enhanced_features 

def enhance_features(features, edge_index, noise_scale=0.1):
    """增强节点特征"""
    num_nodes = features.size(0)  # 获取节点数量
    
    # 添加随机噪声
    noise = torch.randn_like(features) * noise_scale
    features = features + noise
    
    # 计算节点度
    deg = degree(edge_index[0], num_nodes=num_nodes)  # 确保使用正确的节点数
    deg_features = deg.view(-1, 1)
    
    # 计算局部结构特征
    local_struct = compute_local_structure(edge_index, num_nodes)
    
    # 确保所有特征维度匹配
    assert deg_features.size(0) == num_nodes, f"度特征维度不匹配: {deg_features.size(0)} vs {num_nodes}"
    assert local_struct.size(0) == num_nodes, f"局部结构特征维度不匹配: {local_struct.size(0)} vs {num_nodes}"
    
    # 组合所有特征
    enhanced_features = torch.cat([
        features,  # 原始特征
        deg_features,  # 度特征
        local_struct  # 局部结构特征
    ], dim=1)
    
    return enhanced_features

def compute_local_structure(edge_index, num_nodes):
    """计算局部结构特征"""
    # 转换为NetworkX图
    edge_list = edge_index.t().tolist()
    G = nx.Graph(edge_list)
    
    # 确保图包含所有节点
    G.add_nodes_from(range(num_nodes))
    
    # 计算聚类系数
    clustering = torch.zeros(num_nodes)
    nx_clustering = nx.clustering(G)
    for node in range(num_nodes):
        clustering[node] = nx_clustering.get(node, 0.0)
    
    # 计算k-core分解
    kcore = torch.zeros(num_nodes)
    nx_kcore = nx.core_number(G)
    for node in range(num_nodes):
        kcore[node] = nx_kcore.get(node, 0.0)
    
    # 组合特征
    local_features = torch.stack([clustering, kcore], dim=1)
    assert local_features.size(0) == num_nodes, f"局部特征维度错误: {local_features.size(0)} vs {num_nodes}"
    
    return local_features

def compute_edge_features(edge_index, num_nodes):
    """计算边特征"""
    row, col = edge_index
    edge_attr = torch.zeros(edge_index.size(1), 2)
    
    # 计算源节点和目标节点的度
    deg_row = degree(row, num_nodes)
    deg_col = degree(col, num_nodes)
    
    # 边特征：源节点度和目标节点度
    edge_attr[:, 0] = deg_row[row]
    edge_attr[:, 1] = deg_col[col]
    
    return edge_attr 