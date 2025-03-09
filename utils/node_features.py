def enhance_node_features(features, edge_index, num_hops=2):
    """增强节点特征，融合多跳邻居信息"""
    enhanced_features = features.clone()
    
    # 计算邻接表
    adj_list = [[] for _ in range(features.size(0))]
    for i in range(edge_index.size(1)):
        src, dst = edge_index[0, i].item(), edge_index[1, i].item()
        adj_list[dst].append(src)
    
    # 对每个节点，汇总多跳邻居特征
    for node in range(features.size(0)):
        # 1跳邻居
        neighbors = adj_list[node]
        if not neighbors:
            continue
            
        # 计算邻居特征平均值
        neighbor_feats = features[neighbors].mean(dim=0)
        
        # 与原始特征融合
        enhanced_features[node] = features[node] * 0.7 + neighbor_feats * 0.3
    
    return enhanced_features 