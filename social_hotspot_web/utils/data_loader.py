import torch
import networkx as nx
import json
import os
import sys

# 确保能导入模型模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def load_graph_data(data_path, model_path=None, use_real_predictions=True):
    """加载图数据并转换为前端可用的JSON格式，可选择使用真实模型预测"""
    print(f"从{data_path}加载图数据...")
    
    # 加载PyTorch保存的数据
    data = torch.load(data_path)
    
    features = data['node_features']
    edge_index = data['edge_index']
    labels = data['node_labels']
    
    # 创建NetworkX图
    G = nx.Graph()
    
    # 添加节点
    for i in range(features.size(0)):
        G.add_node(i, label=labels[i].item())
    
    # 添加边
    for i in range(edge_index.size(1)):
        src, dst = edge_index[0, i].item(), edge_index[1, i].item()
        G.add_edge(src, dst)
    
    # 计算中介中心性
    try:
        betweenness = nx.betweenness_centrality(G)
    except:
        betweenness = {node: 0.0 for node in G.nodes()}
    
    # 计算度中心性
    degrees = dict(G.degree())
    
    # 如果需要使用真实模型预测
    predictions = []
    if use_real_predictions and model_path and os.path.exists(model_path):
        try:
            # 导入模型定义
            from models.advanced_gnn import AdvancedGNN
            
            # 确保特征维度匹配
            input_dim = 133  # 确认您的模型期望的输入维度
            
            # 如果需要调整特征维度
            if features.size(1) != input_dim:
                if features.size(1) < input_dim:
                    # 填充特征
                    padding = torch.zeros((features.size(0), input_dim - features.size(1)))
                    features = torch.cat([features, padding], dim=1)
                else:
                    # 截断特征
                    features = features[:, :input_dim]
            
            # 加载模型
            model = AdvancedGNN(input_dim=input_dim, hidden_dim=256)
            model.load_state_dict(torch.load(model_path))
            model.eval()
            
            # 切换到CPU上运行预测
            device = torch.device('cpu')
            model = model.to(device)
            features = features.to(device)
            edge_index = edge_index.to(device)
            
            # 进行预测
            with torch.no_grad():
                logits = model(features, edge_index)
                probabilities = torch.sigmoid(logits.squeeze(-1)).cpu().numpy()
                
            print(f"成功使用模型进行预测，获得{len(probabilities)}个节点的预测值")
            predictions = probabilities.tolist()
        except Exception as e:
            print(f"模型预测失败: {str(e)}")
            print("将使用标签作为替代")
            predictions = [float(label) for label in labels.numpy()]
    else:
        # 使用标签作为替代
        predictions = [float(label) for label in labels.numpy()]
    
    # 构建JSON格式数据
    nodes = []
    for i, node_id in enumerate(G.nodes()):
        node_data = {
            'id': node_id,
            'label': G.nodes[node_id]['label'],
            'degree': degrees[node_id],
            'betweenness': betweenness[node_id],
            'prediction': predictions[i] if i < len(predictions) else float(G.nodes[node_id]['label'])
        }
        nodes.append(node_data)
    
    edges = []
    for src, dst in G.edges():
        edges.append({
            'source': src,
            'target': dst
        })
    
    # 计算热点数量（基于预测值）
    hotspot_count = sum(1 for n in nodes if n['prediction'] > 0.5)
    
    return {
        'nodes': nodes,
        'edges': edges,
        'stats': {
            'nodeCount': len(nodes),
            'edgeCount': len(edges),
            'hotspotCount': hotspot_count
        }
    }

def save_network_data(data, output_path):
    """保存网络数据为JSON格式"""
    with open(output_path, 'w') as f:
        json.dump(data, f)
    print(f"网络数据已保存到: {output_path}")

# 预处理数据并保存
def preprocess_data():
    """预处理数据并保存"""
    data_path = 'D:/Code/毕设/social_hotspot/data/processed/mind_full_graph_data.pt'
    model_path = 'D:/Code/毕设/social_hotspot/saved_models/best_model.pth'
    output_path = 'static/data/network_data.json'
    
    if not os.path.exists('static/data'):
        os.makedirs('static/data')
    
    # 使用真实模型预测
    network_data = load_graph_data(data_path, model_path, use_real_predictions=True)
    
    # 保存处理后的数据
    with open(output_path, 'w') as f:
        json.dump(network_data, f)
    print(f"网络数据已保存到: {output_path}")
    
    return network_data
