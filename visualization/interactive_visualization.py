import os
import sys

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

# 打印Python搜索路径进行调试
print("Python搜索路径:")
for path in sys.path:
    print(f"- {path}")

# 检查models目录是否存在
models_dir = os.path.join(project_root, "models")
if os.path.exists(models_dir):
    print(f"models目录存在: {models_dir}")
    print(f"目录内容: {os.listdir(models_dir)}")
else:
    print(f"models目录不存在!")

import torch
import numpy as np
import networkx as nx
import plotly.graph_objects as go
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def load_graph_data(data_path):
    """加载图数据"""
    print(f"从{data_path}加载图数据...")
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
    
    print(f"图构建完成: {G.number_of_nodes()}个节点, {G.number_of_edges()}条边")
    return G, features, edge_index, labels

def add_predictions_to_graph(G, predictions):
    """将预测结果添加到图中"""
    for i, prob in enumerate(predictions):
        G.nodes[i]['prediction'] = float(prob)
        G.nodes[i]['predicted_label'] = 1 if prob > 0.5 else 0
    return G

def create_interactive_visualization(G, output_path='interactive_network.html'):
    """创建交互式网络可视化"""
    
    # 计算节点位置 (使用NetworkX的布局算法)
    pos = nx.spring_layout(G, seed=42)
    
    # 准备节点轨迹数据
    node_x = []
    node_y = []
    node_text = []
    node_colors = []
    node_sizes = []
    
    # 获取节点度
    degrees = dict(G.degree())
    
    for node, position in pos.items():
        node_x.append(position[0])
        node_y.append(position[1])
        
        # 节点文本标签 (悬停显示)
        true_label = G.nodes[node]['label']
        if 'prediction' in G.nodes[node]:
            pred_prob = G.nodes[node]['prediction']
            text = f"节点 {node}<br>度: {degrees[node]}<br>预测概率: {pred_prob:.4f}<br>真实标签: {'热点' if true_label==1 else '非热点'}"
        else:
            text = f"节点 {node}<br>度: {degrees[node]}<br>真实标签: {'热点' if true_label==1 else '非热点'}"
        node_text.append(text)
        
        # 颜色编码
        if 'predicted_label' in G.nodes[node]:
            pred_label = G.nodes[node]['predicted_label']
            if true_label == 1 and pred_label == 1:
                color = 'green'  # 真正例
            elif true_label == 0 and pred_label == 0:
                color = 'blue'   # 真负例
            elif true_label == 0 and pred_label == 1:
                color = 'red'    # 假正例
            else:  # true_label == 1 and pred_label == 0
                color = 'orange' # 假负例
        else:
            color = 'red' if true_label == 1 else 'gray'
        node_colors.append(color)
        
        # 节点大小
        size = 10 + 20 * np.sqrt(degrees[node])
        if true_label == 1:
            size *= 1.5  # 增大热点节点
        node_sizes.append(size)
    
    # 创建节点轨迹
    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers',
        hoverinfo='text',
        text=node_text,
        marker=dict(
            color=node_colors,
            size=node_sizes,
            line=dict(width=1, color='black')
        )
    )
    
    # 改变边的处理方式 - 分组创建
    # 准备热点间连接和普通连接的数据
    edge_hotspot_x = []
    edge_hotspot_y = []
    edge_normal_x = []
    edge_normal_y = []
    
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        
        # 根据边的类型分组
        if G.nodes[edge[0]]['label'] == 1 and G.nodes[edge[1]]['label'] == 1:
            # 热点节点之间的连接
            edge_hotspot_x.extend([x0, x1, None])
            edge_hotspot_y.extend([y0, y1, None])
        else:
            # 普通连接
            edge_normal_x.extend([x0, x1, None])
            edge_normal_y.extend([y0, y1, None])
    
    # 创建热点间连接轨迹
    edge_hotspot_trace = go.Scatter(
        x=edge_hotspot_x, y=edge_hotspot_y,
        mode='lines',
        line=dict(width=1.5, color='rgba(255,0,0,0.3)'),  # 红色半透明
        hoverinfo='none'
    )
    
    # 创建普通连接轨迹
    edge_normal_trace = go.Scatter(
        x=edge_normal_x, y=edge_normal_y,
        mode='lines',
        line=dict(width=1, color='rgba(150,150,150,0.2)'),  # 灰色半透明
        hoverinfo='none'
    )
    
    # 更新图形创建代码，包含两种边轨迹
    fig = go.Figure(data=[edge_normal_trace, edge_hotspot_trace, node_trace],
                   layout=go.Layout(
                       title=dict(text='交互式社交网络热点节点可视化', font=dict(size=16)),
                       showlegend=False,
                       hovermode='closest',
                       margin=dict(b=20,l=5,r=5,t=40),
                       xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                       yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                       width=1000,
                       height=800
                   ))
    
    # 添加图例注释
    legend_items = [
        dict(text="正确预测的热点", x=0.01, y=0.99, showarrow=False, font=dict(color='green')),
        dict(text="误判为热点", x=0.01, y=0.95, showarrow=False, font=dict(color='red')),
        dict(text="正确预测的非热点", x=0.01, y=0.91, showarrow=False, font=dict(color='blue')),
        dict(text="漏检的热点", x=0.01, y=0.87, showarrow=False, font=dict(color='orange'))
    ]
    
    for item in legend_items:
        fig.add_annotation(item)
    
    # 保存为HTML文件
    fig.write_html(output_path)
    print(f"交互式可视化已保存至: {output_path}")

def create_heatmap_analysis(G, output_path='heatmap_analysis.png'):
    """创建热力图分析"""
    # 提取度和中介中心性
    degrees = []
    between_cent = []
    labels = []
    predictions = []
    
    # 计算中介中心性
    betweenness = nx.betweenness_centrality(G)
    
    for node in G.nodes():
        degrees.append(G.degree(node))
        between_cent.append(betweenness[node])
        labels.append(G.nodes[node]['label'])
        if 'prediction' in G.nodes[node]:
            predictions.append(G.nodes[node]['prediction'])
        else:
            predictions.append(0)
    
    # 创建数据框
    df = pd.DataFrame({
        '节点度': degrees,
        '中介中心性': between_cent,
        '真实标签': labels,
        '预测概率': predictions
    })
    
    # 创建热力图
    plt.figure(figsize=(12, 10))
    
    # 热点和非热点节点分离
    df_hotspot = df[df['真实标签'] == 1]
    df_non_hotspot = df[df['真实标签'] == 0]
    
    plt.scatter(
        df_non_hotspot['节点度'], 
        df_non_hotspot['中介中心性'], 
        c=df_non_hotspot['预测概率'], 
        cmap='Blues', 
        alpha=0.8, 
        s=100,
        label='非热点'
    )
    
    plt.scatter(
        df_hotspot['节点度'], 
        df_hotspot['中介中心性'], 
        c=df_hotspot['预测概率'], 
        cmap='Reds', 
        alpha=0.8, 
        s=100,
        marker='^',
        label='热点'
    )
    
    plt.colorbar(label='预测概率')
    plt.xlabel('节点度', fontsize=14)
    plt.ylabel('中介中心性', fontsize=14)
    plt.title('热点节点预测热力图分析', fontsize=16)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

def main():
    # 定义路径 - 使用正确的绝对路径
    data_path = 'D:/Code/毕设/social_hotspot/data/processed/mind_full_graph_data.pt'
    output_dir = 'results/visualizations'
    os.makedirs(output_dir, exist_ok=True)
    
    # 加载数据
    G, features, edge_index, labels = load_graph_data(data_path)
    
    # 如果有模型预测结果，加载并添加到图中
    try:
        # 使用正确的模型路径
        from models.advanced_gnn import AdvancedGNN
        model_path = 'D:/Code/毕设/social_hotspot/saved_models/best_model.pth'  # 修正模型路径
        
        # 使用与训练时相同的输入维度
        model = AdvancedGNN(input_dim=133, hidden_dim=256)  # 固定输入维度为133
        model.load_state_dict(torch.load(model_path))
        model.eval()
        
        # 如果当前特征维度与模型期望维度不匹配，需要调整特征维度
        if features.size(1) != 133:
            print(f"特征维度不匹配！模型期望: 133, 当前特征: {features.size(1)}")
            # 方法1: 填充特征以匹配维度
            if features.size(1) < 133:
                padding = torch.zeros((features.size(0), 133 - features.size(1)))
                features = torch.cat([features, padding], dim=1)
                print(f"已将特征填充至: {features.size()}")
            # 方法2: 截断特征以匹配维度
            else:
                features = features[:, :133]
                print(f"已将特征截断至: {features.size()}")
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        features = features.to(device)
        edge_index = edge_index.to(device)
        
        with torch.no_grad():
            logits = model(features, edge_index)
            probabilities = torch.sigmoid(logits.squeeze(-1)).cpu().numpy()  # 确保返回CPU上的numpy数组
        
        # 添加预测结果到图中
        G = add_predictions_to_graph(G, probabilities)
        print("已将模型预测结果添加到图中")
    except Exception as e:
        print(f"无法加载模型或进行预测: {e}")
        print(f"错误详情: {str(e)}")
        print("将只使用真实标签进行可视化")
    
    # 创建交互式可视化
    interactive_viz_path = os.path.join(output_dir, 'interactive_network.html')
    create_interactive_visualization(G, output_path=interactive_viz_path)
    
    # 创建热力图分析
    heatmap_path = os.path.join(output_dir, 'hotspot_heatmap.png')
    create_heatmap_analysis(G, output_path=heatmap_path)
    
    print("交互式可视化完成！")

if __name__ == "__main__":
    main()
