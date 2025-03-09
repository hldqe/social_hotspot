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
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patches as mpatches
from sklearn.metrics import confusion_matrix

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

def create_network_visualization(G, output_path='network_visualization.png', 
                                 layout='spring', figsize=(16, 12), dpi=300):
    """创建社交网络可视化图"""
    plt.figure(figsize=figsize)
    
    # 选择布局算法
    if layout == 'spring':
        pos = nx.spring_layout(G, k=0.3, iterations=50, seed=42)
    elif layout == 'kamada_kawai':
        pos = nx.kamada_kawai_layout(G)
    elif layout == 'force_atlas2':
        # 需要安装 forceatlas2 库
        try:
            import forceatlas2
            pos = forceatlas2.forceatlas2_layout(G, iterations=100, kr=0.01)
        except ImportError:
            print("forceatlas2未安装，使用spring布局代替")
            pos = nx.spring_layout(G, k=0.3, iterations=50, seed=42)
    else:
        pos = nx.spring_layout(G, seed=42)
    
    # 准备节点颜色和大小
    node_colors = []
    node_sizes = []
    edge_colors = []
    
    # 计算度中心性 - 用于节点大小
    degrees = dict(G.degree())
    max_degree = max(degrees.values()) if degrees else 1
    
    # 节点信息
    for node in G.nodes():
        true_label = G.nodes[node]['label']
        if 'predicted_label' in G.nodes[node]:
            pred_label = G.nodes[node]['predicted_label']
            
            # 颜色编码: 4种情况
            if true_label == 1 and pred_label == 1:
                color = 'green'  # 真正例 - 正确预测的热点
            elif true_label == 0 and pred_label == 0:
                color = 'blue'   # 真负例 - 正确预测的非热点
            elif true_label == 0 and pred_label == 1:
                color = 'red'    # 假正例 - 错误预测为热点
            else:  # true_label == 1 and pred_label == 0
                color = 'orange' # 假负例 - 漏检的热点
        else:
            # 如果没有预测标签，只使用真实标签
            color = 'red' if true_label == 1 else 'gray'
        
        # 节点大小 - 基于度中心性
        size = 100 + 500 * (degrees[node] / max_degree)
        # 热点节点略微放大
        if true_label == 1:
            size *= 1.5
            
        node_colors.append(color)
        node_sizes.append(size)
    
    # 边的颜色 - 给热点节点之间的连接不同颜色
    for u, v in G.edges():
        if G.nodes[u]['label'] == 1 and G.nodes[v]['label'] == 1:
            edge_colors.append('red')  # 热点节点之间的连接
        else:
            edge_colors.append('gray')  # 普通连接
    
    # 绘制边
    nx.draw_networkx_edges(
        G, pos, 
        width=0.6, 
        alpha=0.3, 
        edge_color=edge_colors
    )
    
    # 绘制节点
    nodes = nx.draw_networkx_nodes(
        G, pos, 
        node_color=node_colors, 
        node_size=node_sizes, 
        alpha=0.8
    )
    
    # 添加热点节点标签
    hotspot_labels = {}
    for node in G.nodes():
        if G.nodes[node]['label'] == 1:
            # 只标记热点节点
            hotspot_labels[node] = str(node)
    
    nx.draw_networkx_labels(
        G, pos, 
        labels=hotspot_labels, 
        font_size=10, 
        font_weight='bold'
    )
    
    # 添加图例
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=10, label='正确预测的热点'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, label='误判为热点'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=10, label='正确预测的非热点'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='orange', markersize=10, label='漏检的热点')
    ]
    plt.legend(handles=legend_elements, loc='lower right')
    
    # 添加混淆矩阵作为子图
    if all('predicted_label' in G.nodes[n] for n in G.nodes()):
        # 提取标签和预测
        y_true = [G.nodes[n]['label'] for n in G.nodes()]
        y_pred = [G.nodes[n]['predicted_label'] for n in G.nodes()]
        
        # 创建混淆矩阵子图
        ax1 = plt.axes([0.01, 0.01, 0.2, 0.2])  # 左下角
        cm = confusion_matrix(y_true, y_pred)
        ax1.matshow(cm, cmap='Blues')
        ax1.set_title('混淆矩阵', fontsize=10)
        ax1.set_xlabel('预测', fontsize=8)
        ax1.set_ylabel('实际', fontsize=8)
        ax1.tick_params(axis='both', which='major', labelsize=8)
        
        # 在混淆矩阵中添加数值标签
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax1.text(j, i, str(cm[i, j]), ha='center', va='center', fontsize=8)
    
    # 设置标题和细节
    plt.title('社交网络热点节点预测可视化', fontsize=16)
    plt.axis('off')
    
    # 保存图像
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    print(f"可视化图已保存至: {output_path}")
    plt.close()

def visualize_with_communities(G, output_path='community_visualization.png'):
    """使用社区检测进行可视化"""
    # 使用Louvain算法检测社区
    try:
        from community import community_louvain
        partition = community_louvain.best_partition(G)
        
        plt.figure(figsize=(16, 12))
        pos = nx.spring_layout(G, seed=42)
        
        # 为不同社区着色
        cmap = plt.cm.get_cmap('tab20', max(partition.values()) + 1)
        
        # 绘制节点和边
        nx.draw_networkx(
            G, pos,
            node_color=[partition[n] for n in G.nodes()],
            cmap=cmap,
            node_size=[100 + 500*(1 if G.nodes[n]['label']==1 else 0.5) for n in G.nodes()],
            edge_color='gray',
            alpha=0.7,
            with_labels=False
        )
        
        # 标记热点节点
        hotspot_nodes = [n for n in G.nodes() if G.nodes[n]['label']==1]
        nx.draw_networkx_nodes(
            G, pos, 
            nodelist=hotspot_nodes,
            node_color='red',
            node_size=[150 + 500*(1 if G.nodes[n]['label']==1 else 0.5) for n in hotspot_nodes],
            alpha=0.9,
            linewidths=2,
            edgecolors='black'
        )
        
        # 添加图例
        plt.title('社交网络社区结构与热点节点', fontsize=16)
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"社区可视化已保存至: {output_path}")
    except ImportError:
        print("需要安装python-louvain库来进行社区检测")
        print("pip install python-louvain")

def create_prediction_analysis(G, output_path='prediction_analysis.png'):
    """创建预测分析图 - 展示度中心性与热点预测的关系"""
    if not all('predicted_label' in G.nodes[n] for n in G.nodes()):
        print("图中缺少预测标签，无法创建预测分析图")
        return
    
    plt.figure(figsize=(10, 8))
    
    # 提取度和预测概率
    degrees = np.array([G.degree(n) for n in G.nodes()])
    probs = np.array([G.nodes[n].get('prediction', 0) for n in G.nodes()])
    labels = np.array([G.nodes[n]['label'] for n in G.nodes()])
    
    # 绘制散点图
    plt.scatter(
        degrees[labels==0], probs[labels==0], 
        color='blue', label='非热点', alpha=0.7
    )
    plt.scatter(
        degrees[labels==1], probs[labels==1], 
        color='red', label='热点', alpha=0.7
    )
    
    # 添加决策边界线
    plt.axhline(y=0.5, color='gray', linestyle='--', alpha=0.7)
    
    # 计算趋势线
    from scipy.stats import linregress
    slope, intercept, r_value, p_value, std_err = linregress(degrees, probs)
    x_line = np.linspace(min(degrees), max(degrees), 100)
    plt.plot(x_line, slope*x_line + intercept, 'k--', alpha=0.5)
    
    plt.xlabel('节点度', fontsize=12)
    plt.ylabel('热点预测概率', fontsize=12)
    plt.title('节点度与热点预测概率关系', fontsize=14)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"预测分析图已保存至: {output_path}")

def main():
    # 定义路径
    data_path = 'D:/Code/毕设/social_hotspot/data/processed/mind_full_graph_data.pt'
    model_path = 'D:/Code/毕设/social_hotspot/saved_models/best_model.pth'  # 修改为新的模型文件名
    output_dir = 'results/visualizations'
    os.makedirs(output_dir, exist_ok=True)
    
    # 加载数据
    G, features, edge_index, labels = load_graph_data(data_path)
    
    # 如果有模型预测结果，加载并添加到图中
    try:
        # 加载模型 - 使用与训练时相同的输入维度(133)而不是当前特征的维度
        from models.advanced_gnn import AdvancedGNN
        model = AdvancedGNN(input_dim=133, hidden_dim=256)  # 修改这里，使用与保存模型匹配的维度
        model.load_state_dict(torch.load(model_path))
        model.eval()
        
        # 如果当前特征维度与模型期望维度不匹配，需要调整特征维度
        if features.size(1) != 133:
            print(f"特征维度不匹配！模型期望: 133, 当前特征: {features.size(1)}")
            # 方法1: 填充特征以匹配维度
            if features.size(1) < 133:
                padding = torch.zeros((features.size(0), 133 - features.size(1)), device=features.device)
                features = torch.cat([features, padding], dim=1)
                print(f"已将特征填充至: {features.size()}")
            # 方法2: 截断特征以匹配维度
            else:
                features = features[:, :133]
                print(f"已将特征截断至: {features.size()}")
        
        # 添加设备处理
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        features = features.to(device)
        edge_index = edge_index.to(device)
        
        with torch.no_grad():
            logits = model(features, edge_index)
            probabilities = torch.sigmoid(logits.squeeze(-1)).cpu().numpy()
        
        # 添加预测结果到图中
        G = add_predictions_to_graph(G, probabilities)
        print("已将模型预测结果添加到图中")
    except Exception as e:
        print(f"无法加载模型或进行预测: {e}")
        print("将只使用真实标签进行可视化")
    
    # 创建基础网络可视化
    network_viz_path = os.path.join(output_dir, 'network_visualization.png')
    create_network_visualization(G, output_path=network_viz_path)
    
    # 创建社区可视化
    community_viz_path = os.path.join(output_dir, 'community_visualization.png')
    visualize_with_communities(G, output_path=community_viz_path)
    
    # 创建预测分析图
    if all('prediction' in G.nodes[n] for n in G.nodes()):
        prediction_analysis_path = os.path.join(output_dir, 'prediction_analysis.png')
        create_prediction_analysis(G, output_path=prediction_analysis_path)

if __name__ == "__main__":
    main()
