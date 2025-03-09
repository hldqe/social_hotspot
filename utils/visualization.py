"""
可视化模块 - 精简版
负责可视化图结构和预测结果
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import torch
import json
import logging
from sklearn.manifold import TSNE
from datetime import datetime

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class GraphVisualizer:
    """图可视化类"""
    
    def __init__(self, data_dir=None):
        """初始化可视化器
        
        Args:
            data_dir: 数据目录
        """
        self.data_dir = data_dir
        logger.info("图可视化器初始化完成")
    
    def load_graph_data(self, graph_type='heterogeneous'):
        """加载图数据
        
        Args:
            graph_type: 图类型，'homogeneous'或'heterogeneous'
        
        Returns:
            tuple: (图数据, 节点映射)
        """
        # 加载图数据
        graph_path = os.path.join(self.data_dir, f'{graph_type}_graph.pt')
        if not os.path.exists(graph_path):
            logger.error(f"图数据不存在: {graph_path}")
            return None, None
        
        data = torch.load(graph_path)
        logger.info(f"加载图数据: {graph_path}")
        
        # 加载节点映射
        mappings_path = os.path.join(self.data_dir, 'node_mappings.json')
        with open(mappings_path, 'r', encoding='utf-8') as f:
            mappings = json.load(f)
            # 将字符串键转换回原始类型
            if 'news_id_to_idx' in mappings:
                mappings['news_id_to_idx'] = {eval(k): v for k, v in mappings['news_id_to_idx'].items()}
        
        return data, mappings
    
    def load_news_data(self):
        """加载新闻数据
        
        Returns:
            pandas.DataFrame: 新闻数据
        """
        news_path = os.path.join(self.data_dir, 'processed_news.csv')
        if not os.path.exists(news_path):
            logger.error(f"新闻数据不存在: {news_path}")
            return None
        
        news_df = pd.read_csv(news_path, encoding='utf-8')
        logger.info(f"加载新闻数据: {news_path}, 共{len(news_df)}条新闻")
        
        return news_df
    
    def convert_to_networkx(self, data, mappings, max_nodes=1000):
        """将PyG图转换为NetworkX图
        
        Args:
            data: PyG图数据
            mappings: 节点映射
            max_nodes: 最大节点数量，用于可视化
        
        Returns:
            networkx.Graph: NetworkX图
        """
        # 同构图
        if hasattr(data, 'x') and hasattr(data, 'edge_index'):
            G = nx.Graph()
            
            # 添加节点
            num_news = mappings['num_news']
            num_entities = mappings['num_entities']
            
            # 限制节点数量
            if num_news + num_entities > max_nodes:
                logger.warning(f"节点数量({num_news + num_entities})超过最大限制({max_nodes})，将随机采样节点")
                
                # 随机采样新闻节点
                news_sample_size = min(num_news, max_nodes // 2)
                news_indices = np.random.choice(num_news, size=news_sample_size, replace=False)
                
                # 随机采样实体节点
                entity_sample_size = min(num_entities, max_nodes - news_sample_size)
                entity_indices = np.random.choice(num_entities, size=entity_sample_size, replace=False)
                entity_indices = entity_indices + num_news  # 调整索引
                
                # 合并采样节点
                sampled_indices = np.concatenate([news_indices, entity_indices])
                
                # 创建索引映射
                idx_map = {old_idx: new_idx for new_idx, old_idx in enumerate(sampled_indices)}
                
                # 添加节点
                for i, idx in enumerate(sampled_indices):
                    if idx < num_news:
                        G.add_node(i, type='news')
                    else:
                        G.add_node(i, type='entity')
                
                # 添加边
                edge_index = data.edge_index.numpy()
                for i in range(edge_index.shape[1]):
                    src, dst = edge_index[0, i], edge_index[1, i]
                    if src in idx_map and dst in idx_map:
                        G.add_edge(idx_map[src], idx_map[dst])
            else:
                # 添加所有节点
                for i in range(num_news + num_entities):
                    if i < num_news:
                        G.add_node(i, type='news')
                    else:
                        G.add_node(i, type='entity')
                
                # 添加所有边
                edge_index = data.edge_index.numpy()
                for i in range(edge_index.shape[1]):
                    src, dst = edge_index[0, i], edge_index[1, i]
                    G.add_edge(src, dst)
        
        # 异构图
        else:
            G = nx.Graph()
            
            # 添加新闻节点
            news_x = data['news'].x
            for i in range(len(news_x)):
                G.add_node(i, type='news')
            
            # 添加实体节点
            entity_x = data['entity'].x
            for i in range(len(entity_x)):
                G.add_node(i + len(news_x), type='entity')
            
            # 添加新闻->实体边
            edge_index = data['news', 'contains', 'entity'].edge_index.numpy()
            for i in range(edge_index.shape[1]):
                src, dst = edge_index[0, i], edge_index[1, i] + len(news_x)
                G.add_edge(src, dst)
        
        logger.info(f"转换为NetworkX图，包含{G.number_of_nodes()}个节点和{G.number_of_edges()}条边")
        
        return G
    
    def visualize_graph(self, G, title="新闻-实体图", save_path=None, figsize=(12, 10)):
        """可视化图结构
        
        Args:
            G: NetworkX图
            title: 图标题
            save_path: 保存路径
            figsize: 图大小
        """
        plt.figure(figsize=figsize)
        
        # 获取节点类型
        node_types = nx.get_node_attributes(G, 'type')
        
        # 设置节点颜色
        node_colors = []
        for node in G.nodes():
            if node_types.get(node) == 'news':
                node_colors.append('skyblue')
            else:
                node_colors.append('lightgreen')
        
        # 设置节点大小
        node_sizes = []
        for node in G.nodes():
            if node_types.get(node) == 'news':
                node_sizes.append(100)
            else:
                node_sizes.append(50)
        
        # 使用spring布局
        pos = nx.spring_layout(G, seed=42)
        
        # 绘制节点
        nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes, alpha=0.8)
        
        # 绘制边
        nx.draw_networkx_edges(G, pos, width=0.5, alpha=0.5)
        
        # 添加图例
        news_patch = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='skyblue', markersize=10, label='新闻')
        entity_patch = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='lightgreen', markersize=10, label='实体')
        plt.legend(handles=[news_patch, entity_patch], loc='upper right')
        
        plt.title(title)
        plt.axis('off')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"图可视化已保存到: {save_path}")
        
        plt.show()
    
    def visualize_embeddings(self, data, mappings, model=None, title="节点嵌入可视化", save_path=None, figsize=(12, 10)):
        """可视化节点嵌入
        
        Args:
            data: PyG图数据
            mappings: 节点映射
            model: GNN模型，如果提供则使用模型生成嵌入
            title: 图标题
            save_path: 保存路径
            figsize: 图大小
        """
        # 获取节点嵌入
        if model is not None:
            model.eval()
            with torch.no_grad():
                # 同构图
                if hasattr(data, 'x') and hasattr(data, 'edge_index'):
                    x = data.x
                    edge_index = data.edge_index
                    embeddings = model.gnn(x, edge_index).numpy()
                # 异构图
                else:
                    x_dict = {k: v for k, v in data.x_dict.items()}
                    edge_index_dict = {k: v for k, v in data.edge_index_dict.items()}
                    # 这里假设模型有一个获取嵌入的方法
                    embeddings = model.get_embeddings(x_dict, edge_index_dict).numpy()
        else:
            # 使用原始特征
            if hasattr(data, 'x'):
                embeddings = data.x.numpy()
            else:
                # 合并不同类型的节点特征
                embeddings = np.vstack([data['news'].x.numpy(), data['entity'].x.numpy()])
        
        # 使用t-SNE降维
        tsne = TSNE(n_components=2, random_state=42)
        embeddings_2d = tsne.fit_transform(embeddings)
        
        # 分离新闻和实体节点
        num_news = mappings['num_news']
        news_embeddings = embeddings_2d[:num_news]
        entity_embeddings = embeddings_2d[num_news:]
        
        # 可视化
        plt.figure(figsize=figsize)
        
        # 绘制新闻节点
        plt.scatter(news_embeddings[:, 0], news_embeddings[:, 1], c='skyblue', label='新闻', alpha=0.7, s=50)
        
        # 绘制实体节点
        plt.scatter(entity_embeddings[:, 0], entity_embeddings[:, 1], c='lightgreen', label='实体', alpha=0.7, s=30)
        
        plt.title(title)
        plt.legend()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"嵌入可视化已保存到: {save_path}")
        
        plt.show()
    
    def visualize_hotspots(self, hotspots, title="热点新闻预测", save_path=None, figsize=(12, 8)):
        """可视化热点新闻预测结果
        
        Args:
            hotspots: 热点新闻列表
            title: 图标题
            save_path: 保存路径
            figsize: 图大小
        """
        if not hotspots:
            logger.warning("没有热点新闻数据可视化")
            return
        
        # 提取数据
        titles = [news['title'][:20] + '...' if len(news['title']) > 20 else news['title'] for news in hotspots]
        scores = [news['score'] for news in hotspots]
        
        # 按分数排序
        sorted_indices = np.argsort(scores)
        titles = [titles[i] for i in sorted_indices]
        scores = [scores[i] for i in sorted_indices]
        
        # 可视化
        plt.figure(figsize=figsize)
        
        # 绘制水平条形图
        bars = plt.barh(titles, scores, color='skyblue')
        
        # 添加数据标签
        for bar in bars:
            width = bar.get_width()
            plt.text(width + 0.01, bar.get_y() + bar.get_height()/2, f'{width:.4f}', 
                    ha='left', va='center')
        
        plt.xlabel('热点分数')
        plt.ylabel('新闻标题')
        plt.title(title)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"热点可视化已保存到: {save_path}")
        
        plt.show()
    
    def visualize_entity_importance(self, data, mappings, model=None, top_k=20, title="实体重要性", save_path=None, figsize=(12, 8)):
        """可视化实体重要性
        
        Args:
            data: PyG图数据
            mappings: 节点映射
            model: GNN模型
            top_k: 显示前K个重要实体
            title: 图标题
            save_path: 保存路径
            figsize: 图大小
        """
        # 加载实体数据
        entities_path = os.path.join(self.data_dir, 'entities.csv')
        if not os.path.exists(entities_path):
            logger.error(f"实体数据不存在: {entities_path}")
            return
        
        entities_df = pd.read_csv(entities_path, encoding='utf-8')
        
        # 计算实体重要性
        entity_to_idx = mappings['entity_to_idx']
        entity_importance = {}
        
        # 使用实体出现频率作为重要性指标
        for entity, count in zip(entities_df['entity'], entities_df['count']):
            if entity in entity_to_idx:
                entity_importance[entity] = count
        
        # 选择前K个重要实体
        top_entities = sorted(entity_importance.items(), key=lambda x: x[1], reverse=True)[:top_k]
        
        # 提取数据
        entity_names = [entity for entity, _ in top_entities]
        importance_scores = [score for _, score in top_entities]
        
        # 可视化
        plt.figure(figsize=figsize)
        
        # 绘制水平条形图
        bars = plt.barh(entity_names, importance_scores, color='lightgreen')
        
        # 添加数据标签
        for bar in bars:
            width = bar.get_width()
            plt.text(width + 0.5, bar.get_y() + bar.get_height()/2, f'{int(width)}', 
                    ha='left', va='center')
        
        plt.xlabel('出现频率')
        plt.ylabel('实体名称')
        plt.title(title)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"实体重要性可视化已保存到: {save_path}")
        
        plt.show()
    
    def run(self, graph_type='heterogeneous', visualize_type='graph'):
        """运行可视化
        
        Args:
            graph_type: 图类型，'homogeneous'或'heterogeneous'
            visualize_type: 可视化类型，'graph'、'embeddings'、'hotspots'或'entity_importance'
        """
        # 加载图数据
        data, mappings = self.load_graph_data(graph_type)
        
        if data is None or mappings is None:
            logger.error("加载图数据失败")
            return
        
        # 创建输出目录
        output_dir = os.path.join(self.data_dir, 'visualizations')
        os.makedirs(output_dir, exist_ok=True)
        
        # 时间戳
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        if visualize_type == 'graph':
            # 转换为NetworkX图
            G = self.convert_to_networkx(data, mappings)
            
            # 可视化图结构
            save_path = os.path.join(output_dir, f'graph_{graph_type}_{timestamp}.png')
            self.visualize_graph(G, title=f"新闻-实体图 ({graph_type})", save_path=save_path)
        
        elif visualize_type == 'embeddings':
            # 可视化节点嵌入
            save_path = os.path.join(output_dir, f'embeddings_{graph_type}_{timestamp}.png')
            self.visualize_embeddings(data, mappings, title=f"节点嵌入可视化 ({graph_type})", save_path=save_path)
        
        elif visualize_type == 'hotspots':
            # 加载热点数据
            hotspots_files = [f for f in os.listdir(self.data_dir) if f.startswith('hotspots_') and f.endswith('.json')]
            
            if not hotspots_files:
                logger.error("没有找到热点数据文件")
                return
            
            # 使用最新的热点数据
            latest_hotspots_file = sorted(hotspots_files)[-1]
            hotspots_path = os.path.join(self.data_dir, latest_hotspots_file)
            
            with open(hotspots_path, 'r', encoding='utf-8') as f:
                hotspots = json.load(f)
            
            # 可视化热点
            save_path = os.path.join(output_dir, f'hotspots_{timestamp}.png')
            self.visualize_hotspots(hotspots, title="热点新闻预测", save_path=save_path)
        
        elif visualize_type == 'entity_importance':
            # 可视化实体重要性
            save_path = os.path.join(output_dir, f'entity_importance_{timestamp}.png')
            self.visualize_entity_importance(data, mappings, title="实体重要性", save_path=save_path)
        
        else:
            logger.error(f"不支持的可视化类型: {visualize_type}")

def main():
    """主函数"""
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from config import DATA_DIR
    
    visualizer = GraphVisualizer(DATA_DIR)
    
    # 可视化图结构
    visualizer.run(graph_type='heterogeneous', visualize_type='graph')
    
    # 可视化节点嵌入
    visualizer.run(graph_type='heterogeneous', visualize_type='embeddings')
    
    # 可视化实体重要性
    visualizer.run(graph_type='heterogeneous', visualize_type='entity_importance')

if __name__ == "__main__":
    main() 