"""
预测脚本 - 精简版
用于使用训练好的GNN模型预测热点新闻
"""

import os
import torch
import pandas as pd
import numpy as np
import json
import logging
from datetime import datetime
from tabulate import tabulate

# 导入自定义模块
from utils.graph_builder import GraphBuilder
from models.gnn_model import create_gnn_model
from models.evaluation import ModelEvaluator
from config import (
    DATA_DIR, MODEL_DIR, EMBEDDING_DIM, HIDDEN_DIM, 
    NUM_GNN_LAYERS, TOP_K_PREDICTIONS
)

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_model_and_data(graph_type='heterogeneous'):
    """加载模型和数据
    
    Args:
        graph_type: 图类型，'homogeneous'或'heterogeneous'
    
    Returns:
        tuple: (模型, 图数据, 节点映射)
    """
    # 加载图数据
    graph_path = os.path.join(DATA_DIR, f'{graph_type}_graph.pt')
    if not os.path.exists(graph_path):
        logger.error(f"图数据不存在: {graph_path}")
        return None, None, None
    
    data = torch.load(graph_path)
    logger.info(f"加载图数据: {graph_path}")
    
    # 加载节点映射
    mappings_path = os.path.join(DATA_DIR, 'node_mappings.json')
    with open(mappings_path, 'r', encoding='utf-8') as f:
        mappings = json.load(f)
        # 将字符串键转换回原始类型
        if 'news_id_to_idx' in mappings:
            mappings['news_id_to_idx'] = {eval(k): v for k, v in mappings['news_id_to_idx'].items()}
    
    # 创建模型
    if graph_type == 'homogeneous':
        in_channels = data.x.shape[1]
        model = create_gnn_model(graph_type, in_channels)
    else:
        in_channels = {node_type: data[node_type].x.shape[1] for node_type in data.node_types}
        metadata = data.metadata()
        model = create_gnn_model(graph_type, in_channels, metadata)
    
    # 加载模型权重
    model_path = os.path.join(MODEL_DIR, f'best_model_{graph_type}.pt')
    if not os.path.exists(model_path):
        logger.error(f"模型文件不存在: {model_path}")
        return None, None, None
    
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    logger.info(f"加载模型: {model_path}")
    
    return model, data, mappings

def load_news_data():
    """加载新闻数据
    
    Returns:
        pandas.DataFrame: 新闻数据
    """
    news_path = os.path.join(DATA_DIR, 'processed_news.csv')
    if not os.path.exists(news_path):
        logger.error(f"新闻数据不存在: {news_path}")
        return None
    
    news_df = pd.read_csv(news_path, encoding='utf-8')
    logger.info(f"加载新闻数据: {news_path}, 共{len(news_df)}条新闻")
    
    return news_df

def predict_hotspots(model, data, news_df, mappings, top_k=10):
    """预测热点新闻
    
    Args:
        model: GNN模型
        data: 图数据
        news_df: 新闻数据
        mappings: 节点映射
        top_k: 返回前K个热点
    
    Returns:
        list: 热点新闻列表
    """
    # 创建评估器
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    evaluator = ModelEvaluator(model, device)
    
    # 预测热点
    hotspots = evaluator.predict_hotspots(
        data, news_df, mappings['news_id_to_idx'], top_k=top_k
    )
    
    return hotspots

def display_hotspots(hotspots):
    """在终端显示热点新闻
    
    Args:
        hotspots: 热点新闻列表
    """
    if not hotspots:
        print("没有找到热点新闻")
        return
    
    # 准备表格数据
    table_data = []
    for i, news in enumerate(hotspots):
        # 截断过长的标题和内容
        title = news['title']
        if len(title) > 50:
            title = title[:47] + '...'
        
        content = news.get('content', '')
        if content and len(content) > 100:
            content = content[:97] + '...'
        
        # 格式化时间
        time_str = news.get('time', '')
        
        # 添加到表格
        table_data.append([
            i + 1,
            title,
            time_str,
            f"{news['score']:.4f}"
        ])
    
    # 打印表格
    headers = ["排名", "标题", "发布时间", "热点分数"]
    print("\n" + "="*80)
    print(f"预测热点新闻 (时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')})")
    print("="*80)
    print(tabulate(table_data, headers=headers, tablefmt="grid"))
    print("="*80)

def save_hotspots(hotspots, output_path=None):
    """保存热点新闻到文件
    
    Args:
        hotspots: 热点新闻列表
        output_path: 输出文件路径
    """
    if not hotspots:
        logger.warning("没有找到热点新闻，不保存文件")
        return
    
    if output_path is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_path = os.path.join(DATA_DIR, f'hotspots_{timestamp}.json')
    
    # 保存为JSON
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(hotspots, f, ensure_ascii=False, indent=2)
    
    logger.info(f"热点新闻已保存到: {output_path}")

def main():
    """主函数"""
    print("\n" + "="*50)
    print("社交热点预测系统")
    print("="*50)
    
    # 加载模型和数据
    graph_type = 'heterogeneous'  # 或 'homogeneous'
    model, data, mappings = load_model_and_data(graph_type)
    
    if model is None or data is None or mappings is None:
        logger.error("加载模型或数据失败")
        return
    
    # 加载新闻数据
    news_df = load_news_data()
    
    if news_df is None:
        logger.error("加载新闻数据失败")
        return
    
    # 预测热点
    hotspots = predict_hotspots(model, data, news_df, mappings, top_k=TOP_K_PREDICTIONS)
    
    # 显示热点
    display_hotspots(hotspots)
    
    # 保存热点
    save_hotspots(hotspots)
    
    print("\n预测完成！热点新闻已显示在上方表格中。")

if __name__ == "__main__":
    main() 