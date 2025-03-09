"""
内存优化版MIND数据集处理脚本 - 适用于16GB内存电脑
"""

import os
import pandas as pd
import numpy as np
import json
from tqdm import tqdm
import logging
import sys
import gc
from collections import defaultdict
import torch

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

def read_news_data_chunked(filepath, chunk_size=10000, max_news=None):
    """分块读取新闻数据以减少内存使用"""
    logger.info(f"分批读取新闻数据: {filepath}")
    
    news_data = []
    count = 0
    
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc="读取新闻数据"):
            if max_news and count >= max_news:
                break
                
            news = line.strip().split('\t')
            if len(news) >= 3:
                news_id = news[0]
                category = news[1]
                subcategory = news[2]
                
                news_data.append({
                    'news_id': news_id,
                    'category': category,
                    'subcategory': subcategory
                })
                
                count += 1
                
                # 达到chunk_size时，创建DataFrame并处理
                if len(news_data) >= chunk_size:
                    yield pd.DataFrame(news_data)
                    news_data = []  # 清空列表节省内存
    
    # 处理剩余的数据
    if news_data:
        yield pd.DataFrame(news_data)

def extract_entities_chunked(news_chunks):
    """从分块新闻数据中提取实体"""
    logger.info("提取实体...")
    
    # 使用集合作为数据结构，降低内存使用
    category_entities = set()
    
    for chunk in news_chunks:
        for _, row in chunk.iterrows():
            if pd.notna(row['category']):
                category_entities.add(row['category'])
            if pd.notna(row['subcategory']):
                category_entities.add(row['subcategory'])
        
        # 及时清理内存
        del chunk
        gc.collect()
    
    # 创建实体词典
    entity_list = sorted(list(category_entities))
    entity_dict = {entity: idx for idx, entity in enumerate(entity_list)}
    
    logger.info(f"提取了 {len(entity_dict)} 个实体")
    
    # 保存实体词典
    with open(os.path.join(PROCESSED_DATA_DIR, "mind_full_entity_dict.json"), 'w') as f:
        json.dump(entity_dict, f)
    
    return entity_dict

def read_behaviors_chunked(filepath, chunk_size=5000, max_behaviors=None):
    """分块读取用户行为数据"""
    logger.info(f"分批读取用户行为数据: {filepath}")
    
    behaviors_data = []
    count = 0
    
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc="读取用户行为"):
            if max_behaviors and count >= max_behaviors:
                break
                
            behavior = line.strip().split('\t')
            if len(behavior) >= 4:
                user_id = behavior[1]
                time = behavior[2]
                
                # 减少数据 - 不存储历史记录以节省内存
                if len(behavior) >= 5:
                    impressions = behavior[4].split()
                    
                    # 解析impressions，格式为 "news_id-label"
                    for impression in impressions:
                        parts = impression.split('-')
                        if len(parts) == 2:
                            news_id, label = parts
                            click = int(label)
                            behaviors_data.append({
                                'user_id': user_id,
                                'time': time,
                                'news_id': news_id,
                                'click': click
                            })
                
                count += 1
                
                # 达到chunk_size时，创建DataFrame并处理
                if len(behaviors_data) >= chunk_size:
                    yield pd.DataFrame(behaviors_data)
                    behaviors_data = []  # 清空列表节省内存
    
    # 处理剩余的数据
    if behaviors_data:
        yield pd.DataFrame(behaviors_data)

def assign_entities_to_news_chunked(news_chunks, entity_dict):
    """为每个新闻分配实体，使用迭代器减少内存使用"""
    logger.info("为新闻分配实体...")
    
    news_entities = {}
    
    for chunk in news_chunks:
        for _, row in tqdm(chunk.iterrows(), desc="分配实体"):
            news_id = row['news_id']
            entities = []
            
            # 添加分类和子分类作为实体
            if pd.notna(row['category']) and row['category'] in entity_dict:
                entities.append(row['category'])
            if pd.notna(row['subcategory']) and row['subcategory'] in entity_dict:
                entities.append(row['subcategory'])
                
            if entities:  # 只保存有实体的新闻
                news_entities[news_id] = entities
        
        # 及时清理内存
        del chunk
        gc.collect()
    
    return news_entities

def create_graph_data_chunked(behavior_chunks, news_entities, entity_dict, split_name, max_rows=100000):
    """分批创建图数据，限制输出大小以节省内存"""
    logger.info(f"为 {split_name} 创建图数据...")
    
    # 收集所有行为数据，用于计算热点
    all_behaviors = []
    news_clicks = defaultdict(int)
    
    # 第一遍：统计点击
    total_behaviors = 0
    for chunk in behavior_chunks:
        for _, row in chunk.iterrows():
            news_id = row['news_id']
            if row['click'] == 1:
                news_clicks[news_id] += 1
            total_behaviors += 1
        
        # 及时清理内存
        del chunk
        gc.collect()
    
    logger.info(f"处理了 {total_behaviors} 条行为数据，{len(news_clicks)} 个新闻有点击")
    
    # 计算热点比例
    news_click_items = [(news_id, click_count) for news_id, click_count in news_clicks.items()]
    news_click_items.sort(key=lambda x: x[1], reverse=True)
    
    # 处理测试集特殊情况 - 如果没有点击数据
    if len(news_clicks) == 0:
        logger.warning(f"测试集中没有点击数据，将使用训练集或验证集的热点新闻信息")
        
        # 这里我们只是创建一个空的热点集合，测试集主要是用于评估而不是训练
        hot_news = set()
        
        # 另一种选择是从之前保存的文件中加载热点信息
        try:
            # 尝试从训练集文件中获取热点信息
            train_files = [f for f in os.listdir(PROCESSED_DATA_DIR) 
                          if f.startswith(f"mind_full_train_") and f.endswith(".csv")]
            
            if train_files:
                # 只取一个文件来提取热点信息
                sample_file = os.path.join(PROCESSED_DATA_DIR, train_files[0])
                df = pd.read_csv(sample_file)
                
                # 获取标记为热点的新闻ID
                hot_news = set(df[df['is_hot'] == True]['news_id'].unique())
                logger.info(f"从训练集文件中提取了 {len(hot_news)} 个热点新闻")
                
                del df
                gc.collect()
        except Exception as e:
            logger.warning(f"尝试从训练集提取热点信息失败: {str(e)}")
            # 使用空热点集合继续处理
            hot_news = set()
    else:
        # 正常情况：取前30%的新闻作为热点
        top_percent = 0.3
        num_hot = max(1, int(len(news_click_items) * top_percent))
        hot_news = set([news_id for news_id, _ in news_click_items[:num_hot]])
        
        # 使用条件表达式避免除零错误
        hot_ratio = len(hot_news)/len(news_clicks) if len(news_clicks) > 0 else 0
        logger.info(f"识别了 {len(hot_news)} 个热点新闻，占比 {hot_ratio:.3f}")
    
    # 释放内存
    del news_click_items
    del news_clicks
    gc.collect()
    
    # 第二遍：重新读取数据，创建图数据
    # 重新加载行为数据
    if split_name == "train":
        filepath = os.path.join(RAW_DATA_DIR, "mind_full", "train", "behaviors.tsv")
    elif split_name == "val":
        filepath = os.path.join(RAW_DATA_DIR, "mind_full", "dev", "behaviors.tsv")
    else:  # test
        filepath = os.path.join(RAW_DATA_DIR, "mind_full", "test", "behaviors.tsv")
    
    # 分批处理输出
    output_chunk = []
    total_processed = 0
    chunk_id = 0
    
    behavior_chunks = read_behaviors_chunked(filepath, chunk_size=5000, max_behaviors=None)
    
    for chunk in behavior_chunks:
        for _, row in chunk.iterrows():
            news_id = row['news_id']
            entities = news_entities.get(news_id, [])
            
            # 只处理有实体的新闻
            if entities:
                is_hot = news_id in hot_news
                
                output_chunk.append({
                    'user_id': row['user_id'],
                    'news_id': news_id,
                    'entities': entities,
                    'is_hot': is_hot
                })
                
                total_processed += 1
                
                # 达到最大处理行数时保存并清空
                if len(output_chunk) >= max_rows:
                    df = pd.DataFrame(output_chunk)
                    output_file = os.path.join(PROCESSED_DATA_DIR, f"mind_full_{split_name}_{chunk_id}.csv")
                    df.to_csv(output_file, index=False)
                    logger.info(f"保存了 {len(df)} 条记录到 {output_file}")
                    
                    # 清空列表节省内存
                    output_chunk = []
                    chunk_id += 1
                    
                    # 清理内存
                    del df
                    gc.collect()
        
        # 及时清理内存
        del chunk
        gc.collect()
    
    # 保存剩余数据
    if output_chunk:
        df = pd.DataFrame(output_chunk)
        output_file = os.path.join(PROCESSED_DATA_DIR, f"mind_full_{split_name}_{chunk_id}.csv")
        df.to_csv(output_file, index=False)
        logger.info(f"保存了 {len(df)} 条记录到 {output_file}")
        
        # 清理内存
        del df
        del output_chunk
        gc.collect()
    
    # 创建合并文件的索引
    index_file = os.path.join(PROCESSED_DATA_DIR, f"mind_full_{split_name}.txt")
    with open(index_file, 'w') as f:
        for i in range(chunk_id + 1):
            f.write(f"mind_full_{split_name}_{i}.csv\n")
    
    logger.info(f"总共处理了 {total_processed} 条记录，分成 {chunk_id + 1} 个文件")
    
    return total_processed

def process_mind_full():
    """处理完整MIND数据集的主函数 - 内存优化版"""
    logger.info("开始处理完整MIND数据集...")
    
    # 创建目录
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    
    # 设置内存限制参数
    MAX_NEWS = 200000       # 最多处理的新闻数量，调整此值以适应内存
    MAX_BEHAVIORS = 500000  # 最多处理的用户行为数量，调整此值以适应内存
    CHUNK_SIZE = 5000       # 每批处理的数据量
    MAX_OUTPUT_ROWS = 50000 # 每个输出文件的最大行数
    
    # 读取新闻数据
    train_news_file = os.path.join(RAW_DATA_DIR, "mind_full", "train", "news.tsv")
    news_chunks = list(read_news_data_chunked(train_news_file, CHUNK_SIZE, MAX_NEWS))
    
    # 提取实体
    entity_dict = extract_entities_chunked(news_chunks)
    
    # 读取新闻数据 - 重新读取以便分配实体
    news_chunks = list(read_news_data_chunked(train_news_file, CHUNK_SIZE, MAX_NEWS))
    
    # 为新闻分配实体
    news_entities = assign_entities_to_news_chunked(news_chunks, entity_dict)
    
    # 清理内存
    del news_chunks
    gc.collect()
    
    # 处理训练集
    train_behaviors_file = os.path.join(RAW_DATA_DIR, "mind_full", "train", "behaviors.tsv")
    train_chunks = read_behaviors_chunked(train_behaviors_file, CHUNK_SIZE, MAX_BEHAVIORS)
    train_processed = create_graph_data_chunked(train_chunks, news_entities, entity_dict, "train", MAX_OUTPUT_ROWS)
    
    # 处理验证集
    dev_behaviors_file = os.path.join(RAW_DATA_DIR, "mind_full", "dev", "behaviors.tsv")
    dev_chunks = read_behaviors_chunked(dev_behaviors_file, CHUNK_SIZE, MAX_BEHAVIORS // 5)  # 验证集通常更小
    dev_processed = create_graph_data_chunked(dev_chunks, news_entities, entity_dict, "val", MAX_OUTPUT_ROWS)
    
    # 处理测试集
    test_behaviors_file = os.path.join(RAW_DATA_DIR, "mind_full", "test", "behaviors.tsv")
    test_chunks = read_behaviors_chunked(test_behaviors_file, CHUNK_SIZE, MAX_BEHAVIORS // 5)  # 测试集通常更小
    test_processed = create_graph_data_chunked(test_chunks, news_entities, entity_dict, "test", MAX_OUTPUT_ROWS)
    
    logger.info(f"完整MIND数据集处理完成")
    logger.info(f"训练集: {train_processed} 条记录")
    logger.info(f"验证集: {dev_processed} 条记录")
    logger.info(f"测试集: {test_processed} 条记录")
    logger.info(f"实体数量: {len(entity_dict)}")

if __name__ == "__main__":
    process_mind_full()
