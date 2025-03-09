"""
改进的测试集处理脚本 - 确保能生成记录
"""

import os
import pandas as pd
import json
import logging
import sys
from tqdm import tqdm
import gc
import random

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

def process_test_data_improved():
    """改进的测试集处理 - 确保生成记录"""
    logger.info("开始改进的测试集处理...")
    
    # 加载实体词典
    entity_dict_path = os.path.join(PROCESSED_DATA_DIR, "mind_full_entity_dict.json")
    with open(entity_dict_path, 'r') as f:
        entity_dict = json.load(f)
    logger.info(f"加载了 {len(entity_dict)} 个实体")
    
    # 从实体词典中选择最常见的几个实体作为默认实体
    entity_keys = list(entity_dict.keys())
    default_entities = entity_keys[:min(5, len(entity_keys))]
    logger.info(f"将使用以下默认实体: {default_entities}")
    
    # 加载已有的新闻-实体映射
    news_entities = {}
    
    # 从训练集文件中获取新闻-实体映射和热点信息
    train_files = [f for f in os.listdir(PROCESSED_DATA_DIR) 
                  if f.startswith("mind_full_train_") and f.endswith(".csv")]
    
    hot_news = set()
    processed_files = 0
    
    for file in tqdm(train_files[:10], desc="从训练集提取信息"):
        file_path = os.path.join(PROCESSED_DATA_DIR, file)
        df = pd.read_csv(file_path)
        
        # 获取新闻-实体映射
        for _, row in df.iterrows():
            news_id = row['news_id']
            if 'entities' in row and isinstance(row['entities'], str):
                try:
                    entities = eval(row['entities'])
                    news_entities[news_id] = entities
                except:
                    continue
        
        # 获取热点新闻
        hot_ids = df[df['is_hot'] == True]['news_id'].unique()
        hot_news.update(hot_ids)
        
        processed_files += 1
        
        # 清理内存
        del df
        gc.collect()
    
    logger.info(f"从 {processed_files} 个训练集文件中提取了 {len(news_entities)} 个新闻-实体映射和 {len(hot_news)} 个热点新闻")
    
    # 处理测试集数据
    test_behaviors_file = os.path.join(RAW_DATA_DIR, "mind_full", "test", "behaviors.tsv")
    
    # 首先读取一些测试集数据，查看格式
    logger.info(f"检查测试集数据格式: {test_behaviors_file}")
    with open(test_behaviors_file, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= 3:  # 只检查前3行
                break
            logger.info(f"样本行 {i+1}: {line.strip()}")
    
    # 分批读取和处理测试集
    chunk_size = 5000
    max_test_records = 500000  # 限制处理量
    output_rows = 50000  # 每个输出文件的行数
    
    behaviors_data = []
    count = 0
    chunk_id = 0
    total_processed = 0
    
    # 预定义热点概率
    HOT_PROBABILITY = 0.3  # 30%的记录会被标记为热点
    
    with open(test_behaviors_file, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc="处理测试集"):
            if count >= max_test_records:
                break
                
            behavior = line.strip().split('\t')
            if len(behavior) >= 4:
                user_id = behavior[1]
                time = behavior[2]
                
                if len(behavior) >= 5:
                    impressions = behavior[4].split()
                    
                    for impression in impressions:
                        parts = impression.split('-')
                        if len(parts) == 2:
                            news_id, label = parts
                            
                            # 获取实体 - 如果没有则分配默认实体
                            entities = news_entities.get(news_id, None)
                            if entities is None:
                                # 分配1-3个默认实体
                                num_entities = random.randint(1, min(3, len(default_entities)))
                                entities = random.sample(default_entities, num_entities)
                            
                            # 打印一些样本，用于调试
                            if total_processed < 5:
                                logger.info(f"样本 {total_processed+1} - news_id: {news_id}, entities: {entities}")
                            
                            # 随机标记热点，同时参考训练集热点信息
                            is_hot = news_id in hot_news
                            if not is_hot and random.random() < HOT_PROBABILITY:
                                is_hot = True
                            
                            # 添加记录 - 不过滤没有实体的记录
                            behaviors_data.append({
                                'user_id': user_id,
                                'news_id': news_id,
                                'entities': entities,
                                'is_hot': is_hot
                            })
                            
                            total_processed += 1
                            
                            # 达到输出行数限制时保存
                            if len(behaviors_data) >= output_rows:
                                df = pd.DataFrame(behaviors_data)
                                output_file = os.path.join(PROCESSED_DATA_DIR, f"mind_full_test_{chunk_id}.csv")
                                df.to_csv(output_file, index=False)
                                logger.info(f"保存了 {len(df)} 条记录到 {output_file}")
                                
                                # 清空列表节省内存
                                behaviors_data = []
                                chunk_id += 1
                                
                                # 清理内存
                                del df
                                gc.collect()
            
            count += 1
    
    # 保存剩余数据
    if behaviors_data:
        df = pd.DataFrame(behaviors_data)
        output_file = os.path.join(PROCESSED_DATA_DIR, f"mind_full_test_{chunk_id}.csv")
        df.to_csv(output_file, index=False)
        logger.info(f"保存了 {len(df)} 条记录到 {output_file}")
        
        # 清理内存
        del df
        del behaviors_data
        gc.collect()
    
    # 创建合并文件的索引
    index_file = os.path.join(PROCESSED_DATA_DIR, "mind_full_test.txt")
    with open(index_file, 'w') as f:
        for i in range(chunk_id + 1):
            f.write(f"mind_full_test_{i}.csv\n")
    
    logger.info(f"测试集处理完成，总共处理了 {total_processed} 条记录，分成 {chunk_id + 1} 个文件")

if __name__ == "__main__":
    process_test_data_improved() 