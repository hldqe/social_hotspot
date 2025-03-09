"""
MIND数据集处理模块，专门处理微软新闻数据集
"""

import os
import pandas as pd
import numpy as np
import json
import logging
from typing import Tuple, Dict, List
import zipfile
import requests
from tqdm import tqdm
import sys

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

class MINDDataProcessor:
    """MIND数据集处理类"""
    
    def __init__(self, raw_data_dir: str = os.path.join(RAW_DATA_DIR, 'mind'),
                 processed_data_dir: str = PROCESSED_DATA_DIR,
                 version: str = 'small'):
        """
        初始化MIND数据处理器
        
        Args:
            raw_data_dir: MIND原始数据目录
            processed_data_dir: 处理后数据保存目录
            version: MIND数据版本，可选'small', 'medium', 'large'
        """
        self.raw_data_dir = raw_data_dir
        self.processed_data_dir = processed_data_dir
        self.version = version
        
        # 创建目录
        os.makedirs(self.raw_data_dir, exist_ok=True)
        os.makedirs(self.processed_data_dir, exist_ok=True)
        
        # MIND数据集下载URL
        self.urls = {
            'small': {
                'train': 'https://mind201910small.blob.core.windows.net/release/MINDsmall_train.zip',
                'dev': 'https://mind201910small.blob.core.windows.net/release/MINDsmall_dev.zip'
            },
            'medium': {
                'train': 'https://mind201910small.blob.core.windows.net/release/MINDmedium_train.zip',
                'dev': 'https://mind201910small.blob.core.windows.net/release/MINDmedium_dev.zip'
            }
        }
    
    def download_dataset(self) -> None:
        """下载MIND数据集"""
        if self.version not in self.urls:
            raise ValueError(f"不支持的版本: {self.version}，支持的版本有: {list(self.urls.keys())}")
        
        for split, url in self.urls[self.version].items():
            zip_path = os.path.join(self.raw_data_dir, f"MIND{self.version}_{split}.zip")
            extract_dir = os.path.join(self.raw_data_dir, split)
            
            # 检查是否已下载
            if os.path.exists(extract_dir) and len(os.listdir(extract_dir)) > 0:
                logger.info(f"{split}集已存在，跳过下载")
                continue
                
            if not os.path.exists(zip_path):
                logger.info(f"下载 {split} 集: {url}")
                
                # 分块下载大文件
                response = requests.get(url, stream=True)
                total_size = int(response.headers.get('content-length', 0))
                block_size = 1024  # 1 KB
                
                progress_bar = tqdm(
                    total=total_size, 
                    unit='iB', 
                    unit_scale=True,
                    desc=f"下载 MIND{self.version}_{split}"
                )
                
                with open(zip_path, 'wb') as f:
                    for data in response.iter_content(block_size):
                        progress_bar.update(len(data))
                        f.write(data)
                progress_bar.close()
            else:
                logger.info(f"文件已存在: {zip_path}")
            
            # 解压文件
            logger.info(f"解压 {zip_path} 到 {extract_dir}")
            os.makedirs(extract_dir, exist_ok=True)
            
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(extract_dir)
    
    def load_news_data(self, split: str = 'train') -> pd.DataFrame:
        """
        加载新闻数据
        
        Args:
            split: 数据集划分，'train'或'dev'
            
        Returns:
            新闻数据DataFrame
        """
        news_file = os.path.join(self.raw_data_dir, split, 'news.tsv')
        if not os.path.exists(news_file):
            raise FileNotFoundError(f"新闻文件不存在: {news_file}，请先下载数据集")
        
        # 定义列名
        columns = ['news_id', 'category', 'subcategory', 'title', 'abstract', 
                  'url', 'title_entities', 'abstract_entities']
        
        # 加载数据
        df = pd.read_csv(news_file, sep='\t', names=columns, quoting=3)
        logger.info(f"加载{split}集新闻数据: {len(df)}条记录")
        
        return df
    
    def load_behaviors_data(self, split: str = 'train') -> pd.DataFrame:
        """
        加载用户行为数据
        
        Args:
            split: 数据集划分，'train'或'dev'
            
        Returns:
            用户行为数据DataFrame
        """
        behaviors_file = os.path.join(self.raw_data_dir, split, 'behaviors.tsv')
        if not os.path.exists(behaviors_file):
            raise FileNotFoundError(f"行为文件不存在: {behaviors_file}，请先下载数据集")
        
        # 定义列名
        columns = ['impression_id', 'user_id', 'time', 'history', 'impressions']
        
        # 加载数据
        df = pd.read_csv(behaviors_file, sep='\t', names=columns, quoting=3)
        logger.info(f"加载{split}集用户行为数据: {len(df)}条记录")
        
        return df
    
    def extract_entities(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        从新闻数据中提取并解析实体
        
        Args:
            df: 新闻数据DataFrame
            
        Returns:
            添加解析后实体的DataFrame
        """
        logger.info("解析新闻实体")
        
        def parse_entities(entities_str):
            if pd.isna(entities_str) or entities_str == '':
                return []
            
            try:
                # 实体格式: "实体ID 实体名 位置 类型;实体ID 实体名 位置 类型"
                entities = []
                for entity in entities_str.split(';'):
                    if entity:
                        parts = entity.strip().split(' ')
                        if len(parts) >= 2:
                            entities.append(parts[1])  # 实体名
                return entities
            except:
                return []
        
        # 解析标题和摘要中的实体
        df['title_entity_list'] = df['title_entities'].apply(parse_entities)
        df['abstract_entity_list'] = df['abstract_entities'].apply(parse_entities)
        
        # 合并实体列表
        df['entities'] = df.apply(lambda x: list(set(x['title_entity_list'] + x['abstract_entity_list'])), axis=1)
        
        return df
    
    def calculate_popularity(self, news_df: pd.DataFrame, behaviors_df: pd.DataFrame) -> pd.DataFrame:
        """
        计算新闻热度
        
        Args:
            news_df: 新闻数据DataFrame
            behaviors_df: 用户行为数据DataFrame
            
        Returns:
            添加热度指标的新闻DataFrame
        """
        logger.info("计算新闻热度")
        
        # 统计每篇新闻被点击的次数
        news_clicks = {}
        news_impressions = {}
        
        for _, row in tqdm(behaviors_df.iterrows(), total=len(behaviors_df), desc="统计新闻点击"):
            # 统计历史点击
            if not pd.isna(row['history']):
                for news_id in row['history'].split():
                    if news_id in news_clicks:
                        news_clicks[news_id] += 1
                    else:
                        news_clicks[news_id] = 1
            
            # 统计展示次数和点击
            if not pd.isna(row['impressions']):
                for impression in row['impressions'].split():
                    parts = impression.split('-')
                    if len(parts) == 2:
                        news_id, click = parts
                        # 增加展示次数
                        if news_id in news_impressions:
                            news_impressions[news_id] += 1
                        else:
                            news_impressions[news_id] = 1
                        
                        # 如果被点击，增加点击次数
                        if click == '1':
                            if news_id in news_clicks:
                                news_clicks[news_id] += 1
                            else:
                                news_clicks[news_id] = 1
        
        # 将点击次数添加到新闻数据中
        news_df['clicks'] = news_df['news_id'].apply(lambda x: news_clicks.get(x, 0))
        news_df['impressions'] = news_df['news_id'].apply(lambda x: news_impressions.get(x, 0))
        
        # 计算点击率
        news_df['ctr'] = news_df.apply(
            lambda x: x['clicks'] / x['impressions'] if x['impressions'] > 0 else 0, 
            axis=1
        )
        
        # 计算热度分数 = 点击次数 * (1 + 点击率)
        news_df['popularity'] = news_df['clicks'] * (1 + news_df['ctr'])
        
        # 归一化热度到0-1范围
        max_pop = news_df['popularity'].max()
        min_pop = news_df['popularity'].min()
        if max_pop > min_pop:
            news_df['popularity_normalized'] = (news_df['popularity'] - min_pop) / (max_pop - min_pop)
        else:
            news_df['popularity_normalized'] = 0
        
        # 二分类标签：热度值大于0.7为热点
        news_df['is_hot'] = (news_df['popularity_normalized'] > 0.7).astype(int)
        
        return news_df
    
    def process(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        处理MIND数据集
        
        Returns:
            (训练集, 验证集, 测试集)
        """
        # 下载数据集
        self.download_dataset()
        
        # 加载训练集数据
        train_news_df = self.load_news_data('train')
        train_behaviors_df = self.load_behaviors_data('train')
        
        # 加载验证集数据
        val_news_df = self.load_news_data('dev')
        val_behaviors_df = self.load_behaviors_data('dev')
        
        # 合并训练集和验证集新闻
        all_news_df = pd.concat([train_news_df, val_news_df]).drop_duplicates(subset=['news_id'])
        all_behaviors_df = pd.concat([train_behaviors_df, val_behaviors_df])
        
        # 提取实体
        all_news_df = self.extract_entities(all_news_df)
        
        # 计算热度
        all_news_df = self.calculate_popularity(all_news_df, all_behaviors_df)
        
        # 根据时间排序
        all_behaviors_df['timestamp'] = pd.to_datetime(all_behaviors_df['time'])
        all_behaviors_df = all_behaviors_df.sort_values('timestamp')
        
        # 获取所有新闻ID
        news_ids = []
        for _, row in all_behaviors_df.iterrows():
            if not pd.isna(row['impressions']):
                news_ids.extend([imp.split('-')[0] for imp in row['impressions'].split()])
        news_ids = list(set(news_ids))
        
        # 过滤出行为数据中的新闻
        filtered_news_df = all_news_df[all_news_df['news_id'].isin(news_ids)]
        
        # 分割数据集
        # 使用时间顺序，前70%作为训练集，接下来15%作为验证集，最后15%作为测试集
        train_ratio, val_ratio = 0.7, 0.15
        total_count = len(filtered_news_df)
        
        train_end_idx = int(total_count * train_ratio)
        val_end_idx = train_end_idx + int(total_count * val_ratio)
        
        sorted_df = filtered_news_df.sort_values('news_id')  # 使用news_id排序确保一致性
        
        train_df = sorted_df.iloc[:train_end_idx]
        val_df = sorted_df.iloc[train_end_idx:val_end_idx]
        test_df = sorted_df.iloc[val_end_idx:]
        
        # 确保数据中的实体列表是列表而非字符串
        for df in [train_df, val_df, test_df]:
            # 确保entities是列表
            if 'entities' in df.columns:
                df['entities'] = df['entities'].apply(lambda x: x if isinstance(x, list) else [])
        
        # 保存处理后的数据
        train_df.to_csv(os.path.join(self.processed_data_dir, "mind_train.csv"), index=False)
        val_df.to_csv(os.path.join(self.processed_data_dir, "mind_val.csv"), index=False)
        test_df.to_csv(os.path.join(self.processed_data_dir, "mind_test.csv"), index=False)
        
        # 保存实体词典
        all_entities = set()
        for entities in filtered_news_df['entities']:
            if isinstance(entities, list):
                all_entities.update(entities)
        
        entity_dict = {entity: idx for idx, entity in enumerate(sorted(all_entities))}
        with open(os.path.join(self.processed_data_dir, "entity_dict.json"), 'w') as f:
            json.dump(entity_dict, f)
        
        logger.info(f"数据处理完成 - 实体总数: {len(entity_dict)}")
        logger.info(f"训练集: {len(train_df)}条, 验证集: {len(val_df)}条, 测试集: {len(test_df)}条")
        
        return train_df, val_df, test_df


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="MIND数据集处理工具")
    parser.add_argument("--version", type=str, default="small", choices=["small", "medium"],
                        help="MIND数据集版本，可选small或medium")
    parser.add_argument("--output_dir", type=str, default=PROCESSED_DATA_DIR,
                        help="处理后数据保存目录")
    
    args = parser.parse_args()
    
    processor = MINDDataProcessor(
        processed_data_dir=args.output_dir,
        version=args.version
    )
    
    train_df, val_df, test_df = processor.process()
    
    logger.info("MIND数据集处理完成!")
