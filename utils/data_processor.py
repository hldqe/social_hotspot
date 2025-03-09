"""
数据处理模块，负责加载、清洗和预处理数据
"""

import os
import pandas as pd
import numpy as np
import json
import re
import logging
from datetime import datetime
from typing import Tuple, List, Dict, Any, Optional
import spacy
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import torch

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

# 加载spaCy模型用于实体识别
try:
    nlp = spacy.load("en_core_web_sm")
    logger.info("成功加载spaCy模型")
except OSError:
    logger.warning("spaCy模型未找到，尝试下载...")
    import subprocess
    subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
    nlp = spacy.load("en_core_web_sm")
    logger.info("成功下载并加载spaCy模型")

class DataProcessor:
    """数据处理类，提供数据加载、清洗和预处理功能"""
    
    def __init__(self, raw_data_dir: str = RAW_DATA_DIR, 
                 processed_data_dir: str = PROCESSED_DATA_DIR):
        """
        初始化数据处理器
        
        Args:
            raw_data_dir: 原始数据目录
            processed_data_dir: 处理后数据保存目录
        """
        self.raw_data_dir = raw_data_dir
        self.processed_data_dir = processed_data_dir
        os.makedirs(self.processed_data_dir, exist_ok=True)
        self.entity_counter = {}  # 用于统计实体频率
        
    def load_data(self, filename: str) -> pd.DataFrame:
        """
        加载原始数据
        
        Args:
            filename: 数据文件名
            
        Returns:
            加载的数据DataFrame
        """
        file_path = os.path.join(self.raw_data_dir, filename)
        logger.info(f"加载数据文件: {file_path}")
        
        if filename.endswith('.csv'):
            df = pd.read_csv(file_path)
        elif filename.endswith('.json'):
            df = pd.read_json(file_path)
        elif filename.endswith('.tsv'):
            df = pd.read_csv(file_path, sep='\t')
        else:
            raise ValueError(f"不支持的文件格式: {filename}")
        
        logger.info(f"成功加载数据，共 {len(df)} 条记录")
        return df
    
    def clean_text(self, text: str) -> str:
        """
        清洗文本数据
        
        Args:
            text: 原始文本
            
        Returns:
            清洗后的文本
        """
        if not isinstance(text, str):
            return ""
        
        # 移除HTML标签
        text = re.sub(r'<.*?>', '', text)
        # 移除URL
        text = re.sub(r'http\S+', '', text)
        # 移除特殊字符
        text = re.sub(r'[^\w\s]', ' ', text)
        # 移除多余空格
        text = re.sub(r'\s+', ' ', text).strip()
        # 截断过长文本
        text = text[:MAX_TEXT_LENGTH]
        
        return text
    
    def extract_entities(self, text: str) -> List[str]:
        """
        从文本中提取实体
        
        Args:
            text: 输入文本
            
        Returns:
            提取的实体列表
        """
        if not text:
            return []
        
        doc = nlp(text)
        # 提取命名实体
        entities = [ent.text.lower() for ent in doc.ents 
                   if len(ent.text) > 1 and ent.label_ in ['PERSON', 'ORG', 'GPE', 'LOC', 'PRODUCT', 'EVENT']]
        
        # 限制每篇文档的实体数量
        entities = entities[:MAX_ENTITIES_PER_DOC]
        
        # 更新实体计数
        for entity in entities:
            self.entity_counter[entity] = self.entity_counter.get(entity, 0) + 1
        
        return entities
    
    def calculate_popularity(self, df: pd.DataFrame, method: str = 'views') -> pd.DataFrame:
        """
        计算内容热度
        
        Args:
            df: 数据DataFrame
            method: 热度计算方法，可选 'views'(浏览量), 'engagement'(互动量), 'combined'(综合)
            
        Returns:
            添加热度列的DataFrame
        """
        if 'popularity' in df.columns:
            logger.info("数据已包含热度列，跳过计算")
            return df
        
        logger.info(f"使用 {method} 方法计算热度")
        
        if method == 'views':
            # 如果有浏览量列，直接使用
            if 'views' in df.columns:
                df['popularity'] = df['views']
            # 否则模拟生成
            else:
                # 基于时间的衰减模型，越新的内容基础热度越高
                now = datetime.now()
                df['time_diff'] = df['publish_time'].apply(
                    lambda x: (now - pd.to_datetime(x)).total_seconds() / 3600)  # 小时差
                # 热度 = 基础热度 * exp(-时间差/半衰期)
                df['popularity'] = np.random.normal(1000, 300, len(df)) * np.exp(-df['time_diff'] / 72)
                df.drop('time_diff', axis=1, inplace=True)
                
        elif method == 'engagement':
            # 如果有互动量相关列，计算综合互动量
            engagement_cols = [col for col in df.columns if col in 
                              ['likes', 'comments', 'shares', 'reactions']]
            if engagement_cols:
                df['popularity'] = df[engagement_cols].sum(axis=1)
            else:
                # 否则模拟生成
                df['popularity'] = np.random.normal(500, 200, len(df))
                
        elif method == 'combined':
            # 综合多种因素
            df['popularity'] = np.random.normal(1000, 300, len(df))
            # 如果有时间列，考虑时间衰减
            if 'publish_time' in df.columns:
                now = datetime.now()
                df['time_diff'] = df['publish_time'].apply(
                    lambda x: (now - pd.to_datetime(x)).total_seconds() / 3600)
                df['popularity'] *= np.exp(-df['time_diff'] / 72)
                df.drop('time_diff', axis=1, inplace=True)
        
        # 归一化热度值到0-1范围
        max_pop = df['popularity'].max()
        min_pop = df['popularity'].min()
        df['popularity_normalized'] = (df['popularity'] - min_pop) / (max_pop - min_pop)
        
        # 二分类标签：热度值大于0.7为热点
        df['is_hot'] = (df['popularity_normalized'] > 0.7).astype(int)
        
        return df
    
    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        预处理数据
        
        Args:
            df: 原始数据DataFrame
            
        Returns:
            预处理后的DataFrame
        """
        logger.info("开始数据预处理")
        
        # 确保必要的列存在
        required_cols = ['news_id', 'title', 'content', 'publish_time']
        for col in required_cols:
            if col not in df.columns:
                if col == 'news_id':
                    df['news_id'] = [f"news_{i}" for i in range(len(df))]
                elif col == 'publish_time':
                    df['publish_time'] = datetime.now()
                else:
                    raise ValueError(f"缺少必要的列: {col}")
        
        # 清洗文本
        logger.info("清洗文本数据")
        df['title_clean'] = df['title'].apply(self.clean_text)
        df['content_clean'] = df['content'].apply(self.clean_text)
        
        # 提取实体
        logger.info("提取实体")
        tqdm.pandas(desc="提取实体")
        df['entities'] = df.progress_apply(
            lambda row: self.extract_entities(f"{row['title_clean']} {row['content_clean']}"), 
            axis=1
        )
        
        # 过滤低频实体
        logger.info(f"过滤低频实体 (最小频率: {MIN_ENTITY_FREQ})")
        frequent_entities = {e for e, c in self.entity_counter.items() if c >= MIN_ENTITY_FREQ}
        df['entities'] = df['entities'].apply(lambda ents: [e for e in ents if e in frequent_entities])
        
        # 计算热度
        df = self.calculate_popularity(df, method='combined')
        
        logger.info("数据预处理完成")
        return df
    
    def split_data(self, df: pd.DataFrame, 
                  val_ratio: float = VALIDATION_RATIO, 
                  test_ratio: float = TEST_RATIO) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        划分训练、验证和测试集
        
        Args:
            df: 预处理后的DataFrame
            val_ratio: 验证集比例
            test_ratio: 测试集比例
            
        Returns:
            (训练集, 验证集, 测试集)
        """
        logger.info(f"划分数据集 (验证集: {val_ratio}, 测试集: {test_ratio})")
        
        # 按时间排序
        if 'publish_time' in df.columns:
            df = df.sort_values('publish_time')
        
        # 先划分出测试集
        train_val_df, test_df = train_test_split(df, test_size=test_ratio, random_state=RANDOM_SEED)
        
        # 再从剩余数据中划分验证集
        train_df, val_df = train_test_split(
            train_val_df, test_size=val_ratio/(1-test_ratio), random_state=RANDOM_SEED
        )
        
        logger.info(f"数据集划分完成 - 训练集: {len(train_df)}, 验证集: {len(val_df)}, 测试集: {len(test_df)}")
        return train_df, val_df, test_df
    
    def save_processed_data(self, train_df: pd.DataFrame, val_df: pd.DataFrame, 
                           test_df: pd.DataFrame, prefix: str = "news") -> None:
        """
        保存处理后的数据
        
        Args:
            train_df: 训练集
            val_df: 验证集
            test_df: 测试集
            prefix: 文件名前缀
        """
        logger.info(f"保存处理后的数据到 {self.processed_data_dir}")
        
        # 保存数据集
        train_df.to_csv(os.path.join(self.processed_data_dir, f"{prefix}_train.csv"), index=False)
        val_df.to_csv(os.path.join(self.processed_data_dir, f"{prefix}_val.csv"), index=False)
        test_df.to_csv(os.path.join(self.processed_data_dir, f"{prefix}_test.csv"), index=False)
        
        # 保存实体词典
        entity_list = sorted([e for e, c in self.entity_counter.items() if c >= MIN_ENTITY_FREQ])
        entity_dict = {entity: idx for idx, entity in enumerate(entity_list)}
        with open(os.path.join(self.processed_data_dir, "entity_dict.json"), 'w') as f:
            json.dump(entity_dict, f)
        
        logger.info(f"数据保存完成，共 {len(entity_dict)} 个实体")
    
    def process_pipeline(self, filename: str, prefix: str = "news") -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        完整的数据处理流程
        
        Args:
            filename: 原始数据文件名
            prefix: 输出文件前缀
            
        Returns:
            (训练集, 验证集, 测试集)
        """
        logger.info(f"开始数据处理流程: {filename}")
        
        # 加载数据
        df = self.load_data(filename)
        
        # 预处理
        df = self.preprocess(df)
        
        # 划分数据集
        train_df, val_df, test_df = self.split_data(df)
        
        # 保存处理后的数据
        self.save_processed_data(train_df, val_df, test_df, prefix)
        
        return train_df, val_df, test_df


# 辅助函数，用于从外部调用
def load_processed_data(data_dir: str = PROCESSED_DATA_DIR, 
                       prefix: str = "news") -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    加载已处理的数据
    
    Args:
        data_dir: 数据目录
        prefix: 文件前缀
        
    Returns:
        (训练集, 验证集, 测试集)
    """
    train_path = os.path.join(data_dir, f"{prefix}_train.csv")
    val_path = os.path.join(data_dir, f"{prefix}_val.csv")
    test_path = os.path.join(data_dir, f"{prefix}_test.csv")
    
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)
    
    # 将字符串形式的实体列表转换回Python列表
    for df in [train_df, val_df, test_df]:
        if 'entities' in df.columns and isinstance(df['entities'].iloc[0], str):
            df['entities'] = df['entities'].apply(eval)
    
    return train_df, val_df, test_df

def load_entity_dict(data_dir: str = PROCESSED_DATA_DIR) -> Dict[str, int]:
    """
    加载实体词典
    
    Args:
        data_dir: 数据目录
        
    Returns:
        实体词典 {实体: 索引}
    """
    dict_path = os.path.join(data_dir, "entity_dict.json")
    with open(dict_path, 'r') as f:
        entity_dict = json.load(f)
    return entity_dict

def download_sample_data(output_dir: str = RAW_DATA_DIR) -> str:
    """
    下载示例数据集
    
    Args:
        output_dir: 输出目录
        
    Returns:
        下载的文件路径
    """
    import requests
    from io import BytesIO
    from zipfile import ZipFile
    
    logger.info("下载示例数据集...")
    
    # 使用Kaggle News Category Dataset的一个小样本
    # 实际使用时，应该使用完整数据集或其他推荐的数据集
    url = "https://github.com/user/repo/raw/main/sample_news_data.zip"
    
    try:
        response = requests.get(url)
        with ZipFile(BytesIO(response.content)) as zip_file:
            zip_file.extractall(output_dir)
        
        output_file = os.path.join(output_dir, "sample_news_data.csv")
        logger.info(f"示例数据下载完成: {output_file}")
        return output_file
    except Exception as e:
        logger.error(f"下载示例数据失败: {str(e)}")
        
        # 如果下载失败，创建一个小的示例数据集
        logger.info("创建模拟数据集...")
        sample_data = {
            'news_id': [f"news_{i}" for i in range(100)],
            'title': [f"Sample News Title {i}" for i in range(100)],
            'content': [
                f"This is sample content for news {i}. It contains entities like Apple, Google, Microsoft, and mentions people like Joe Biden, Elon Musk. It discusses topics such as technology, politics, and economy."
                for i in range(100)
            ],
            'publish_time': pd.date_range(start='2023-01-01', periods=100)
        }
        
        df = pd.DataFrame(sample_data)
        output_file = os.path.join(output_dir, "sample_news_data.csv")
        df.to_csv(output_file, index=False)
        logger.info(f"模拟数据集创建完成: {output_file}")
        return output_file


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="新闻数据处理工具")
    parser.add_argument("--input", type=str, help="输入文件名")
    parser.add_argument("--output_dir", type=str, default=PROCESSED_DATA_DIR, help="输出目录")
    parser.add_argument("--prefix", type=str, default="news", help="输出文件前缀")
    parser.add_argument("--download_sample", action="store_true", help="下载示例数据")
    
    args = parser.parse_args()
    
    if args.download_sample:
        input_file = download_sample_data()
    elif args.input:
        input_file = args.input
    else:
        parser.error("请提供输入文件或使用 --download_sample 下载示例数据")
    
    processor = DataProcessor(processed_data_dir=args.output_dir)
    train_df, val_df, test_df = processor.process_pipeline(input_file, args.prefix)
    
    logger.info("数据处理完成!") 