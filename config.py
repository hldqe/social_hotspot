"""
项目配置文件
"""

import os
import torch
from datetime import datetime

# 数据路径
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")

# 确保目录存在
os.makedirs(RAW_DATA_DIR, exist_ok=True)
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

# 模型保存目录
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "saved_models")
os.makedirs(MODEL_DIR, exist_ok=True)

# 图表保存目录
FIGURE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
os.makedirs(FIGURE_DIR, exist_ok=True)

# 日志文件
LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, "training.log")

# 模型配置
MODEL_TYPE = "SimpleGNN"  # 可选: "SimpleGNN", "GraphSAGE", "GAT"
INPUT_DIM = 128  # 根据实际特征维度调整
HIDDEN_DIM = 256  # 增加隐藏层维度，适应更复杂的数据
OUTPUT_DIM = 1  # 二分类问题：热点与非热点，使用sigmoid激活
NUM_LAYERS = 3    # 可能需要更深的网络
DROPOUT = 0.3     # 降低dropout
USE_BATCH_NORM = True

# 训练配置
NUM_EPOCHS = 100
LEARNING_RATE = 0.001     # 可能需要调整学习率
WEIGHT_DECAY = 1e-5       # 增加正则化，防止过拟合
EARLY_STOPPING = 30  # 从20增加到30
BATCH_SIZE = 1024         # 增大批处理大小

# 图构建参数
MIN_ENTITY_FREQ = 5  # 实体最小出现频率
MAX_VOCABULARY_SIZE = 10000  # 最大词汇表大小
HOTSPOT_THRESHOLD = 0.05  # 热点阈值

# 设备配置
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 数据处理配置
MAX_TEXT_LENGTH = 1000  # 最大文本长度
MAX_ENTITIES_PER_DOC = 20  # 每篇文档最多提取的实体数量

# 图构建配置
MIN_EDGE_WEIGHT = 0.3   # 最小边权重
MAX_NEIGHBORS = 15      # 每个节点最大邻居数

# 日志配置
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# 可视化配置
VIZ_MAX_NODES = 100     # 可视化时最大节点数
VIZ_NODE_SIZE_FACTOR = 10  # 节点大小因子
VIZ_EDGE_WIDTH_FACTOR = 5  # 边宽度因子

# 数据相关配置
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "saved_models")

# 确保目录存在
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# MongoDB配置
MONGO_URI = "mongodb://localhost:27017/"
MONGO_DB = "social_hotspot"
NEWS_COLLECTION = "sina_news"

# 图构建配置
MAX_ENTITY_PER_NEWS = 10  # 每篇新闻最多提取的实体数量
MIN_NEWS_PER_ENTITY = 2  # 每个实体最少关联的新闻数量

# 图神经网络配置
EMBEDDING_DIM = 128  # 嵌入维度
NUM_GNN_LAYERS = 2  # GNN层数

# 热点预测配置
TOP_K_PREDICTIONS = 10  # 预测前K个热点
PREDICTION_THRESHOLD = 0.5  # 预测阈值

# 路径配置
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULT_DIR = os.path.join(BASE_DIR, "results")

# 确保目录存在
for dir_path in [RESULT_DIR]:
    os.makedirs(dir_path, exist_ok=True)

# 添加随机种子（用于可视化和模型训练）
RANDOM_SEED = 42  # 之前在图可视化中我们手动使用了42

# 数据集拆分比例（目前没有明确定义）
TRAIN_SPLIT = 0.7
VALIDATION_SPLIT = 0.15
TEST_SPLIT = 0.15

# 类别权重 - 解决类别不平衡问题
# [非热点权重, 热点权重] - 热点类别权重更高
CLASS_WEIGHTS = [1.0, 10.0]  

# 评估指标阈值
PREDICTION_THRESHOLD = 0.5 

# 添加EnhancedGNN相关配置
ENHANCED_MODEL_CONFIG = {
    'DROPOUT': 0.4,  # 适当的dropout率
    'NUM_LAYERS': 3,  # 更深的网络
    'FOCAL_ALPHA': 0.8,  # Focal Loss的alpha参数
    'FOCAL_GAMMA': 2.0,  # Focal Loss的gamma参数
    'NOISE_LEVEL': 0.05  # 特征噪声级别
} 

# 大规模数据配置
LARGE_BATCH_SIZE = 10000  # 图构建时每批处理的数据量
MAX_GRAPH_SIZE = 100000   # 最大图节点数量，超过此数量将进行采样
GRAPH_SAMPLE_RATE = 0.8   # 如果图太大，按此比例采样

# 新增配置用于大规模图
NODE_SAMPLING = True      # 是否使用节点采样
SAMPLING_NEIGHBORS = [15, 10, 5]  # 每层采样的邻居数 