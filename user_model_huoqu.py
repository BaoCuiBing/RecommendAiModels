"""
简化版推荐模型加载和使用模块
只需指定一个文件夹路径，即可自动加载推荐所需的所有模型和数据文件
"""

import os
import joblib
import numpy as np
import pandas as pd
import json
import logging
import random
from typing import List, Dict, Any, Union, Optional, Tuple
import concurrent.futures
import time

# 设置日志记录 - 降低日志级别以减少I/O开销
logging.basicConfig(
    level=logging.WARNING,  # 将INFO改为WARNING减少日志输出
    format='%(asctime)s - %(levelname)s - %(message)s',  # 简化日志格式
    handlers=[
        logging.FileHandler("recommendation.log", encoding='utf-8', delay=True),  # 添加delay=True延迟文件创建
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("recommendation")

# 全局模型缓存
_MODEL_CACHE = {}

class RecommendationModel:
    """
    简化版推荐模型类
    只需加载模型文件夹，自动识别和加载所需的模型和数据文件
    """
    
    def __init__(self, model_dir: str = None):
        """
        初始化推荐模型
        
        Args:
            model_dir: 模型文件夹路径，包含模型文件和相关数据
        """
        # 模型相关变量
        self.model_dir = model_dir
        self.model = None
        self.metadata = {}
        self.question_data = None
        self._question_data_loaded = False
        self.feature_mapping = {}
        self.is_loaded = False
        
        # 如果提供了模型目录，立即加载模型
        if model_dir:
            # 检查缓存中是否已存在
            global _MODEL_CACHE
            if model_dir in _MODEL_CACHE:
                # 复用缓存的模型实例属性
                cached_model = _MODEL_CACHE[model_dir]
                self.model = cached_model.model
                self.metadata = cached_model.metadata
                self.feature_mapping = cached_model.feature_mapping
                self.is_loaded = cached_model.is_loaded
                logger.debug(f"从缓存加载模型: {model_dir}")
            else:
                # 加载并加入缓存
                self.load_model_directory(model_dir)
                if self.is_loaded:
                    _MODEL_CACHE[model_dir] = self
    
    def load_model_directory(self, directory_path: str) -> bool:
        """
        从指定目录加载模型和所有相关文件
        
        Args:
            directory_path: 包含模型和相关文件的目录路径
            
        Returns:
            bool: 加载是否成功
        """
        start_time = time.time()
        try:
            if not os.path.exists(directory_path):
                logger.error(f"模型目录不存在: {directory_path}")
                return False
            
            self.model_dir = directory_path
            logger.info(f"正在从目录加载模型: {self.model_dir}")
            
            # 查找模型文件 (优先寻找.joblib文件)
            model_file = self._find_model_file()
            if not model_file:
                return False
            
            # 并行加载必要文件
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
                # 提交模型加载任务
                model_future = executor.submit(self._load_model_file, model_file)
                
                # 提交元数据加载任务
                metadata_future = executor.submit(self._load_metadata_file)
                
                # 等待模型加载完成（必须的）
                self.model = model_future.result()
                
                # 尝试获取元数据结果
                try:
                    self.metadata = metadata_future.result() or {}
                except Exception as e:
                    logger.warning(f"加载元数据失败: {str(e)}")
                    self.metadata = {}
            
            # 加载特征映射 (如果存在) - 延迟到需要时再加载
            self._load_feature_mapping()
            
            # 题目数据使用懒加载策略，只在需要时才加载
            
            self.is_loaded = self.model is not None
            if self.is_loaded:
                logger.info(f"模型加载完成，耗时: {time.time() - start_time:.2f}秒")
            return self.is_loaded
        
        except Exception as e:
            logger.error(f"加载模型目录失败: {str(e)}")
            return False
    
    def _find_model_file(self) -> Optional[str]:
        """查找模型文件"""
        # 优先查找.joblib文件
        for filename in os.listdir(self.model_dir):
            if filename.endswith('.joblib') and filename != 'model_metadata.joblib':
                return os.path.join(self.model_dir, filename)
        
        # 回退到.pkl文件
        for filename in os.listdir(self.model_dir):
            if filename.endswith('.pkl') and filename != 'model_metadata.pkl':
                logger.warning(f"使用旧格式模型文件: {filename}，建议升级为.joblib格式")
                return os.path.join(self.model_dir, filename)
        
        logger.error(f"在目录中未找到模型文件: {self.model_dir}")
        return None
    
    def _load_model_file(self, file_path: str) -> Any:
        """加载模型文件"""
        try:
            start_time = time.time()
            model = joblib.load(file_path)
            logger.info(f"已加载模型: {file_path}，耗时: {time.time() - start_time:.2f}秒")
            return model
        except Exception as e:
            logger.error(f"加载模型文件失败: {str(e)}")
            return None
    
    def _load_metadata_file(self) -> Dict:
        """加载元数据文件"""
        try:
            # 优先寻找.joblib格式
            metadata_path = os.path.join(self.model_dir, 'model_metadata.joblib')
            if not os.path.exists(metadata_path):
                # 尝试找.pkl格式(向后兼容)
                old_metadata_path = os.path.join(self.model_dir, 'model_metadata.pkl')
                if os.path.exists(old_metadata_path):
                    metadata_path = old_metadata_path
                    logger.warning("使用旧格式元数据文件，建议升级为.joblib格式")
                else:
                    return {}
            
            metadata = joblib.load(metadata_path)
            logger.info(f"已加载模型元数据: {metadata_path}")
            return metadata
        except Exception as e:
            logger.warning(f"加载元数据文件失败: {str(e)}")
            return {}
    
    def _load_feature_mapping(self) -> None:
        """加载特征映射"""
        try:
            feature_mapping_path = os.path.join(self.model_dir, 'feature_mapping.json')
            if os.path.exists(feature_mapping_path):
                with open(feature_mapping_path, 'r', encoding='utf-8') as f:
                    self.feature_mapping = json.load(f)
                logger.info(f"已加载特征映射: {feature_mapping_path}")
        except Exception as e:
            logger.warning(f"加载特征映射失败: {str(e)}")
            self.feature_mapping = {}
    
    def _lazy_load_question_data(self) -> None:
        """懒加载题目数据"""
        if not self._question_data_loaded and self.model_dir:
            try:
                questions_path = os.path.join(self.model_dir, 'questions.csv')
                if os.path.exists(questions_path):
                    start_time = time.time()
                    self.question_data = pd.read_csv(questions_path, encoding='utf-8')
                    self._question_data_loaded = True
                    logger.info(f"已加载题目数据: {questions_path} ({len(self.question_data)} 条记录)，耗时: {time.time() - start_time:.2f}秒")
            except Exception as e:
                logger.warning(f"加载题目数据失败: {str(e)}")
                self.question_data = None
    
    def recommend_questions(self, user_id: int, title_length: int = 10,
                          difficulty_level: int = 3, **kwargs) -> List[int]:
        """
        为指定用户推荐题目
        
        Args:
            user_id: 用户ID
            title_length: 推荐题目数量
            difficulty_level: 难度级别 (1-5)
            **kwargs: 其他用户特征参数
            
        Returns:
            List[int]: 推荐题目ID列表
        """
        start_time = time.time()
        try:
            # 确保模型已加载
            if not self.is_loaded:
                logger.warning("模型未加载，无法推荐题目")
                return self._generate_fallback_recommendations(title_length)
            
            # 获取用户特征
            user_features = self._extract_user_features(user_id, difficulty_level, **kwargs)
            
            # 获取候选题目
            candidates = self._get_candidate_questions(user_id)
            
            # 如果没有足够的候选题目，使用冷启动策略
            if len(candidates) < title_length:
                logger.info(f"候选题目不足 ({len(candidates)} < {title_length})，使用冷启动策略")
                return self._generate_fallback_recommendations(title_length)
            
            # 优化：使用批量处理计算分数
            if len(candidates) > 100:
                # 如果候选题目太多，随机抽样减少计算量
                np.random.shuffle(candidates)
                candidates = candidates[:100]
            
            # 计算每个候选题目的分数
            scores = self._batch_calculate_scores(user_features, candidates)
            
            # 排序并返回前N个题目，确保转换为原生int类型
            recommendations = [int(qid) for qid, _ in sorted(scores, key=lambda x: x[1], reverse=True)[:title_length]]
            
            logger.info(f"为用户 {user_id} 生成了 {len(recommendations)} 条推荐，耗时: {time.time() - start_time:.2f}秒")
            return recommendations
            
        except Exception as e:
            logger.error(f"生成推荐时出错: {str(e)}")
            return self._generate_fallback_recommendations(title_length)
    
    def _batch_calculate_scores(self, user_features: Dict[str, Any], candidates: np.ndarray) -> List[Tuple[int, float]]:
        """批量计算推荐分数"""
        # 如果有模型且支持批量预测
        if self.model is not None and hasattr(self.model, 'predict_proba'):
            try:
                # 真实实现中应该根据实际模型准备批量特征
                # 这里仍使用简化版的得分计算方式，但使用numpy向量化操作提高性能
                base_score = 0.5 + 0.5 * (user_features['knowledge_level'] - user_features['error_rate'])
                # 生成随机分数向量
                random_components = np.random.uniform(-0.2, 0.2, size=len(candidates))
                scores = base_score + random_components
                
                # 返回(题目id, 分数)元组列表，确保id是原生int类型
                return [(int(qid), float(score)) for qid, score in zip(candidates, scores)]
            except Exception as e:
                logger.error(f"批量计算推荐分数失败: {str(e)}")
        
        # 回退到逐个计算
        return [(int(qid), self._calculate_recommendation_score(user_features, qid)) for qid in candidates]
    
    def simple_recommend(self, user_id: int, title_length: int = 10) -> List[int]:
        """
        简化版推荐函数，只需要用户ID和推荐数量
        
        Args:
            user_id: 用户ID
            title_length: 推荐题目数量
            
        Returns:
            List[int]: 推荐题目ID列表
        """
        return self.recommend_questions(user_id, title_length)
    
    def _extract_user_features(self, user_id: int, difficulty_level: int = 3, **kwargs) -> Dict[str, Any]:
        """
        提取用户特征
        
        Args:
            user_id: 用户ID
            difficulty_level: 难度级别 (1-5)
            **kwargs: 其他用户特征
            
        Returns:
            Dict: 用户特征字典
        """
        # 基本用户特征 - 使用字典推导简化代码
        default_values = {
            'knowledge_level': 0.5,
            'learning_speed': 0.5,
            'error_rate': 0.3,
            'active_days': 30,
        }
        
        # 合并默认值和传入的参数
        features = {k: kwargs.get(k, v) for k, v in default_values.items()}
        
        # 添加用户ID和难度偏好
        features.update({
            'user_id': user_id,
            'difficulty_preference': difficulty_level / 5.0,  # 归一化到0-1
        })
        
        return features
    
    def _get_candidate_questions(self, user_id: int) -> np.ndarray:
        """
        获取候选题目列表
        
        Args:
            user_id: 用户ID
            
        Returns:
            np.ndarray: 候选题目ID数组
        """
        # 懒加载题目数据
        if not self._question_data_loaded:
            self._lazy_load_question_data()
        
        # 如果有题目数据，使用题目数据中的所有题目ID
        if self.question_data is not None and not self.question_data.empty:
            # 优化：使用values而不是tolist()，直接获取numpy数组
            return self.question_data['question_id'].values
        
        # 否则生成一些虚拟题目ID
        return np.arange(1, 101)  # 使用numpy.arange替代list(range())，更高效
    
    def _calculate_recommendation_score(self, user_features: Dict[str, Any], question_id: int) -> float:
        """
        计算推荐分数
        
        Args:
            user_features: 用户特征
            question_id: 题目ID
            
        Returns:
            float: 推荐分数
        """
        # 如果有模型，使用模型预测
        if self.model is not None and hasattr(self.model, 'predict_proba'):
            try:
                # 这里应根据实际模型准备输入特征
                # 简化起见，我们直接返回一个基于用户特征和题目ID的随机分数
                base_score = 0.5 + 0.5 * (user_features['knowledge_level'] - user_features['error_rate'])
                random_component = random.uniform(-0.2, 0.2)
                return base_score + random_component
            except Exception as e:
                logger.error(f"使用模型计算分数失败: {str(e)}")
                return random.random()  # 失败时返回随机分数
        
        # 如果没有模型，使用简单的随机策略
        return random.random()
    
    def _generate_fallback_recommendations(self, count: int) -> List[int]:
        """
        生成备选推荐（冷启动策略）
        
        Args:
            count: 需要的推荐数量
            
        Returns:
            List[int]: 推荐题目ID列表
        """
        # 懒加载题目数据
        if not self._question_data_loaded:
            self._lazy_load_question_data()
            
        # 如果有题目数据，从中随机选择
        if self.question_data is not None and len(self.question_data) > 0:
            if len(self.question_data) >= count:
                # 确保返回的是原生int类型
                return [int(x) for x in self.question_data['question_id'].sample(count).tolist()]
            else:
                # 确保返回的是原生int类型
                return [int(x) for x in self.question_data['question_id'].tolist()]
        
        # 返回Python原生int列表
        return list(range(1, count + 1))


# 全局函数用于快速使用
_model_instance = None

def load_model_directory(directory_path: str) -> bool:
    """
    从指定目录加载模型和所有相关文件
    
    Args:
        directory_path: 包含模型和相关文件的目录路径
        
    Returns:
        bool: 加载是否成功
    """
    global _model_instance
    
    # 检查缓存，避免重复加载
    if directory_path in _MODEL_CACHE:
        _model_instance = _MODEL_CACHE[directory_path]
        return _model_instance.is_loaded
    
    # 创建新实例
    _model_instance = RecommendationModel(directory_path)
    return _model_instance.is_loaded

def recommend_questions(user_id: int, title_length: int = 10, **kwargs) -> List[int]:
    """
    获取题目推荐
    
    Args:
        user_id: 用户ID
        title_length: 推荐题目数量
        **kwargs: 其他用户特征
        
    Returns:
        List[int]: 推荐题目ID列表
    """
    global _model_instance
    
    # 如果模型尚未初始化，尝试查找默认位置
    if _model_instance is None:
        # 优化：更高效地检查多个默认路径
        for path in ["./models", "./exported_model", "."]:
            if os.path.exists(path):
                if load_model_directory(path):
                    break
    
    # 如果模型已加载，调用推荐方法
    if _model_instance and _model_instance.is_loaded:
        return _model_instance.recommend_questions(user_id, title_length, **kwargs)
    else:
        # 如果模型未加载，返回备选推荐
        logger.warning("模型未加载，返回备选推荐")
        # 确保返回Python原生int列表
        return list(range(1, title_length + 1))

def simple_recommend(user_id: int, title_length: int = 10) -> List[int]:
    """
    简化版推荐函数，只需要用户ID和推荐数量
    
    Args:
        user_id: 用户ID
        title_length: 推荐题目数量
        
    Returns:
        List[int]: 推荐题目ID列表
    """
    return recommend_questions(user_id, title_length)


if __name__ == "__main__":
    # 测试代码
    # 假设我们有一个导出的模型目录
    test_model_dir = "exported_model"
    
    if os.path.exists(test_model_dir):
        # 测试加载模型
        print(f"从目录加载模型: {test_model_dir}")
        start = time.time()
        success = load_model_directory(test_model_dir)
        print(f"模型加载{'成功' if success else '失败'}，耗时: {time.time() - start:.2f}秒")
        
        # 测试推荐功能
        user_ids = [1, 2, 3, 100]
        for user_id in user_ids:
            start = time.time()
            recommendations = simple_recommend(user_id, 5)
            print(f"用户 {user_id} 的推荐题目: {recommendations}，耗时: {time.time() - start:.2f}秒")
    else:
        print(f"测试目录不存在: {test_model_dir}")
        print("创建模型推荐器实例并使用备选策略...")
        
        recommender = RecommendationModel()
        for user_id in [1, 42, 100]:
            start = time.time()
            recommendations = recommender.simple_recommend(user_id, 5)
            print(f"用户 {user_id} 的备选推荐: {recommendations}，耗时: {time.time() - start:.2f}秒") 