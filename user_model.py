"""
用户模型API，用于获取题目推荐
本模块只负责调用已训练好的模型，获取题目推荐
"""

import os
import joblib
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Union, Tuple
import logging
import random
import threading
from datetime import datetime

# 设置日志记录
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("user_model.log", encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("user_model")

# 线程锁，用于保护模型的线程安全
model_lock = threading.RLock()

class UserModel:
    """用户模型类，表示单个用户及其属性"""
    
    def __init__(self, user_id: int, **kwargs):
        """
        初始化用户模型
        
        Args:
            user_id: 用户ID
            **kwargs: 用户其他属性
        """
        self.user_id = user_id
        
        # 用户基本属性
        self.username = kwargs.get('username', f'用户_{user_id}')
        self.grade = kwargs.get('grade', '初中')
        self.subject_preferences = kwargs.get('subject_preferences', {})
        self.difficulty_preference = kwargs.get('difficulty_preference', 3)  # 1-5
        
        # 学习特征
        self.knowledge_level = kwargs.get('knowledge_level', 0.5)  # 0-1
        self.learning_speed = kwargs.get('learning_speed', 0.5)  # 0-1
        self.error_rate = kwargs.get('error_rate', 0.3)  # 0-1
        self.attempts_count = kwargs.get('attempts_count', 0)
        self.correct_count = kwargs.get('correct_count', 0)
        self.learning_time = kwargs.get('learning_time', 0)  # 分钟
        self.active_time = kwargs.get('active_time', 60)  # 每天活跃分钟数
        
        # 历史记录
        self.question_history = kwargs.get('question_history', [])
        self.last_active = kwargs.get('last_active', datetime.now())
        
        # 推荐器实例
        self.recommender = None
        
    def get_features(self) -> Dict[str, Any]:
        """
        获取用户特征
        
        Returns:
            Dict: 用户特征字典
        """
        return {
            'user_id': self.user_id,
            'knowledge_level': self.knowledge_level,
            'learning_speed': self.learning_speed,
            'error_rate': self.error_rate,
            'difficulty_preference': self.difficulty_preference,
            'attempts_count': self.attempts_count,
            'correct_count': self.correct_count,
            'accuracy': self.correct_count / max(1, self.attempts_count),
            'learning_time': self.learning_time,
            'active_time': self.active_time,
            'question_count': len(self.question_history)
        }
    
    def update_from_attempt(self, question_id: int, is_correct: bool, 
                          time_spent: int, difficulty: float) -> None:
        """
        根据答题结果更新用户特征
        
        Args:
            question_id: 题目ID
            is_correct: 是否正确
            time_spent: 花费时间(秒)
            difficulty: 题目难度(0-1)
        """
        # 更新答题记录
        self.attempts_count += 1
        if is_correct:
            self.correct_count += 1
        
        # 更新学习时间
        self.learning_time += time_spent / 60  # 转换为分钟
        
        # 添加到历史记录
        self.question_history.append(question_id)
        
        # 更新知识水平和学习速度(简化模型)
        if is_correct:
            # 答对难题，提高知识水平增长
            knowledge_gain = 0.01 * (1 + difficulty)
            self.knowledge_level = min(1.0, self.knowledge_level + knowledge_gain)
            
            # 回答速度影响学习速度
            if time_spent < 30:  # 快速答对
                self.learning_speed = min(1.0, self.learning_speed + 0.005)
        else:
            # 答错简单题，降低知识水平更多
            knowledge_loss = 0.01 * (2 - difficulty)
            self.knowledge_level = max(0.0, self.knowledge_level - knowledge_loss)
            
            # 答错会轻微降低学习速度
            self.learning_speed = max(0.0, self.learning_speed - 0.002)
        
        # 更新错误率
        self.error_rate = 1.0 - (self.correct_count / self.attempts_count)
        
        # 更新最后活跃时间
        self.last_active = datetime.now()
    
    def get_recommendations(self, count: int = 10) -> List[int]:
        """
        获取推荐题目
        
        Args:
            count: 推荐题目数量
            
        Returns:
            List[int]: 推荐题目ID列表
        """
        # 如果没有初始化推荐器，初始化一个
        if self.recommender is None:
            self.recommender = UserRecommender()
            self.recommender.load_model()
        
        # 使用推荐器获取推荐
        recommendations = self.recommender.get_recommendations(self.user_id, count)
        return recommendations
    
    def save(self) -> bool:
        """
        保存用户模型到文件
        
        Returns:
            bool: 是否保存成功
        """
        try:
            user_dir = "users"
            if not os.path.exists(user_dir):
                os.makedirs(user_dir)
            
            filename = os.path.join(user_dir, f"user_{self.user_id}.pkl")
            joblib.dump(self, filename)
            logger.info(f"用户模型已保存: {filename}")
            return True
        except Exception as e:
            logger.error(f"保存用户模型失败: {str(e)}")
            return False
    
    @classmethod
    def load(cls, user_id: int) -> 'UserModel':
        """
        从文件加载用户模型
        
        Args:
            user_id: 用户ID
            
        Returns:
            UserModel: 用户模型实例，如果不存在则创建新的
        """
        try:
            filename = os.path.join("users", f"user_{user_id}.pkl")
            if os.path.exists(filename):
                model = joblib.load(filename)
                logger.info(f"加载用户模型: {filename}")
                return model
        except Exception as e:
            logger.error(f"加载用户模型失败: {str(e)}")
        
        # 如果加载失败或文件不存在，创建新的用户模型
        logger.info(f"创建新用户模型: user_id={user_id}")
        return cls(user_id)

class UserRecommender:
    """用户题目推荐类，管理模型加载和推荐逻辑"""
    
    def __init__(self, model_dir: str = "models"):
        """
        初始化推荐器
        
        Args:
            model_dir: 模型存储目录
        """
        self.model_dir = model_dir
        self.model = None
        self.model_metadata = {}
        self.feature_importances = {}
        self.category_weights = {}
        self.question_data = None
        self.user_data = {}
        self.cold_start_items = []
        self.is_loaded = False
        
    def load_model(self, model_name: str = "recommender_model.pkl") -> bool:
        """
        加载已训练好的模型
        
        Args:
            model_name: 模型文件名
            
        Returns:
            bool: 模型是否加载成功
        """
        try:
            model_path = os.path.join(self.model_dir, model_name)
            
            with model_lock:
                logger.info(f"加载模型: {model_path}")
                
                if not os.path.exists(model_path):
                    logger.error(f"模型文件不存在: {model_path}")
                    return False
                
                # 加载模型和元数据
                self.model = joblib.load(model_path)
                
                # 加载问题数据集(如果存在)
                question_path = os.path.join("data", "questions.csv")
                if os.path.exists(question_path):
                    self.question_data = pd.read_csv(question_path, encoding='utf-8')
                    logger.info(f"加载题目数据: {len(self.question_data)}条记录")
                    
                # 加载模型元数据(如果存在)
                metadata_path = os.path.join(self.model_dir, "model_metadata.pkl")
                if os.path.exists(metadata_path):
                    self.model_metadata = joblib.load(metadata_path)
                    logger.info("加载模型元数据成功")
                    
                    # 提取特征重要性和类别权重
                    if 'feature_importances' in self.model_metadata:
                        self.feature_importances = self.model_metadata['feature_importances']
                    if 'category_weights' in self.model_metadata:
                        self.category_weights = self.model_metadata['category_weights']
                    
                # 准备冷启动题目列表
                self._prepare_cold_start_items()
                
                self.is_loaded = True
                logger.info("模型加载完成")
                return True
                
        except Exception as e:
            logger.error(f"加载模型失败: {str(e)}")
            return False
    
    def _prepare_cold_start_items(self) -> None:
        """准备冷启动题目列表，用于新用户推荐"""
        if self.question_data is not None:
            # 按不同难度和类别准备一组多样化的题目
            simple_questions = self.question_data[self.question_data['difficulty_level'] == '简单']
            medium_questions = self.question_data[self.question_data['difficulty_level'] == '中等']
            
            # 为新用户准备一些入门级题目
            self.cold_start_items = []
            
            # 添加各类简单题目
            if not simple_questions.empty:
                categories = simple_questions['category'].unique()
                for category in categories[:min(5, len(categories))]:
                    category_questions = simple_questions[simple_questions['category'] == category]
                    if not category_questions.empty:
                        self.cold_start_items.extend(
                            category_questions['question_id'].sample(
                                min(3, len(category_questions))
                            ).tolist()
                        )
            
            # 添加一些中等难度题目
            if not medium_questions.empty:
                self.cold_start_items.extend(
                    medium_questions['question_id'].sample(
                        min(5, len(medium_questions))
                    ).tolist()
                )
            
            # 确保有足够的冷启动题目
            if len(self.cold_start_items) < 10 and not self.question_data.empty:
                additional_items = self.question_data['question_id'].sample(
                    min(10 - len(self.cold_start_items), len(self.question_data))
                ).tolist()
                self.cold_start_items.extend(additional_items)
            
            # 去重
            self.cold_start_items = list(set(self.cold_start_items))
            logger.info(f"准备冷启动题目列表: {len(self.cold_start_items)}道题目")
            
    def get_recommendations(self, user_id: int, row_length: int = 10) -> List[int]:
        """
        获取用户推荐题目
        
        Args:
            user_id: 用户ID
            row_length: 要返回的题目数量
            
        Returns:
            List[int]: 推荐题目ID列表
        """
        with model_lock:
            try:
                # 检查模型是否已加载
                if not self.is_loaded or self.model is None:
                    logger.warning("模型未加载，尝试加载默认模型")
                    if not self.load_model():
                        logger.error("无法加载模型，使用冷启动策略")
                        return self._cold_start_recommendation(row_length)
                
                # 检查是否有用户历史数据
                user_history = self._get_user_history(user_id)
                
                # 如果用户没有历史数据，使用冷启动策略
                if user_history is None or len(user_history) == 0:
                    logger.info(f"用户 {user_id} 没有历史数据，使用冷启动策略")
                    return self._cold_start_recommendation(row_length)
                
                # 使用模型生成推荐
                logger.info(f"为用户 {user_id} 生成 {row_length} 条推荐")
                recommendations = self._generate_recommendations(user_id, user_history, row_length)
                
                # 如果推荐不足，用冷启动策略补充
                if len(recommendations) < row_length:
                    cold_recommendations = self._cold_start_recommendation(row_length - len(recommendations))
                    # 确保不重复
                    cold_recommendations = [r for r in cold_recommendations if r not in recommendations]
                    recommendations.extend(cold_recommendations)
                
                return recommendations[:row_length]
                
            except Exception as e:
                logger.error(f"获取推荐失败: {str(e)}")
                return self._cold_start_recommendation(row_length)
    
    def _get_user_history(self, user_id: int) -> List[Dict[str, Any]]:
        """
        获取用户历史数据
        
        Args:
            user_id: 用户ID
            
        Returns:
            List[Dict]: 用户历史记录
        """
        try:
            # 检查是否已缓存用户数据
            if user_id in self.user_data:
                return self.user_data[user_id]
            
            # 从数据文件加载用户答题记录
            attempts_path = os.path.join("data", "quiz_attempts.csv")
            if not os.path.exists(attempts_path):
                logger.warning(f"答题记录文件不存在: {attempts_path}")
                return []
            
            # 读取答题记录
            attempts_df = pd.read_csv(attempts_path, encoding='utf-8')
            user_attempts = attempts_df[attempts_df['user_id'] == user_id]
            
            if user_attempts.empty:
                logger.info(f"用户 {user_id} 没有答题记录")
                return []
            
            # 转换为字典列表
            history = user_attempts.to_dict('records')
            
            # 缓存用户数据
            self.user_data[user_id] = history
            
            return history
            
        except Exception as e:
            logger.error(f"获取用户历史记录失败: {str(e)}")
            return []
    
    def _generate_recommendations(self, user_id: int, user_history: List[Dict[str, Any]], 
                                 count: int) -> List[int]:
        """
        使用模型生成推荐
        
        Args:
            user_id: 用户ID
            user_history: 用户历史数据
            count: 推荐数量
            
        Returns:
            List[int]: 推荐题目ID列表
        """
        # 这里实现实际的推荐逻辑，使用训练好的模型
        # 实际中应该使用模型对用户特征和题目特征进行预测
        
        # 如果是协同过滤模型
        if hasattr(self.model, 'predict'):
            # 从用户历史中提取特征
            user_features = self._extract_user_features(user_id, user_history)
            
            # 获取候选题目列表
            candidate_questions = self._get_candidate_questions(user_history)
            
            if len(candidate_questions) == 0:
                logger.warning("没有候选题目，使用冷启动策略")
                return self._cold_start_recommendation(count)
            
            # 为每个候选题目生成预测得分
            scores = []
            for question_id in candidate_questions:
                # 生成题目特征
                item_features = self._extract_question_features(question_id)
                
                # 合并用户和题目特征
                features = {**user_features, **item_features}
                
                # 转换为模型需要的格式
                feature_array = self._prepare_features_for_model(features)
                
                # 预测分数
                try:
                    score = self.model.predict(feature_array)[0]
                    scores.append((question_id, score))
                except Exception as e:
                    logger.error(f"预测分数失败: {str(e)}")
            
            # 排序并返回前count个推荐
            scores.sort(key=lambda x: x[1], reverse=True)
            recommendations = [question_id for question_id, _ in scores[:count]]
            
            return recommendations
        else:
            # 如果没有合适的模型，使用冷启动策略
            logger.warning("模型不支持预测功能，使用冷启动策略")
            return self._cold_start_recommendation(count)
    
    def _extract_user_features(self, user_id: int, user_history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        从用户历史中提取特征
        
        Args:
            user_id: 用户ID
            user_history: 用户历史数据
            
        Returns:
            Dict: 用户特征
        """
        # 从用户历史中提取特征
        total_attempts = len(user_history)
        correct_answers = sum(1 for record in user_history if record.get('is_correct', 0) == 1)
        accuracy = correct_answers / total_attempts if total_attempts > 0 else 0
        
        # 计算平均响应时间
        response_times = [record.get('response_time', 0) for record in user_history if record.get('response_time', 0) > 0]
        avg_response_time = sum(response_times) / len(response_times) if response_times else 0
        
        # 统计不同类型的题目
        question_types = {}
        difficulty_levels = {}
        categories = {}
        
        if self.question_data is not None:
            # 获取用户做过的题目ID
            question_ids = [record.get('question_id') for record in user_history]
            
            # 找到这些题目的详细信息
            user_questions = self.question_data[self.question_data['question_id'].isin(question_ids)]
            
            # 统计题目类型
            if 'question_type' in user_questions.columns:
                for qtype in user_questions['question_type'].unique():
                    question_types[qtype] = len(user_questions[user_questions['question_type'] == qtype])
            
            # 统计难度级别
            if 'difficulty_level' in user_questions.columns:
                for level in user_questions['difficulty_level'].unique():
                    difficulty_levels[level] = len(user_questions[user_questions['difficulty_level'] == level])
            
            # 统计科目分类
            if 'category' in user_questions.columns:
                for category in user_questions['category'].unique():
                    categories[category] = len(user_questions[user_questions['category'] == category])
        
        # 返回用户特征
        return {
            'user_id': user_id,
            'total_attempts': total_attempts,
            'correct_answers': correct_answers,
            'accuracy': accuracy,
            'avg_response_time': avg_response_time,
            'question_types': question_types,
            'difficulty_levels': difficulty_levels,
            'categories': categories
        }
    
    def _extract_question_features(self, question_id: int) -> Dict[str, Any]:
        """
        提取题目特征
        
        Args:
            question_id: 题目ID
            
        Returns:
            Dict: 题目特征
        """
        # 默认特征
        features = {
            'question_id': question_id,
            'question_type': '',
            'difficulty_level': '',
            'category': ''
        }
        
        # 如果有题目数据，补充详细特征
        if self.question_data is not None:
            question = self.question_data[self.question_data['question_id'] == question_id]
            if not question.empty:
                features['question_type'] = question['question_type'].iloc[0] if 'question_type' in question.columns else ''
                features['difficulty_level'] = question['difficulty_level'].iloc[0] if 'difficulty_level' in question.columns else ''
                features['category'] = question['category'].iloc[0] if 'category' in question.columns else ''
        
        return features
    
    def _prepare_features_for_model(self, features: Dict[str, Any]) -> np.ndarray:
        """
        准备模型所需的特征数组
        
        Args:
            features: 特征字典
            
        Returns:
            np.ndarray: 模型输入特征
        """
        # 根据模型的预期输入格式准备特征
        # 这部分需要根据实际训练的模型调整
        
        # 示例：将特征转换为数组
        # 这里假设模型期望一个包含用户特征和题目特征的NumPy数组
        feature_array = np.array([
            features.get('user_id', 0),
            features.get('total_attempts', 0),
            features.get('accuracy', 0),
            features.get('avg_response_time', 0),
            features.get('question_id', 0),
            # 将分类特征转换为数值
            1 if features.get('question_type') == '单选' else
            2 if features.get('question_type') == '多选' else
            3 if features.get('question_type') == '判断' else
            4 if features.get('question_type') == '填空' else 0,
            
            1 if features.get('difficulty_level') == '简单' else
            2 if features.get('difficulty_level') == '中等' else
            3 if features.get('difficulty_level') == '困难' else 0
        ]).reshape(1, -1)
        
        return feature_array
    
    def _get_candidate_questions(self, user_history: List[Dict[str, Any]]) -> List[int]:
        """
        获取候选题目列表，排除用户已经做过的题目
        
        Args:
            user_history: 用户历史数据
            
        Returns:
            List[int]: 候选题目ID列表
        """
        if self.question_data is None:
            return self.cold_start_items
        
        # 获取用户做过的题目ID
        done_questions = set(record.get('question_id') for record in user_history)
        
        # 从所有题目中筛选出用户未做过的题目
        all_questions = set(self.question_data['question_id'].tolist())
        candidates = list(all_questions - done_questions)
        
        # 如果没有足够的候选题目，返回冷启动列表
        if len(candidates) == 0:
            return self.cold_start_items
        
        return candidates
    
    def _cold_start_recommendation(self, count: int) -> List[int]:
        """
        冷启动推荐策略，用于新用户或模型失效时
        
        Args:
            count: 推荐数量
            
        Returns:
            List[int]: 推荐题目ID列表
        """
        # 如果已有冷启动题目列表，从中选择
        if self.cold_start_items:
            # 如果冷启动题目不足，先重新准备
            if len(self.cold_start_items) < count:
                self._prepare_cold_start_items()
            
            # 随机选择指定数量的题目
            if len(self.cold_start_items) >= count:
                return random.sample(self.cold_start_items, count)
            else:
                return self.cold_start_items.copy()
        
        # 如果没有冷启动题目列表，但有题目数据，从中选择
        if self.question_data is not None and not self.question_data.empty:
            # 优先选择简单难度的题目
            simple_questions = self.question_data[self.question_data['difficulty_level'] == '简单']
            
            if not simple_questions.empty and len(simple_questions) >= count:
                return simple_questions['question_id'].sample(count).tolist()
            else:
                return self.question_data['question_id'].sample(min(count, len(self.question_data))).tolist()
        
        # 如果没有题目数据，返回1到count的列表（作为虚拟题目ID）
        logger.warning("没有可用的题目数据，返回虚拟题目ID")
        return list(range(1, count + 1))


# 创建全局用户推荐器实例
recommender = UserRecommender()

def load_model(model_name: str = "recommender_model.pkl") -> bool:
    """
    加载推荐模型
    
    Args:
        model_name: 模型文件名
        
    Returns:
        bool: 加载是否成功
    """
    return recommender.load_model(model_name)

def get_recommendations(user_id: int, row_length: int = 10) -> List[int]:
    """
    获取用户推荐题目
    
    Args:
        user_id: 用户ID
        row_length: 要返回的题目数量
        
    Returns:
        List[int]: 推荐题目ID列表
    """
    return recommender.get_recommendations(user_id, row_length)


if __name__ == "__main__":
    # 测试代码
    success = load_model()
    print(f"模型加载{'成功' if success else '失败'}")
    
    # 获取推荐
    user_ids = [1, 2, 3, 100]
    for user_id in user_ids:
        recommendations = get_recommendations(user_id, 5)
        print(f"用户 {user_id} 的推荐题目: {recommendations}") 