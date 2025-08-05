"""
数据生成器模块
生成随机测试数据用于模型训练
包括用户数据、题目数据、答题记录和答题历史
"""

import os
import pandas as pd
import numpy as np
import json
import random
from datetime import datetime, timedelta
import logging
from typing import List, Dict, Union, Tuple, Optional
import threading

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join("data", "generator.log"), encoding='utf-8', mode='a'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("data_generator")

# 线程锁，防止并发写入问题
file_lock = threading.RLock()

class DataGenerator:
    """随机数据生成器类"""
    
    # 题目类型列表
    QUESTION_TYPES = ['单选', '多选', '判断', '填空']
    
    # 难度级别列表
    DIFFICULTY_LEVELS = ['简单', '中等', '困难']
    
    # 科目分类列表
    CATEGORIES = [
        'WPS_Office', '计算机基础', 'Photoshop', '网络安全', 'MS_Office',
        'C语言设计', 'Access', 'C++语言', 'Java语言', 'MySQL', 'OpenGauss',
        'Python语言', 'Web设计', '网络技术', '数据库技术', '信息安全技术',
        '嵌入式系统开发技术', 'Linux应用与开发技术', '网络工程师', '数据库工程师',
        '信息安全工程师', '嵌入式系统开发工程师', 'Linux应用与开发工程师'
    ]
    
    # 答题类型列表
    QUIZ_TYPES = ['在线挑战', '离线练习', '每日一练', '考点练习']
    
    # 答题模式列表
    QUIZ_MODES = ['练习', 'PK']
    
    # 完成状态列表
    COMPLETION_STATUS = ['完成', '未完成']
    
    # 信心程度列表
    CONFIDENCE_LEVELS = ['低', '中', '高']
    
    # 性别列表
    GENDERS = ['男', '女', '其他']
    
    # 教育水平列表
    EDUCATION_LEVELS = ['初中', '高中', '专科', '本科', '硕士', '博士']
    
    # 职业列表
    OCCUPATIONS = ['学生', '教师', '程序员', '工程师', '设计师', '销售', '市场', '管理', '其他']
    
    def __init__(self, data_dir: str = "data"):
        """
        初始化数据生成器
        
        Args:
            data_dir: 数据存储目录
        """
        self.data_dir = data_dir
        self.ensure_data_dir()
        
        # 内部计数器，确保ID递增
        self._user_id_counter = 1
        self._question_id_counter = 1
        self._attempt_id_counter = 1
        self._history_id_counter = 1
        
        # 已生成的数据缓存
        self.users_data = []
        self.questions_data = []
        self.attempts_data = []
        self.history_data = []
        
        # 进度追踪
        self.generation_progress = {
            'users': 0,
            'questions': 0,
            'attempts': 0,
            'history': 0,
            'total': 0
        }
    
    def ensure_data_dir(self) -> None:
        """确保数据目录存在"""
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
            logger.info(f"创建数据目录: {self.data_dir}")
    
    def _update_progress(self, dataset_name: str, current: int, total: int) -> None:
        """
        更新进度信息
        
        Args:
            dataset_name: 数据集名称
            current: 当前进度
            total: 总数量
        """
        self.generation_progress[dataset_name] = int(current / total * 100)
        self.generation_progress['total'] = sum(
            v for k, v in self.generation_progress.items() if k != 'total'
        ) // 4
    
    def generate_users(self, count: int = 100) -> pd.DataFrame:
        """
        生成用户数据
        
        Args:
            count: 要生成的用户数量
            
        Returns:
            pd.DataFrame: 生成的用户数据
        """
        logger.info(f"开始生成 {count} 个用户数据")
        
        users = []
        for i in range(count):
            # 更新进度
            if i % max(1, count // 10) == 0:
                self._update_progress('users', i, count)
            
            # 注册日期和最后登录日期
            registration_date = datetime.now() - timedelta(
                days=random.randint(1, 1000)
            )
            last_login = registration_date + timedelta(
                days=random.randint(0, (datetime.now() - registration_date).days)
            )
            
            # 生成随机出生日期 (18-60岁)
            birth_year = datetime.now().year - random.randint(18, 60)
            birth_month = random.randint(1, 12)
            birth_day = random.randint(1, 28)  # 简化处理，避免日期问题
            birthdate = datetime(birth_year, birth_month, birth_day)
            
            # 生成用户数据
            user = {
                'user_id': self._user_id_counter,
                'username': f"user_{self._user_id_counter}",
                'password': f"password_hash_{self._user_id_counter}",  # 实际应用中应该是哈希值
                'email': f"user_{self._user_id_counter}@example.com",
                'phone': f"1{random.randint(3, 9)}{random.choices('0123456789', k=9)}",
                'registration_date': registration_date.strftime('%Y-%m-%d %H:%M:%S'),
                'last_login': last_login.strftime('%Y-%m-%d %H:%M:%S'),
                'vip_status': random.choices([0, 1], weights=[0.8, 0.2])[0],
                'user_level': random.randint(1, 100),
                'profile_picture': f"avatar_{self._user_id_counter}.jpg",
                'profile_picture_url': f"https://example.com/avatars/user_{self._user_id_counter}.jpg",
                'bio': f"这是用户 {self._user_id_counter} 的个人简介",
                'address': f"示例地址 {self._user_id_counter}",
                'gender': random.choice(self.GENDERS),
                'birthdate': birthdate.strftime('%Y-%m-%d'),
                'occupation': random.choice(self.OCCUPATIONS),
                'education_level': random.choice(self.EDUCATION_LEVELS)
            }
            
            users.append(user)
            self._user_id_counter += 1
        
        # 更新进度为100%
        self._update_progress('users', count, count)
        
        # 将数据转换为DataFrame
        users_df = pd.DataFrame(users)
        
        # 缓存生成的数据
        self.users_data = users
        
        logger.info(f"已成功生成 {len(users)} 个用户数据")
        return users_df
    
    def generate_questions(self, count: int = 1000) -> pd.DataFrame:
        """
        生成题目数据
        
        Args:
            count: 要生成的题目数量
            
        Returns:
            pd.DataFrame: 生成的题目数据
        """
        logger.info(f"开始生成 {count} 道题目数据")
        
        questions = []
        for i in range(count):
            # 更新进度
            if i % max(1, count // 10) == 0:
                self._update_progress('questions', i, count)
            
            # 选择题目类型和生成选项
            question_type = random.choice(self.QUESTION_TYPES)
            options = None
            answer = None
            
            if question_type == '单选':
                # 生成选项A-D
                options = {
                    'A': f"选项A_{self._question_id_counter}",
                    'B': f"选项B_{self._question_id_counter}",
                    'C': f"选项C_{self._question_id_counter}",
                    'D': f"选项D_{self._question_id_counter}"
                }
                answer = random.choice(list(options.keys()))
            
            elif question_type == '多选':
                # 生成选项A-F
                options = {
                    'A': f"选项A_{self._question_id_counter}",
                    'B': f"选项B_{self._question_id_counter}",
                    'C': f"选项C_{self._question_id_counter}",
                    'D': f"选项D_{self._question_id_counter}",
                    'E': f"选项E_{self._question_id_counter}",
                    'F': f"选项F_{self._question_id_counter}"
                }
                # 随机选择2-4个选项作为答案
                num_answers = random.randint(2, min(4, len(options)))
                answer = ','.join(random.sample(list(options.keys()), num_answers))
            
            elif question_type == '判断':
                options = {'T': '正确', 'F': '错误'}
                answer = random.choice(['T', 'F'])
            
            elif question_type == '填空':
                options = None
                answer = f"答案_{self._question_id_counter}"
            
            # 选择类别和难度
            category = random.choice(self.CATEGORIES)
            difficulty = random.choice(self.DIFFICULTY_LEVELS)
            
            # 创建和审核日期
            created_at = datetime.now() - timedelta(days=random.randint(1, 365))
            reviewed = random.choice([0, 1])
            
            reviewed_by = None
            review_time = None
            if reviewed == 1:
                reviewed_by = random.randint(1, max(1, self._user_id_counter - 1))
                review_time = created_at + timedelta(days=random.randint(1, 30))
            
            # 生成标签
            tag_count = random.randint(1, 5)
            tags = [f"标签{j}" for j in range(1, tag_count + 1)]
            
            # 生成题目数据
            question = {
                'question_id': self._question_id_counter,
                'question_type': question_type,
                'question_text': f"这是第 {self._question_id_counter} 道题目，类型为{question_type}，类别为{category}，难度为{difficulty}",
                'options': json.dumps(options, ensure_ascii=False) if options else None,
                'answer': answer,
                'explanation': f"这是第 {self._question_id_counter} 道题目的解析",
                'difficulty_level': difficulty,
                'category': category,
                'created_at': created_at.strftime('%Y-%m-%d %H:%M:%S'),
                'updated_at': created_at.strftime('%Y-%m-%d %H:%M:%S'),  # 简化，创建时间和更新时间相同
                'active': 1,  # 默认激活
                'reviewed': reviewed,
                'reviewed_by': reviewed_by,
                'review_time': review_time.strftime('%Y-%m-%d %H:%M:%S') if review_time else None,
                'tags': json.dumps(tags, ensure_ascii=False)
            }
            
            questions.append(question)
            self._question_id_counter += 1
        
        # 更新进度为100%
        self._update_progress('questions', count, count)
        
        # 将数据转换为DataFrame
        questions_df = pd.DataFrame(questions)
        
        # 缓存生成的数据
        self.questions_data = questions
        
        logger.info(f"已成功生成 {len(questions)} 道题目数据")
        return questions_df
    
    def generate_quiz_attempts(self, count: int = 10000) -> pd.DataFrame:
        """
        生成答题记录数据
        
        Args:
            count: 要生成的答题记录数量
            
        Returns:
            pd.DataFrame: 生成的答题记录数据
        """
        logger.info(f"开始生成 {count} 条答题记录数据")
        
        # 确保有足够的用户和题目数据
        if not self.users_data:
            logger.warning("没有用户数据，生成少量用户数据用于测试")
            self.generate_users(100)
        
        if not self.questions_data:
            logger.warning("没有题目数据，生成少量题目数据用于测试")
            self.generate_questions(500)
        
        attempts = []
        for i in range(count):
            # 更新进度
            if i % max(1, count // 10) == 0:
                self._update_progress('attempts', i, count)
            
            # 随机选择用户和题目
            user_id = random.randint(1, self._user_id_counter - 1)
            question_id = random.randint(1, self._question_id_counter - 1)
            
            # 随机生成答题时间（过去一年内）
            attempt_time = datetime.now() - timedelta(
                days=random.randint(0, 365),
                hours=random.randint(0, 23),
                minutes=random.randint(0, 59)
            )
            
            # 随机生成答题类型
            quiz_type = random.choice(self.QUIZ_TYPES)
            
            # 随机生成是否答对
            is_correct = random.choices([0, 1], weights=[0.4, 0.6])[0]
            
            # 生成对战对手ID (仅在线挑战)
            opponent_id = None
            if quiz_type == '在线挑战':
                opponent_candidates = [uid for uid in range(1, self._user_id_counter)
                                      if uid != user_id]
                if opponent_candidates:
                    opponent_id = random.choice(opponent_candidates)
            
            # 随机生成答题时间（5-300秒）
            response_time = random.randint(5, 300)
            
            # 随机生成使用的提示次数
            hints_used = random.randint(0, 3)
            
            # 随机生成信心程度
            confidence_level = random.choice(self.CONFIDENCE_LEVELS)
            
            # 生成答题记录
            attempt = {
                'attempt_id': self._attempt_id_counter,
                'user_id': user_id,
                'question_id': question_id,
                'is_correct': is_correct,
                'attempt_time': attempt_time.strftime('%Y-%m-%d %H:%M:%S'),
                'quiz_type': quiz_type,
                'opponent_id': opponent_id,
                'response_time': response_time,
                'response_text': f"用户回答内容_{self._attempt_id_counter}",
                'hints_used': hints_used,
                'confidence_level': confidence_level
            }
            
            attempts.append(attempt)
            self._attempt_id_counter += 1
        
        # 更新进度为100%
        self._update_progress('attempts', count, count)
        
        # 将数据转换为DataFrame
        attempts_df = pd.DataFrame(attempts)
        
        # 缓存生成的数据
        self.attempts_data = attempts
        
        logger.info(f"已成功生成 {len(attempts)} 条答题记录数据")
        return attempts_df
    
    def generate_quiz_history(self, count: int = 2000) -> pd.DataFrame:
        """
        生成答题历史数据
        
        Args:
            count: 要生成的答题历史数量
            
        Returns:
            pd.DataFrame: 生成的答题历史数据
        """
        logger.info(f"开始生成 {count} 条答题历史数据")
        
        # 确保有足够的用户数据
        if not self.users_data:
            logger.warning("没有用户数据，生成少量用户数据用于测试")
            self.generate_users(100)
        
        histories = []
        for i in range(count):
            # 更新进度
            if i % max(1, count // 10) == 0:
                self._update_progress('history', i, count)
            
            # 随机选择用户
            user_id = random.randint(1, self._user_id_counter - 1)
            
            # 随机生成开始答题时间（过去一年内）
            start_time = datetime.now() - timedelta(
                days=random.randint(0, 365),
                hours=random.randint(0, 23),
                minutes=random.randint(0, 59)
            )
            
            # 随机生成答题时长（5-60分钟，转换为秒）
            time_spent = random.randint(5 * 60, 60 * 60)
            
            # 计算结束答题时间
            end_time = start_time + timedelta(seconds=time_spent)
            
            # 随机生成题目数量（5-50题）
            total_questions = random.randint(5, 50)
            
            # 随机生成正确答题数量
            correct_answers = random.randint(0, total_questions)
            
            # 计算准确率
            accuracy = (correct_answers / total_questions) * 100 if total_questions > 0 else 0
            
            # 计算得分 (满分100)
            score = accuracy
            
            # 随机生成答题类型和模式
            quiz_type = random.choice(self.QUIZ_TYPES)
            quiz_mode = random.choice(self.QUIZ_MODES)
            
            # 随机生成完成状态
            completion_status = random.choice(self.COMPLETION_STATUS)
            
            # 生成答题历史
            history = {
                'history_id': self._history_id_counter,
                'user_id': user_id,
                'quiz_type': quiz_type,
                'start_time': start_time.strftime('%Y-%m-%d %H:%M:%S'),
                'end_time': end_time.strftime('%Y-%m-%d %H:%M:%S'),
                'total_questions': total_questions,
                'correct_answers': correct_answers,
                'score': score,
                'quiz_mode': quiz_mode,
                'completion_status': completion_status,
                'time_spent': time_spent,
                'accuracy': accuracy
            }
            
            histories.append(history)
            self._history_id_counter += 1
        
        # 更新进度为100%
        self._update_progress('history', count, count)
        
        # 将数据转换为DataFrame
        history_df = pd.DataFrame(histories)
        
        # 缓存生成的数据
        self.history_data = histories
        
        logger.info(f"已成功生成 {len(histories)} 条答题历史数据")
        return history_df
    
    def save_data_to_csv(self, data: pd.DataFrame, filename: str) -> bool:
        """
        将数据保存为CSV文件
        
        Args:
            data: 要保存的数据
            filename: 文件名
            
        Returns:
            bool: 是否保存成功
        """
        try:
            filepath = os.path.join(self.data_dir, filename)
            
            with file_lock:
                # 确保目录存在
                self.ensure_data_dir()
                
                # 保存数据
                data.to_csv(filepath, index=False, encoding='utf-8')
                logger.info(f"数据已保存到 {filepath}")
                return True
                
        except Exception as e:
            logger.error(f"保存数据失败: {str(e)}")
            return False
    
    def generate_all_data(self, users_count: int = 500, 
                         questions_count: int = 2000, 
                         attempts_count: int = 20000, 
                         history_count: int = 3000) -> Dict[str, pd.DataFrame]:
        """
        生成所有测试数据
        
        Args:
            users_count: 用户数量
            questions_count: 题目数量
            attempts_count: 答题记录数量
            history_count: 答题历史数量
            
        Returns:
            Dict[str, pd.DataFrame]: 包含所有生成数据的字典
        """
        logger.info("开始生成测试数据")
        
        # 重置进度
        self.generation_progress = {
            'users': 0,
            'questions': 0,
            'attempts': 0,
            'history': 0,
            'total': 0
        }
        
        # 生成数据
        users_df = self.generate_users(users_count)
        questions_df = self.generate_questions(questions_count)
        attempts_df = self.generate_quiz_attempts(attempts_count)
        history_df = self.generate_quiz_history(history_count)
        
        # 保存数据
        self.save_data_to_csv(users_df, "users.csv")
        self.save_data_to_csv(questions_df, "questions.csv")
        self.save_data_to_csv(attempts_df, "quiz_attempts.csv")
        self.save_data_to_csv(history_df, "quiz_history.csv")
        
        logger.info("所有测试数据生成完成")
        
        # 返回生成的数据
        return {
            'users': users_df,
            'questions': questions_df,
            'attempts': attempts_df,
            'history': history_df
        }
    
    def get_progress(self) -> Dict[str, int]:
        """
        获取数据生成进度
        
        Returns:
            Dict[str, int]: 包含各数据集生成进度的字典
        """
        return self.generation_progress.copy()


# 函数封装，便于外部调用
def generate_test_data(users_count: int = 500, 
                      questions_count: int = 2000, 
                      attempts_count: int = 20000, 
                      history_count: int = 3000,
                      data_dir: str = "data") -> Dict[str, pd.DataFrame]:
    """
    生成测试数据
    
    Args:
        users_count: 用户数量
        questions_count: 题目数量
        attempts_count: 答题记录数量
        history_count: 答题历史数量
        data_dir: 数据存储目录
        
    Returns:
        Dict[str, pd.DataFrame]: 包含所有生成数据的字典
    """
    generator = DataGenerator(data_dir)
    return generator.generate_all_data(
        users_count, questions_count, attempts_count, history_count
    )

# 获取生成进度的全局实例
_generator_instance = None

def get_generator_instance(data_dir: str = "data") -> DataGenerator:
    """
    获取数据生成器实例
    
    Args:
        data_dir: 数据存储目录
        
    Returns:
        DataGenerator: 数据生成器实例
    """
    global _generator_instance
    if _generator_instance is None:
        _generator_instance = DataGenerator(data_dir)
    return _generator_instance

def get_generation_progress() -> Dict[str, int]:
    """
    获取数据生成进度
    
    Returns:
        Dict[str, int]: 包含各数据集生成进度的字典
    """
    generator = get_generator_instance()
    return generator.get_progress()


if __name__ == "__main__":
    # 测试代码
    generator = DataGenerator()
    
    # 生成少量测试数据
    data = generator.generate_all_data(
        users_count=50,
        questions_count=200,
        attempts_count=1000,
        history_count=300
    )
    
    # 打印数据统计信息
    for name, df in data.items():
        print(f"{name} 数据: {len(df)} 条记录") 