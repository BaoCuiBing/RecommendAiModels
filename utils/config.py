"""
配置模块
存储全局配置参数
"""

import os
from datetime import timedelta
import logging
import json
from typing import Dict, Any, Optional

class AppConfig:
    """应用程序配置类"""
    
    # 默认配置
    DEFAULT_CONFIG = {
        # 路径配置
        'paths': {
            'data_dir': 'data',
            'models_dir': 'models',
            'logs_dir': 'logs',
            'temp_dir': 'temp'
        },
        
        # 性能配置
        'performance': {
            'n_jobs': -1,  # -1 表示使用所有可用CPU核心
            'memory_limit_mb': 0,  # 0 表示不限制内存
            'cache_timeout': 30,  # 分钟，多久未使用的缓存会被清理
            'enable_threading': True,  # 是否启用多线程
            'async_loading': True  # 是否启用异步加载
        },
        
        # 数据生成配置
        'data_generator': {
            'users_count': 500,
            'questions_count': 2000,
            'attempts_count': 20000,
            'history_count': 3000
        },
        
        # 模型训练配置
        'model_trainer': {
            'default_algorithm': 'rf',  # 默认算法：随机森林
            'enable_optimization': False,  # 是否启用超参数优化
            'test_size': 0.2,  # 测试集比例
            'random_state': 42  # 随机种子
        },
        
        # UI配置
        'ui': {
            'theme': 'default',  # 界面主题
            'font_size': 10,  # 基础字体大小
            'chart_dpi': 100,  # 图表DPI
            'line_width': 1.0,  # 线条宽度
            'window_width': 1200,  # 窗口宽度
            'window_height': 800,  # 窗口高度
            'refresh_interval': 1000  # 刷新间隔（毫秒）
        },
        
        # 日志配置
        'logging': {
            'level': 'INFO',
            'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            'file_enabled': True
        }
    }
    
    def __init__(self, config_file: str = 'config.json'):
        """
        初始化配置
        
        Args:
            config_file: 配置文件路径
        """
        self.config_file = config_file
        self.config = self.DEFAULT_CONFIG.copy()
        
        # 加载配置文件
        self.load_config()
        
        # 确保所有必要的目录存在
        self.ensure_dirs()
    
    def load_config(self) -> None:
        """加载配置文件"""
        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    file_config = json.load(f)
                
                # 递归更新配置
                self._update_config(self.config, file_config)
                logging.info(f"从{self.config_file}加载配置成功")
                
        except Exception as e:
            logging.warning(f"加载配置文件失败: {str(e)}，使用默认配置")
    
    def _update_config(self, target: Dict[str, Any], source: Dict[str, Any]) -> None:
        """
        递归更新配置
        
        Args:
            target: 目标配置字典
            source: 源配置字典
        """
        for key, value in source.items():
            if key in target and isinstance(value, dict) and isinstance(target[key], dict):
                self._update_config(target[key], value)
            else:
                target[key] = value
    
    def save_config(self) -> None:
        """保存配置到文件"""
        try:
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, indent=4, ensure_ascii=False)
            logging.info(f"配置已保存到 {self.config_file}")
            
        except Exception as e:
            logging.error(f"保存配置文件失败: {str(e)}")
    
    def ensure_dirs(self) -> None:
        """确保所有必要的目录存在"""
        for key, path in self.config['paths'].items():
            if not os.path.exists(path):
                os.makedirs(path)
                logging.info(f"创建目录: {path}")
    
    def get(self, section: str, key: str, default: Any = None) -> Any:
        """
        获取配置项
        
        Args:
            section: 配置节
            key: 配置键
            default: 默认值
            
        Returns:
            Any: 配置值或默认值
        """
        if section in self.config and key in self.config[section]:
            return self.config[section][key]
        return default
    
    def set(self, section: str, key: str, value: Any) -> None:
        """
        设置配置项
        
        Args:
            section: 配置节
            key: 配置键
            value: 配置值
        """
        if section not in self.config:
            self.config[section] = {}
        
        self.config[section][key] = value
    
    def get_path(self, key: str) -> str:
        """
        获取路径配置
        
        Args:
            key: 路径键名
            
        Returns:
            str: 路径值
        """
        return self.get('paths', key, '')
    
    def get_all(self) -> Dict[str, Any]:
        """
        获取所有配置
        
        Returns:
            Dict[str, Any]: 所有配置
        """
        return self.config.copy()


# 全局配置实例
app_config = AppConfig()

# 便捷函数
def get_config(section: str, key: str, default: Any = None) -> Any:
    """
    获取配置项
    
    Args:
        section: 配置节
        key: 配置键
        default: 默认值
        
    Returns:
        Any: 配置值或默认值
    """
    return app_config.get(section, key, default)

def set_config(section: str, key: str, value: Any) -> None:
    """
    设置配置项
    
    Args:
        section: 配置节
        key: 配置键
        value: 配置值
    """
    app_config.set(section, key, value)
    
def save_config() -> None:
    """保存配置到文件"""
    app_config.save_config()

def get_data_dir() -> str:
    """获取数据目录"""
    return app_config.get_path('data_dir')

def get_models_dir() -> str:
    """获取模型目录"""
    return app_config.get_path('models_dir')

def get_n_jobs() -> int:
    """获取并行任务数"""
    return app_config.get('performance', 'n_jobs')

def get_memory_limit() -> int:
    """获取内存限制"""
    return app_config.get('performance', 'memory_limit_mb')

def get_default_algorithm() -> str:
    """获取默认算法"""
    return app_config.get('model_trainer', 'default_algorithm')

def is_threading_enabled() -> bool:
    """是否启用多线程"""
    return app_config.get('performance', 'enable_threading')

def is_async_loading_enabled() -> bool:
    """是否启用异步加载"""
    return app_config.get('performance', 'async_loading')

def get_cache_timeout() -> int:
    """获取缓存超时时间（分钟）"""
    return app_config.get('performance', 'cache_timeout')


if __name__ == "__main__":
    # 测试代码
    print("配置测试:")
    print(f"数据目录: {get_data_dir()}")
    print(f"模型目录: {get_models_dir()}")
    print(f"并行任务数: {get_n_jobs()}")
    print(f"默认算法: {get_default_algorithm()}")
    
    # 修改配置
    set_config('ui', 'theme', 'dark')
    print(f"主题: {get_config('ui', 'theme')}")
    
    # 保存配置
    save_config() 