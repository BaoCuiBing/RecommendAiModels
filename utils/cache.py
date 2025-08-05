"""
缓存管理工具
优化Tkinter界面切换性能，管理页面缓存
"""

import time
import logging
import threading
from typing import Dict, Any, List, Tuple, Optional, Callable
from datetime import datetime, timedelta

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("cache_manager")

class CacheEntry:
    """缓存条目类，存储一个缓存项和其元数据"""
    
    def __init__(self, key: str, value: Any):
        """
        初始化缓存条目
        
        Args:
            key: 缓存键
            value: 缓存值
        """
        self.key = key
        self.value = value
        self.last_accessed = datetime.now()
        self.created = datetime.now()
        self.access_count = 0
    
    def access(self) -> None:
        """访问缓存，更新访问时间和计数"""
        self.last_accessed = datetime.now()
        self.access_count += 1
    
    def get_age(self) -> timedelta:
        """
        获取缓存条目的年龄
        
        Returns:
            timedelta: 年龄
        """
        return datetime.now() - self.created
    
    def get_idle_time(self) -> timedelta:
        """
        获取缓存条目的闲置时间
        
        Returns:
            timedelta: 闲置时间
        """
        return datetime.now() - self.last_accessed


class CacheManager:
    """缓存管理器类，管理内存中的缓存"""
    
    def __init__(self, max_size: int = 100, timeout_minutes: int = 30):
        """
        初始化缓存管理器
        
        Args:
            max_size: 最大缓存条目数
            timeout_minutes: 缓存超时时间（分钟）
        """
        self.cache = {}  # 缓存字典 {key: CacheEntry}
        self.max_size = max_size
        self.timeout = timedelta(minutes=timeout_minutes)
        self.lock = threading.RLock()  # 线程锁
        
        # 清理线程
        self.cleanup_thread = None
        self.cleanup_interval = 5  # 分钟
        self.running = False
        
        # 启动清理线程
        self.start_cleanup_thread()
    
    def start_cleanup_thread(self) -> None:
        """启动缓存清理线程"""
        if self.cleanup_thread is None or not self.cleanup_thread.is_alive():
            self.running = True
            self.cleanup_thread = threading.Thread(
                target=self._cleanup_worker,
                daemon=True
            )
            self.cleanup_thread.start()
            logger.info("缓存清理线程已启动")
    
    def stop_cleanup_thread(self) -> None:
        """停止缓存清理线程"""
        self.running = False
        if self.cleanup_thread and self.cleanup_thread.is_alive():
            self.cleanup_thread.join(timeout=1.0)
            logger.info("缓存清理线程已停止")
    
    def _cleanup_worker(self) -> None:
        """缓存清理工作线程"""
        while self.running:
            try:
                # 清理过期缓存
                self.cleanup()
                
                # 休眠一段时间
                time.sleep(self.cleanup_interval * 60)
            except Exception as e:
                logger.error(f"缓存清理线程出错: {str(e)}")
                time.sleep(60)  # 出错后休眠1分钟
    
    def set(self, key: str, value: Any) -> None:
        """
        设置缓存
        
        Args:
            key: 缓存键
            value: 缓存值
        """
        with self.lock:
            # 如果键已存在，更新值
            if key in self.cache:
                self.cache[key].value = value
                self.cache[key].access()
                return
            
            # 如果缓存已满，移除最久未访问的条目
            if len(self.cache) >= self.max_size:
                self._remove_least_recently_used()
            
            # 添加新缓存
            self.cache[key] = CacheEntry(key, value)
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        获取缓存值
        
        Args:
            key: 缓存键
            default: 默认值
            
        Returns:
            Any: 缓存值或默认值
        """
        with self.lock:
            if key in self.cache:
                entry = self.cache[key]
                entry.access()
                return entry.value
            return default
    
    def has(self, key: str) -> bool:
        """
        检查键是否存在
        
        Args:
            key: 缓存键
            
        Returns:
            bool: 是否存在
        """
        with self.lock:
            return key in self.cache
    
    def remove(self, key: str) -> bool:
        """
        移除缓存
        
        Args:
            key: 缓存键
            
        Returns:
            bool: 是否成功移除
        """
        with self.lock:
            if key in self.cache:
                del self.cache[key]
                return True
            return False
    
    def clear(self) -> None:
        """清空所有缓存"""
        with self.lock:
            self.cache.clear()
    
    def _remove_least_recently_used(self) -> None:
        """移除最久未访问的缓存条目"""
        if not self.cache:
            return
        
        # 找出最久未访问的条目
        oldest_key = None
        oldest_time = datetime.now()
        
        for key, entry in self.cache.items():
            if entry.last_accessed < oldest_time:
                oldest_time = entry.last_accessed
                oldest_key = key
        
        # 移除该条目
        if oldest_key:
            del self.cache[oldest_key]
    
    def cleanup(self) -> int:
        """
        清理过期缓存
        
        Returns:
            int: 清理的条目数量
        """
        with self.lock:
            # 当前时间
            now = datetime.now()
            
            # 找出过期的键
            expired_keys = [
                key for key, entry in self.cache.items()
                if now - entry.last_accessed > self.timeout
            ]
            
            # 移除过期条目
            for key in expired_keys:
                del self.cache[key]
            
            count = len(expired_keys)
            if count > 0:
                logger.info(f"已清理 {count} 个过期缓存条目")
            
            return count
    
    def get_stats(self) -> Dict[str, Any]:
        """
        获取缓存统计信息
        
        Returns:
            Dict[str, Any]: 缓存统计信息
        """
        with self.lock:
            stats = {
                'total_entries': len(self.cache),
                'max_size': self.max_size,
                'timeout_minutes': self.timeout.total_seconds() / 60,
                'entries': []
            }
            
            for key, entry in self.cache.items():
                stats['entries'].append({
                    'key': key,
                    'age_seconds': entry.get_age().total_seconds(),
                    'idle_seconds': entry.get_idle_time().total_seconds(),
                    'access_count': entry.access_count
                })
            
            return stats


# 全局缓存管理器实例
_cache_manager = None

def get_cache_manager(max_size: int = 100, timeout_minutes: int = 30) -> CacheManager:
    """
    获取全局缓存管理器实例
    
    Args:
        max_size: 最大缓存条目数
        timeout_minutes: 缓存超时时间（分钟）
        
    Returns:
        CacheManager: 缓存管理器实例
    """
    global _cache_manager
    if _cache_manager is None:
        _cache_manager = CacheManager(max_size, timeout_minutes)
    return _cache_manager

def cache_set(key: str, value: Any) -> None:
    """
    设置缓存
    
    Args:
        key: 缓存键
        value: 缓存值
    """
    manager = get_cache_manager()
    manager.set(key, value)

def cache_get(key: str, default: Any = None) -> Any:
    """
    获取缓存值
    
    Args:
        key: 缓存键
        default: 默认值
        
    Returns:
        Any: 缓存值或默认值
    """
    manager = get_cache_manager()
    return manager.get(key, default)

def cache_has(key: str) -> bool:
    """
    检查键是否存在
    
    Args:
        key: 缓存键
        
    Returns:
        bool: 是否存在
    """
    manager = get_cache_manager()
    return manager.has(key)

def cache_remove(key: str) -> bool:
    """
    移除缓存
    
    Args:
        key: 缓存键
        
    Returns:
        bool: 是否成功移除
    """
    manager = get_cache_manager()
    return manager.remove(key)

def cache_clear() -> None:
    """清空所有缓存"""
    manager = get_cache_manager()
    manager.clear()

def cache_cleanup() -> int:
    """
    清理过期缓存
    
    Returns:
        int: 清理的条目数量
    """
    manager = get_cache_manager()
    return manager.cleanup()

def get_cache_stats() -> Dict[str, Any]:
    """
    获取缓存统计信息
    
    Returns:
        Dict[str, Any]: 缓存统计信息
    """
    manager = get_cache_manager()
    return manager.get_stats()

def shutdown_cache_manager() -> None:
    """关闭缓存管理器"""
    global _cache_manager
    if _cache_manager is not None:
        _cache_manager.stop_cleanup_thread()
        _cache_manager = None


if __name__ == "__main__":
    # 测试代码
    cache = CacheManager(max_size=10, timeout_minutes=1)
    
    # 设置一些缓存
    for i in range(5):
        cache.set(f"key_{i}", f"value_{i}")
    
    # 读取缓存
    for i in range(3):
        value = cache.get(f"key_{i}")
        print(f"缓存 key_{i} = {value}")
    
    # 查看缓存统计
    stats = cache.get_stats()
    print(f"缓存统计: {stats['total_entries']} 个条目")
    
    # 测试过期清理
    print("等待缓存过期...")
    time.sleep(65)  # 等待超过1分钟
    cleaned = cache.cleanup()
    print(f"已清理 {cleaned} 个过期缓存条目")
    
    # 再次查看统计
    stats = cache.get_stats()
    print(f"清理后的缓存统计: {stats['total_entries']} 个条目")
    
    # 关闭缓存管理器
    cache.stop_cleanup_thread() 