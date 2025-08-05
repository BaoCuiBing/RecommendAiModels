"""
线程池管理工具
用于优化多线程处理，提供统一的任务调度和管理
"""

import threading
import queue
import time
import logging
from typing import Callable, Dict, List, Any, Tuple, Optional, Union
from concurrent.futures import ThreadPoolExecutor, Future, as_completed
import traceback

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("thread_pool")

class Task:
    """任务类，封装一个待执行的任务"""
    
    def __init__(self, func: Callable, *args, **kwargs):
        """
        初始化任务
        
        Args:
            func: 任务函数
            *args: 位置参数
            **kwargs: 关键字参数
        """
        self.func = func
        self.args = args
        self.kwargs = kwargs
        self.result = None
        self.error = None
        self.completed = False
        self.future = None
        self.start_time = None
        self.end_time = None
    
    def execute(self) -> Any:
        """
        执行任务
        
        Returns:
            Any: 任务执行结果
        """
        try:
            self.start_time = time.time()
            self.result = self.func(*self.args, **self.kwargs)
            self.completed = True
            return self.result
        except Exception as e:
            self.error = e
            logger.error(f"任务执行失败: {str(e)}")
            logger.debug(traceback.format_exc())
            return None
        finally:
            self.end_time = time.time()
    
    def get_execution_time(self) -> float:
        """
        获取任务执行时间（秒）
        
        Returns:
            float: 执行时间，未执行则返回0
        """
        if self.start_time is None or self.end_time is None:
            return 0
        return self.end_time - self.start_time


class ThreadPool:
    """线程池类，管理线程和任务执行"""
    
    def __init__(self, max_workers: int = None, thread_name_prefix: str = "ThreadPool"):
        """
        初始化线程池
        
        Args:
            max_workers: 最大工作线程数，None表示使用CPU核心数
            thread_name_prefix: 线程名称前缀
        """
        self.executor = ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix=thread_name_prefix
        )
        self.tasks = {}  # 任务字典，{task_id: Task}
        self.futures = {}  # Future字典，{future: task_id}
        self.task_id_counter = 0  # 任务ID计数器
        self.lock = threading.RLock()  # 线程锁，保护任务字典
    
    def submit(self, func: Callable, *args, **kwargs) -> int:
        """
        提交任务到线程池
        
        Args:
            func: 任务函数
            *args: 位置参数
            **kwargs: 关键字参数
            
        Returns:
            int: 任务ID
        """
        with self.lock:
            # 创建任务
            task = Task(func, *args, **kwargs)
            
            # 分配任务ID
            task_id = self.task_id_counter
            self.task_id_counter += 1
            
            # 提交任务到线程池
            future = self.executor.submit(task.execute)
            task.future = future
            
            # 记录任务和Future
            self.tasks[task_id] = task
            self.futures[future] = task_id
            
            return task_id
    
    def map(self, func: Callable, iterable: List[Any]) -> List[int]:
        """
        批量提交任务
        
        Args:
            func: 任务函数
            iterable: 参数列表
            
        Returns:
            List[int]: 任务ID列表
        """
        return [self.submit(func, item) for item in iterable]
    
    def get_result(self, task_id: int, timeout: Optional[float] = None) -> Tuple[bool, Any]:
        """
        获取任务结果
        
        Args:
            task_id: 任务ID
            timeout: 超时时间（秒），None表示一直等待
            
        Returns:
            Tuple[bool, Any]: (成功标志, 结果/错误)
        """
        with self.lock:
            if task_id not in self.tasks:
                return False, f"任务 {task_id} 不存在"
            
            task = self.tasks[task_id]
            
            if task.completed:
                if task.error:
                    return False, task.error
                return True, task.result
            
            # 如果任务未完成，等待其完成
            try:
                future = task.future
                result = future.result(timeout=timeout)
                return True, result
            except TimeoutError:
                return False, f"获取任务 {task_id} 结果超时"
            except Exception as e:
                return False, str(e)
    
    def wait_for_tasks(self, task_ids: List[int], timeout: Optional[float] = None) -> Dict[int, Tuple[bool, Any]]:
        """
        等待多个任务完成
        
        Args:
            task_ids: 任务ID列表
            timeout: 总超时时间（秒），None表示一直等待
            
        Returns:
            Dict[int, Tuple[bool, Any]]: {任务ID: (成功标志, 结果/错误)}
        """
        start_time = time.time()
        results = {}
        
        # 获取要等待的任务
        futures_to_wait = []
        with self.lock:
            for task_id in task_ids:
                if task_id in self.tasks:
                    task = self.tasks[task_id]
                    if not task.completed and task.future:
                        futures_to_wait.append(task.future)
                    else:
                        # 如果任务已完成，直接添加结果
                        if task.error:
                            results[task_id] = (False, task.error)
                        else:
                            results[task_id] = (True, task.result)
        
        # 等待未完成的任务
        if futures_to_wait:
            remaining_timeout = None
            if timeout is not None:
                elapsed = time.time() - start_time
                remaining_timeout = max(0, timeout - elapsed)
            
            # 等待完成
            for future in as_completed(futures_to_wait, timeout=remaining_timeout):
                with self.lock:
                    if future in self.futures:
                        task_id = self.futures[future]
                        task = self.tasks[task_id]
                        
                        if task.error:
                            results[task_id] = (False, task.error)
                        else:
                            results[task_id] = (True, task.result)
        
        # 添加未完成的任务结果
        with self.lock:
            for task_id in task_ids:
                if task_id not in results and task_id in self.tasks:
                    results[task_id] = (False, "任务未完成")
        
        return results
    
    def cancel_task(self, task_id: int) -> bool:
        """
        取消任务
        
        Args:
            task_id: 任务ID
            
        Returns:
            bool: 是否成功取消
        """
        with self.lock:
            if task_id not in self.tasks:
                return False
            
            task = self.tasks[task_id]
            if task.completed:
                return False  # 已完成的任务无法取消
            
            if task.future:
                return task.future.cancel()
            
            return False
    
    def cancel_all_tasks(self) -> int:
        """
        取消所有未完成的任务
        
        Returns:
            int: 成功取消的任务数量
        """
        cancelled_count = 0
        with self.lock:
            for task_id, task in self.tasks.items():
                if not task.completed and task.future:
                    if task.future.cancel():
                        cancelled_count += 1
        return cancelled_count
    
    def get_task_status(self, task_id: int) -> Dict[str, Any]:
        """
        获取任务状态
        
        Args:
            task_id: 任务ID
            
        Returns:
            Dict[str, Any]: 任务状态信息
        """
        with self.lock:
            if task_id not in self.tasks:
                return {'exists': False}
            
            task = self.tasks[task_id]
            future = task.future
            
            status = {
                'exists': True,
                'completed': task.completed,
                'running': future.running() if future else False,
                'cancelled': future.cancelled() if future else False,
                'error': str(task.error) if task.error else None,
                'execution_time': task.get_execution_time()
            }
            
            return status
    
    def get_all_task_status(self) -> Dict[int, Dict[str, Any]]:
        """
        获取所有任务的状态
        
        Returns:
            Dict[int, Dict[str, Any]]: {任务ID: 任务状态}
        """
        with self.lock:
            status = {}
            for task_id in self.tasks:
                status[task_id] = self.get_task_status(task_id)
            return status
    
    def clear_completed_tasks(self, keep_last_n: int = 10) -> int:
        """
        清理已完成的任务
        
        Args:
            keep_last_n: 保留最近完成的任务数量
            
        Returns:
            int: 清理的任务数量
        """
        with self.lock:
            # 获取已完成的任务
            completed_tasks = [
                (task_id, task) for task_id, task in self.tasks.items()
                if task.completed
            ]
            
            # 按完成时间排序
            completed_tasks.sort(key=lambda x: x[1].end_time or 0)
            
            # 计算要删除的任务
            to_remove = completed_tasks[:-keep_last_n] if keep_last_n > 0 else completed_tasks
            
            # 删除任务
            removed_count = 0
            for task_id, task in to_remove:
                del self.tasks[task_id]
                if task.future in self.futures:
                    del self.futures[task.future]
                removed_count += 1
            
            return removed_count
    
    def shutdown(self, wait: bool = True) -> None:
        """
        关闭线程池
        
        Args:
            wait: 是否等待所有任务完成
        """
        self.executor.shutdown(wait=wait)


# 全局线程池
_global_thread_pool = None

def get_thread_pool(max_workers: int = None) -> ThreadPool:
    """
    获取全局线程池实例
    
    Args:
        max_workers: 最大工作线程数，None表示使用CPU核心数
            
    Returns:
        ThreadPool: 线程池实例
    """
    global _global_thread_pool
    if _global_thread_pool is None:
        _global_thread_pool = ThreadPool(max_workers=max_workers)
    return _global_thread_pool

def submit_task(func: Callable, *args, **kwargs) -> int:
    """
    提交任务到全局线程池
    
    Args:
        func: 任务函数
        *args: 位置参数
        **kwargs: 关键字参数
            
    Returns:
        int: 任务ID
    """
    pool = get_thread_pool()
    return pool.submit(func, *args, **kwargs)

def get_task_result(task_id: int, timeout: Optional[float] = None) -> Tuple[bool, Any]:
    """
    获取任务结果
    
    Args:
        task_id: 任务ID
        timeout: 超时时间（秒），None表示一直等待
            
    Returns:
        Tuple[bool, Any]: (成功标志, 结果/错误)
    """
    pool = get_thread_pool()
    return pool.get_result(task_id, timeout)

def wait_for_tasks(task_ids: List[int], timeout: Optional[float] = None) -> Dict[int, Tuple[bool, Any]]:
    """
    等待多个任务完成
    
    Args:
        task_ids: 任务ID列表
        timeout: 总超时时间（秒），None表示一直等待
            
    Returns:
        Dict[int, Tuple[bool, Any]]: {任务ID: (成功标志, 结果/错误)}
    """
    pool = get_thread_pool()
    return pool.wait_for_tasks(task_ids, timeout)

def cancel_task(task_id: int) -> bool:
    """
    取消任务
    
    Args:
        task_id: 任务ID
            
    Returns:
        bool: 是否成功取消
    """
    pool = get_thread_pool()
    return pool.cancel_task(task_id)

def get_task_status(task_id: int) -> Dict[str, Any]:
    """
    获取任务状态
    
    Args:
        task_id: 任务ID
            
    Returns:
        Dict[str, Any]: 任务状态信息
    """
    pool = get_thread_pool()
    return pool.get_task_status(task_id)

def shutdown_thread_pool(wait: bool = True) -> None:
    """
    关闭全局线程池
    
    Args:
        wait: 是否等待所有任务完成
    """
    global _global_thread_pool
    if _global_thread_pool is not None:
        _global_thread_pool.shutdown(wait=wait)
        _global_thread_pool = None


if __name__ == "__main__":
    # 测试代码
    def test_function(n: int) -> int:
        """测试函数，返回参数的平方"""
        time.sleep(1)  # 模拟耗时操作
        return n * n
    
    # 创建线程池
    pool = ThreadPool(max_workers=4)
    
    # 提交任务
    task_ids = []
    for i in range(10):
        task_id = pool.submit(test_function, i)
        task_ids.append(task_id)
        print(f"提交任务 {task_id}: test_function({i})")
    
    # 等待所有任务完成
    results = pool.wait_for_tasks(task_ids)
    
    # 打印结果
    for task_id, (success, result) in results.items():
        print(f"任务 {task_id}: {'成功' if success else '失败'}, 结果: {result}")
    
    # 关闭线程池
    pool.shutdown() 