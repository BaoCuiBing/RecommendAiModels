"""
智能题目推荐系统 - 主程序入口
集成数据生成、模型训练和题目推荐功能
"""

import os
import sys
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from typing import Dict, List, Any, Optional, Tuple, Union
import logging
import traceback
import threading
import time
import importlib
from datetime import datetime

# 确保模块能够被导入
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

# 导入配置
from utils.config import (
    get_config, set_config, save_config, 
    get_data_dir, get_models_dir
)

# 导入线程管理
from utils.thread_pool import (
    get_thread_pool, submit_task, wait_for_tasks,
    get_task_status, cancel_task, shutdown_thread_pool
)

# 导入缓存管理
from utils.cache import (
    get_cache_manager, cache_get, cache_set,
    cache_remove, cache_clear, shutdown_cache_manager
)

# 创建必要的目录
for dir_path in [get_data_dir(), get_models_dir(), 'ui', 'logs', 'temp']:
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

# 配置日志
log_dir = 'logs'
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(log_dir, "app.log"), encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("main")


class RecommenderApp(tk.Tk):
    """智能题目推荐系统应用主类"""
    
    def __init__(self, *args, **kwargs):
        """初始化应用"""
        super().__init__(*args, **kwargs)
        
        # 设置窗口属性
        self.title("Cookie答题用户推荐模型训练")
        self.geometry("1200x1000")
        self.minsize(800, 1000)
        
        # 设置窗口图标
        icon_path = os.path.join(current_dir, "data", "head.png")
        if os.path.exists(icon_path):
            try:
                icon_image = tk.PhotoImage(file=icon_path)
                self.iconphoto(True, icon_image)
                logger.info("成功加载应用程序图标")
            except Exception as e:
                logger.warning(f"加载应用程序图标失败: {str(e)}")
        else:
            logger.warning(f"图标文件不存在: {icon_path}")
        
        # 共享数据
        self.shared_data = {
            'app_name': 'Cookie答题用户推荐模型训练',
            'version': '1.0.0',
            'last_update': datetime.now().strftime('%Y-%m-%d'),
        }
        
        # 状态标志
        self.is_training = False
        self.is_generating_data = False

        # 创建主框架
        self.main_frame = ttk.Frame(self)
        self.main_frame.pack(fill=tk.BOTH, expand=True)
        
        # 创建顶部导航栏
        self._create_navbar()
        
        # 创建内容区域
        self.content_frame = ttk.Frame(self.main_frame)
        self.content_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # 创建状态栏
        self._create_statusbar()
        
        # 创建页面
        self.pages = {}
        self._setup_pages()
        
        # 显示训练页面作为初始页面
        self.show_page("TrainPage")
        
        # 绑定窗口关闭事件
        self.protocol("WM_DELETE_WINDOW", self._on_closing)
        
        # 设置初始状态
        self.update_status("就绪")
        
        logger.info("应用程序已启动")
    
    def _create_navbar(self):
        """创建顶部导航栏"""
        self.navbar = ttk.Frame(self.main_frame)
        self.navbar.pack(fill=tk.X, padx=5, pady=5)
        
        # 创建导航按钮
        self.nav_buttons = {
            "data": ttk.Button(self.navbar, text="数据生成", command=lambda: self.show_page("DataPage"), style="Nav.TButton"),
            "train": ttk.Button(self.navbar, text="模型训练", command=lambda: self.show_page("TrainPage"), style="Nav.TButton"),
            "monitor": ttk.Button(self.navbar, text="性能监控", command=lambda: self.show_page("MonitorPage"), style="Nav.TButton"),
            "recommend": ttk.Button(self.navbar, text="题目推荐", command=lambda: self.show_page("RecommendPage"), style="Nav.TButton"),
            "settings": ttk.Button(self.navbar, text="设置", command=lambda: self.show_page("SettingsPage"), style="Nav.TButton"),
        }
        
        # 布局导航按钮
        for button in self.nav_buttons.values():
            button.pack(side=tk.LEFT, padx=5)
        
        # 创建帮助菜单按钮
        help_button = ttk.Button(self.navbar, text="帮助", command=self.show_help, style="Nav.TButton")
        help_button.pack(side=tk.RIGHT, padx=5)
        
        # 创建导航分隔线
        ttk.Separator(self.main_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, padx=5, pady=2)
    
    def _create_statusbar(self):
        """创建状态栏"""
        self.statusbar = ttk.Frame(self.main_frame)
        self.statusbar.pack(fill=tk.X, side=tk.BOTTOM)
        
        ttk.Separator(self.statusbar, orient=tk.HORIZONTAL).pack(fill=tk.X)
        
        # 状态信息
        self.status_var = tk.StringVar(value="就绪")
        status_label = ttk.Label(self.statusbar, textvariable=self.status_var, style="Status.TLabel")
        status_label.pack(side=tk.LEFT, padx=10)
        
        # 进度条
        self.progress_var = tk.DoubleVar(value=0)
        self.progress_bar = ttk.Progressbar(
            self.statusbar, 
            variable=self.progress_var,
            length=200, 
            mode='determinate'
        )
        self.progress_bar.pack(side=tk.RIGHT, padx=10, pady=2)
        
        # 任务标签
        self.task_var = tk.StringVar(value="")
        task_label = ttk.Label(self.statusbar, textvariable=self.task_var, style="Status.TLabel")
        task_label.pack(side=tk.RIGHT, padx=5)
    
    def _setup_pages(self):
        """设置应用页面"""
        try:
            # 导入页面模块
            from ui.pages.data_page import DataPage
            from ui.pages.train_page import TrainPage
            from ui.pages.monitor_page import MonitorPage
            from ui.pages.recommend_page import RecommendPage
            from ui.pages.settings_page import SettingsPage
            
            # 创建页面实例
            self.pages = {
                "DataPage": DataPage(self.content_frame, self),
                "TrainPage": TrainPage(self.content_frame, self),
                "MonitorPage": MonitorPage(self.content_frame, self),
                "RecommendPage": RecommendPage(self.content_frame, self),
                "SettingsPage": SettingsPage(self.content_frame, self)
            }
            
            # 隐藏所有页面
            for page in self.pages.values():
                page.hide()
                
        except ImportError as e:
            error_msg = f"导入页面模块失败: {str(e)}"
            logger.error(error_msg)
            self._create_error_page(error_msg)
    
    def _create_error_page(self, error_msg: str):
        """创建错误页面"""
        from ui.pages.base_page import BasePage
        
        error_page = BasePage(self.content_frame, self)
        error_page.set_title("错误")
        
        frame = ttk.Frame(error_page)
        frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        ttk.Label(
            frame, 
            text="应用程序加载出错",
            font=("Helvetica", 14, "bold"),
            foreground="red"
        ).pack(pady=10)
        
        ttk.Label(
            frame,
            text=error_msg,
            wraplength=600
        ).pack(pady=10)
        
        ttk.Label(
            frame,
            text="请检查是否已正确安装所有依赖项，并确保相关模块存在。",
            wraplength=600
        ).pack(pady=10)
        
        ttk.Button(
            frame,
            text="退出应用",
            command=self.quit
        ).pack(pady=20)
        
        self.pages = {"ErrorPage": error_page}
    
    def show_page(self, page_name: str):
        """
        显示指定页面
        
        Args:
            page_name: 页面名称
        """
        if page_name not in self.pages:
            logger.warning(f"页面不存在: {page_name}")
            return
        
        # 隐藏当前页面
        for name, page in self.pages.items():
            if page.is_visible:
                page.hide()
        
        # 显示新页面
        self.pages[page_name].show()
        
        # 高亮当前选中的导航按钮
        self._highlight_nav_button(page_name)
        
        logger.debug(f"切换到页面: {page_name}")
    
    def _highlight_nav_button(self, page_name: str):
        """
        高亮当前选中的导航按钮
        
        Args:
            page_name: 页面名称
        """
        # 重置所有按钮样式
        style = ttk.Style()
        for name, button in self.nav_buttons.items():
            button.configure(style="Nav.TButton")
        
        # 设置选中按钮样式
        button_map = {
            "DataPage": "data",
            "TrainPage": "train",
            "MonitorPage": "monitor",
            "RecommendPage": "recommend",
            "SettingsPage": "settings"
        }
        
        if page_name in button_map and button_map[page_name] in self.nav_buttons:
            # 使用反转色突出显示当前按钮
            self.nav_buttons[button_map[page_name]].configure(style="Nav.TButton")
            # 可以在这里添加更多样式修改，例如改变按钮颜色或边框
    
    def update_status(self, status: str):
        """
        更新状态栏信息
        
        Args:
            status: 状态信息
        """
        self.status_var.set(status)
    
    def update_progress(self, value: float, task: str = ""):
        """
        更新进度条
        
        Args:
            value: 进度值 (0-100)
            task: 任务描述
        """
        self.progress_var.set(value)
        self.task_var.set(task)
    
    def show_message(self, message: str, message_type: str = "info"):
        """
        显示消息对话框
        
        Args:
            message: 消息内容
            message_type: 消息类型 (info, warning, error)
        """
        if message_type == "info":
            messagebox.showinfo("信息", message)
        elif message_type == "warning":
            messagebox.showwarning("警告", message)
        elif message_type == "error":
            messagebox.showerror("错误", message)
    
    def confirm_action(self, message: str, callback: callable = None):
        """
        确认操作对话框
        
        Args:
            message: 消息内容
            callback: 确认后的回调函数
        """
        result = messagebox.askyesno("确认", message)
        if result and callback:
            callback()
    
    def show_help(self):
        """显示帮助信息"""
        help_text = """
        Cookie答题用户推荐模型训练系统使用帮助
        
        1. 数据生成：用于生成训练和测试数据
        2. 模型训练：训练用户推荐模型
        3. 性能监控：监控模型训练性能
        4. 题目推荐：根据用户特征推荐题目
        5. 设置：配置系统参数
        """
        
        messagebox.showinfo("帮助", help_text)
    
    def execute_task(self, func, *args, show_progress=True, **kwargs):
        """
        在后台执行任务
        
        Args:
            func: 要执行的函数
            *args: 函数参数
            show_progress: 是否显示进度
            **kwargs: 函数关键字参数
            
        Returns:
            int: 任务ID
        """
        if show_progress:
            self.update_progress(0, "任务启动中...")
        
        task_id = submit_task(func, *args, **kwargs)
        
        if show_progress:
            self._monitor_task_progress(task_id)
        
        return task_id
    
    def _monitor_task_progress(self, task_id: int):
        """
        监控任务进度
        
        Args:
            task_id: 任务ID
        """
        def check_progress():
            status = get_task_status(task_id)
            
            if not status['exists']:
                self.update_progress(0, "")
                return
            
            if status['completed']:
                self.update_progress(100, "完成")
                return
            
            if status['cancelled']:
                self.update_progress(0, "已取消")
                return
            
            # 如果任务仍在运行，继续检查
            self.after(500, check_progress)
        
        # 开始检查
        self.after(100, check_progress)
    
    def on_shared_data_changed(self, key: str, value: Any):
        """
        共享数据变更回调
        
        Args:
            key: 数据键
            value: 数据值
        """
        # 通知所有页面刷新
        for page in self.pages.values():
            if hasattr(page, 'refresh'):
                page.refresh()
    
    def _on_closing(self):
        """窗口关闭处理"""
        if self.is_training or self.is_generating_data:
            result = messagebox.askyesno("确认", "有任务正在运行，确定要退出吗？")
            if not result:
                return
        
        try:
            # 保存配置
            save_config()
            
            # 清理资源
            shutdown_thread_pool()
            shutdown_cache_manager()
            
            logger.info("应用程序正常关闭")
            
            # 退出应用
            self.quit()
            
        except Exception as e:
            logger.error(f"关闭应用时出错: {str(e)}")
            traceback.print_exc()
            self.quit()

    def _on_back(self):
        """返回按钮处理"""
        if self.has_changes:
            self.confirm_action(
                "有未保存的更改，确定要离开吗？",
                lambda: self.controller.show_page("TrainPage")
            )
        else:
            self.controller.show_page("TrainPage")

    def _do_reset_to_default(self):
        """执行恢复默认设置"""
        # 加载默认值
        for key, setting in self.settings.items():
            widget = setting["widget"]
            var = setting["var"]
            
            if isinstance(widget, ttk.Entry) or isinstance(widget, ttk.Combobox):
                var.set("" if not hasattr(widget, "default") else widget.default)
            elif isinstance(widget, ttk.Checkbutton):
                var.set(False)
            elif isinstance(widget, ttk.Spinbox) or isinstance(widget, ttk.Scale):
                var.set(widget.cget("from"))
        
        # 设置特定默认值
        self.settings["log_level"]["var"].set("INFO") if "log_level" in self.settings else None
        self.settings["thread_pool_size"]["var"].set(4)
        self.settings["cache_size"]["var"].set(100)
        self.settings["random_seed"]["var"].set(42)
        self.settings["default_count"]["var"].set(25000)
        self.settings["test_size"]["var"].set(0.2)
        self.settings["cv_folds"]["var"].set(5)
        
        # 标记更改
        self.has_changes = True
        
        self.show_message("已恢复默认设置，点击应用以保存更改", "info")


def main():
    """主程序入口"""
    try:
        # 创建应用实例
        app = RecommenderApp()
        
        # 启动主循环
        app.mainloop()
        
    except Exception as e:
        logger.error(f"应用程序启动失败: {str(e)}")
        traceback.print_exc()
        
        # 使用tkinter显示错误消息
        root = tk.Tk()
        root.withdraw()
        messagebox.showerror("错误", f"应用程序启动失败:\n{str(e)}")
        root.destroy()


if __name__ == "__main__":
    main() 