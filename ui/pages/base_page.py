"""
基础页面模块
定义所有页面的基类
"""

import os
import tkinter as tk
from tkinter import ttk
from typing import Dict, Any, Optional, Callable
import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("base_page")


class BasePage(ttk.Frame):
    """基础页面类，所有页面继承自此类"""
    
    def __init__(self, parent, controller=None, **kwargs):
        """
        初始化基础页面
        
        Args:
            parent: 父级窗口或Frame
            controller: 页面控制器，用于页面间导航
            **kwargs: 传递给ttk.Frame的参数
        """
        super().__init__(parent, **kwargs)
        self.parent = parent
        self.controller = controller
        self.is_initialized = False
        self.is_visible = False
        
        # 共享数据
        self.shared_data = {} if controller is None else controller.shared_data
        
        # 创建UI
        self._create_widgets()
        
        # 布局
        self.pack(fill=tk.BOTH, expand=True)
        
        # 标记为已初始化
        self.is_initialized = True
        
    def _create_widgets(self):
        """创建页面小部件（子类需要重写此方法）"""
        # 默认实现：创建标题标签
        self.title_label = ttk.Label(
            self, 
            text=self.__class__.__name__, 
            font=("Helvetica", 16, "bold")
        )
        self.title_label.pack(pady=10)
        
        # 示例内容
        content_frame = ttk.Frame(self)
        content_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        
        ttk.Label(
            content_frame, 
            text="这是一个基础页面示例，请在子类中重写_create_widgets方法"
        ).pack(pady=20)
    
    def show(self):
        """显示页面"""
        if not self.is_initialized:
            logger.warning(f"页面 {self.__class__.__name__} 还未初始化")
            self._create_widgets()
            self.is_initialized = True
        
        self.pack(fill=tk.BOTH, expand=True)
        self.is_visible = True
        self.on_show()
    
    def hide(self):
        """隐藏页面"""
        self.pack_forget()
        self.is_visible = False
        self.on_hide()
    
    def refresh(self):
        """刷新页面内容"""
        # 子类可以重写此方法来更新动态内容
        pass
    
    def on_show(self):
        """页面显示时的回调"""
        # 子类可以重写此方法
        logger.debug(f"页面 {self.__class__.__name__} 显示")
    
    def on_hide(self):
        """页面隐藏时的回调"""
        # 子类可以重写此方法
        logger.debug(f"页面 {self.__class__.__name__} 隐藏")
    
    def set_title(self, title: str):
        """
        设置页面标题
        
        Args:
            title: 页面标题
        """
        if hasattr(self, 'title_label'):
            self.title_label.config(text=title)
    
    def get_shared_data(self, key: str, default: Any = None) -> Any:
        """
        获取共享数据
        
        Args:
            key: 数据键
            default: 默认值
            
        Returns:
            Any: 共享数据值
        """
        return self.shared_data.get(key, default)
    
    def set_shared_data(self, key: str, value: Any):
        """
        设置共享数据
        
        Args:
            key: 数据键
            value: 数据值
        """
        self.shared_data[key] = value
        
        # 如果控制器存在，通知共享数据已更新
        if self.controller is not None and hasattr(self.controller, 'on_shared_data_changed'):
            self.controller.on_shared_data_changed(key, value)
    
    def create_section_frame(self, title: str = None, padx: int = 10, pady: int = 5) -> ttk.LabelFrame:
        """
        创建带标题的区域框架
        
        Args:
            title: 区域标题
            padx: 水平内边距
            pady: 垂直内边距
            
        Returns:
            ttk.LabelFrame: 创建的框架
        """
        frame = ttk.LabelFrame(self, text=title) if title else ttk.Frame(self)
        frame.pack(fill=tk.BOTH, expand=True, padx=padx, pady=pady)
        return frame
    
    def create_button_bar(self, padx: int = 10, pady: int = 5) -> ttk.Frame:
        """
        创建按钮栏
        
        Args:
            padx: 水平内边距
            pady: 垂直内边距
            
        Returns:
            ttk.Frame: 按钮栏框架
        """
        button_bar = ttk.Frame(self)
        button_bar.pack(fill=tk.X, padx=padx, pady=pady)
        return button_bar
    
    def create_form_field(self, parent, label_text: str, field_type: str = "entry", 
                         options: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        创建表单字段（标签+输入控件）
        
        Args:
            parent: 父级窗口或Frame
            label_text: 标签文本
            field_type: 字段类型 (entry, combobox, checkbox, radio, text)
            options: 字段选项
            
        Returns:
            Dict[str, Any]: 包含创建的控件
        """
        if options is None:
            options = {}
        
        # 创建容器框架
        frame = ttk.Frame(parent)
        frame.pack(fill=tk.X, padx=5, pady=3)
        
        # 创建标签
        label = ttk.Label(frame, text=label_text, width=15, anchor=tk.W)
        label.pack(side=tk.LEFT, padx=5)
        
        # 根据字段类型创建输入控件
        field = None
        var = None
        
        if field_type == "entry":
            # 文本输入框
            var = tk.StringVar(value=options.get("default", ""))
            field = ttk.Entry(frame, textvariable=var, width=options.get("width", 30))
            field.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
            
        elif field_type == "combobox":
            # 下拉选择框
            var = tk.StringVar(value=options.get("default", ""))
            field = ttk.Combobox(frame, textvariable=var, width=options.get("width", 28))
            field['values'] = options.get("values", [])
            field.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
            
            if "default" in options:
                field.current(field['values'].index(options["default"]))
            
        elif field_type == "checkbox":
            # 复选框
            var = tk.BooleanVar(value=options.get("default", False))
            field = ttk.Checkbutton(frame, variable=var, text=options.get("text", ""))
            field.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
            
        elif field_type == "radio":
            # 单选按钮组
            var = tk.StringVar(value=options.get("default", ""))
            radio_frame = ttk.Frame(frame)
            radio_frame.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
            
            for i, (value, text) in enumerate(options.get("choices", {}).items()):
                radio = ttk.Radiobutton(radio_frame, text=text, variable=var, value=value)
                radio.pack(side=tk.LEFT, padx=5)
                if i == 0:
                    field = radio
            
        elif field_type == "text":
            # 多行文本框
            var = tk.StringVar(value=options.get("default", ""))
            field = tk.Text(frame, width=options.get("width", 30), height=options.get("height", 5))
            field.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
            
            # 设置默认值
            if "default" in options:
                field.insert("1.0", options["default"])
        
        # 返回创建的控件
        return {
            "frame": frame,
            "label": label,
            "field": field,
            "var": var,
            "type": field_type
        }
    
    def show_message(self, message: str, message_type: str = "info"):
        """
        显示消息
        
        Args:
            message: 消息内容
            message_type: 消息类型 (info, warning, error)
        """
        if self.controller is not None and hasattr(self.controller, 'show_message'):
            self.controller.show_message(message, message_type)
        else:
            logger.info(f"消息 ({message_type}): {message}")
    
    def confirm_action(self, message: str, callback: Callable[[], None] = None):
        """
        确认操作
        
        Args:
            message: 确认消息
            callback: 确认后的回调函数
        """
        if self.controller is not None and hasattr(self.controller, 'confirm_action'):
            self.controller.confirm_action(message, callback)
        elif callback is not None:
            # 如果没有控制器，直接执行回调
            callback()


# 测试代码
if __name__ == "__main__":
    root = tk.Tk()
    root.title("基础页面测试")
    root.geometry("800x600")
    
    # 创建基础页面
    page = BasePage(root)
    
    # 显示窗口
    root.mainloop() 