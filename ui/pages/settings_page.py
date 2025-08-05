"""
设置页面模块
提供系统参数配置功能
"""

import os
import tkinter as tk
from tkinter import ttk, filedialog, colorchooser, messagebox
import logging
from typing import Dict, Any, List, Optional
import json

# 导入基础页面类
from ui.pages.base_page import BasePage

# 导入配置工具
from utils.config import get_config, set_config, save_config, get_data_dir, get_models_dir

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("settings_page")


class SettingsPage(BasePage):
    """设置页面类"""
    
    def __init__(self, parent, controller=None, **kwargs):
        """
        初始化设置页面
        
        Args:
            parent: 父级窗口或Frame
            controller: 页面控制器
            **kwargs: 传递给BasePage的参数
        """
        # 设置变量
        self.settings = {}
        self.has_changes = False
        
        super().__init__(parent, controller, **kwargs)
        logger.debug("设置页面已初始化")
    
    def _create_widgets(self):
        """创建页面小部件"""
        # 设置页面标题
        self.title_label = ttk.Label(
            self, 
            text="系统设置", 
            font=("Helvetica", 18, "bold")
        )
        self.title_label.pack(pady=(20, 10))
        
        # 创建主框架
        main_frame = ttk.Frame(self)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        
        # 创建标签页
        self.notebook = ttk.Notebook(main_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        
        # 性能设置标签页
        performance_frame = ttk.Frame(self.notebook)
        self.notebook.add(performance_frame, text="性能设置")
        
        # 数据设置标签页
        data_frame = ttk.Frame(self.notebook)
        self.notebook.add(data_frame, text="数据设置")
        
        # 添加各标签页内容
        self._create_performance_settings(performance_frame)
        self._create_data_settings(data_frame)
        
        # 底部按钮
        button_frame = ttk.Frame(self)
        button_frame.pack(fill=tk.X, padx=20, pady=15)
        
        ttk.Button(
            button_frame, 
            text="应用",
            command=self._apply_settings,
            width=15
        ).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(
            button_frame, 
            text="恢复默认",
            command=self._reset_to_default,
            width=15
        ).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(
            button_frame, 
            text="返回",
            command=self._on_back,
            width=15
        ).pack(side=tk.RIGHT, padx=5)
        
        # 加载设置
        self._load_settings()
    
    def _create_performance_settings(self, parent):
        """
        创建性能设置
        
        Args:
            parent: 父级容器
        """
        # 创建框架
        frame = ttk.Frame(parent, padding=10)
        frame.pack(fill=tk.BOTH, expand=True)
        
        # 线程设置
        thread_frame = ttk.LabelFrame(frame, text="线程设置")
        thread_frame.pack(fill=tk.X, pady=10)
        
        # 线程池大小
        thread_pool_var = tk.IntVar()
        thread_pool_field = self._add_setting_field(
            thread_frame, 
            "线程池大小:", 
            thread_pool_var, 
            "spinbox", 
            {"from_": 1, "to": 32, "default": 4}
        )
        self.settings["thread_pool_size"] = {
            "var": thread_pool_var,
            "widget": thread_pool_field["field"],
            "section": "performance",
            "key": "thread_pool_size"
        }
        
        # 是否使用多线程
        use_threading_var = tk.BooleanVar()
        use_threading_field = self._add_setting_field(
            thread_frame, 
            "启用多线程:", 
            use_threading_var, 
            "checkbox", 
            {"default": True}
        )
        self.settings["use_threading"] = {
            "var": use_threading_var,
            "widget": use_threading_field["field"],
            "section": "performance",
            "key": "use_threading"
        }
        
        # 缓存设置
        cache_frame = ttk.LabelFrame(frame, text="缓存设置")
        cache_frame.pack(fill=tk.X, pady=10)
        
        # 是否使用缓存
        use_cache_var = tk.BooleanVar()
        use_cache_field = self._add_setting_field(
            cache_frame, 
            "启用缓存:", 
            use_cache_var, 
            "checkbox", 
            {"default": True}
        )
        self.settings["use_cache"] = {
            "var": use_cache_var,
            "widget": use_cache_field["field"],
            "section": "performance",
            "key": "use_cache"
        }
        
        # 缓存大小限制(MB)
        cache_size_var = tk.IntVar()
        cache_size_field = self._add_setting_field(
            cache_frame, 
            "缓存大小限制(MB):", 
            cache_size_var, 
            "spinbox", 
            {"from_": 10, "to": 1024, "default": 100}
        )
        self.settings["cache_size"] = {
            "var": cache_size_var,
            "widget": cache_size_field["field"],
            "section": "performance",
            "key": "cache_size_mb"
        }
        
        # 缓存过期时间(分钟)
        cache_ttl_var = tk.IntVar()
        cache_ttl_field = self._add_setting_field(
            cache_frame, 
            "缓存过期时间(分钟):", 
            cache_ttl_var, 
            "spinbox", 
            {"from_": 1, "to": 1440, "default": 60}
        )
        self.settings["cache_ttl"] = {
            "var": cache_ttl_var,
            "widget": cache_ttl_field["field"],
            "section": "performance",
            "key": "cache_ttl_minutes"
        }
        
        # 模型设置
        model_frame = ttk.LabelFrame(frame, text="模型设置")
        model_frame.pack(fill=tk.X, pady=10)
        
        # 模型预加载
        model_preload_var = tk.BooleanVar()
        model_preload_field = self._add_setting_field(
            model_frame, 
            "预加载模型:", 
            model_preload_var, 
            "checkbox", 
            {"default": False}
        )
        self.settings["model_preload"] = {
            "var": model_preload_var,
            "widget": model_preload_field["field"],
            "section": "model",
            "key": "preload"
        }
        
        # 批处理大小
        batch_size_var = tk.IntVar()
        batch_size_field = self._add_setting_field(
            model_frame, 
            "批处理大小:", 
            batch_size_var, 
            "spinbox", 
            {"from_": 1, "to": 1000, "default": 32}
        )
        self.settings["batch_size"] = {
            "var": batch_size_var,
            "widget": batch_size_field["field"],
            "section": "model",
            "key": "batch_size"
        }
    
    def _create_data_settings(self, parent):
        """
        创建数据设置
        
        Args:
            parent: 父级容器
        """
        # 创建框架
        frame = ttk.Frame(parent, padding=10)
        frame.pack(fill=tk.BOTH, expand=True)
        
        # 数据生成设置
        data_gen_frame = ttk.LabelFrame(frame, text="数据生成设置")
        data_gen_frame.pack(fill=tk.X, pady=10)
        
        # 随机种子
        random_seed_var = tk.IntVar()
        random_seed_field = self._add_setting_field(
            data_gen_frame, 
            "随机种子:", 
            random_seed_var, 
            "spinbox", 
            {"from_": 0, "to": 1000000, "default": 42}
        )
        self.settings["random_seed"] = {
            "var": random_seed_var,
            "widget": random_seed_field["field"],
            "section": "data",
            "key": "random_seed"
        }
        
        # 默认生成数量
        default_count_var = tk.IntVar()
        default_count_field = self._add_setting_field(
            data_gen_frame, 
            "默认生成数量:", 
            default_count_var, 
            "spinbox", 
            {"from_": 1000, "to": 1000000, "default": 25000}
        )
        self.settings["default_count"] = {
            "var": default_count_var,
            "widget": default_count_field["field"],
            "section": "data",
            "key": "default_count"
        }
        
        # 特征数量
        feature_count_var = tk.IntVar()
        feature_count_field = self._add_setting_field(
            data_gen_frame, 
            "特征数量:", 
            feature_count_var, 
            "spinbox", 
            {"from_": 5, "to": 100, "default": 20}
        )
        self.settings["feature_count"] = {
            "var": feature_count_var,
            "widget": feature_count_field["field"],
            "section": "data",
            "key": "feature_count"
        }
        
        # 训练设置
        train_frame = ttk.LabelFrame(frame, text="训练设置")
        train_frame.pack(fill=tk.X, pady=10)
        
        # 默认测试集比例
        test_size_var = tk.DoubleVar()
        test_size_field = self._add_setting_field(
            train_frame, 
            "默认测试集比例:", 
            test_size_var, 
            "scale", 
            {"from_": 0.1, "to": 0.5, "default": 0.2}
        )
        self.settings["test_size"] = {
            "var": test_size_var,
            "widget": test_size_field["field"],
            "section": "training",
            "key": "default_test_size"
        }
        
        # 交叉验证折数
        cv_folds_var = tk.IntVar()
        cv_folds_field = self._add_setting_field(
            train_frame, 
            "交叉验证折数:", 
            cv_folds_var, 
            "spinbox", 
            {"from_": 2, "to": 10, "default": 5}
        )
        self.settings["cv_folds"] = {
            "var": cv_folds_var,
            "widget": cv_folds_field["field"],
            "section": "training",
            "key": "cv_folds"
        }
        
        # 是否使用早停
        early_stopping_var = tk.BooleanVar()
        early_stopping_field = self._add_setting_field(
            train_frame, 
            "使用早停:", 
            early_stopping_var, 
            "checkbox", 
            {"default": True}
        )
        self.settings["early_stopping"] = {
            "var": early_stopping_var,
            "widget": early_stopping_field["field"],
            "section": "training",
            "key": "early_stopping"
        }
        
        # 保存设置
        save_frame = ttk.LabelFrame(frame, text="保存设置")
        save_frame.pack(fill=tk.X, pady=10)
        
        # 是否自动保存模型
        auto_save_var = tk.BooleanVar()
        auto_save_field = self._add_setting_field(
            save_frame, 
            "自动保存模型:", 
            auto_save_var, 
            "checkbox", 
            {"default": True}
        )
        self.settings["auto_save"] = {
            "var": auto_save_var,
            "widget": auto_save_field["field"],
            "section": "saving",
            "key": "auto_save_model"
        }
        
        # 是否保存训练历史
        save_history_var = tk.BooleanVar()
        save_history_field = self._add_setting_field(
            save_frame, 
            "保存训练历史:", 
            save_history_var, 
            "checkbox", 
            {"default": True}
        )
        self.settings["save_history"] = {
            "var": save_history_var,
            "widget": save_history_field["field"],
            "section": "saving",
            "key": "save_history"
        }
    
    def _add_setting_field(self, parent, label_text, variable, field_type, options=None):
        """
        添加设置字段
        
        Args:
            parent: 父级容器
            label_text: 标签文本
            variable: 变量
            field_type: 字段类型
            options: 选项
            
        Returns:
            Dict: 字段信息
        """
        if options is None:
            options = {}
        
        frame = ttk.Frame(parent)
        frame.pack(fill=tk.X, padx=5, pady=3)
        
        # 创建标签
        label = ttk.Label(frame, text=label_text, width=20, anchor=tk.W)
        label.pack(side=tk.LEFT, padx=5)
        
        field = None
        
        # 创建不同类型的输入字段
        if field_type == "entry":
            field = ttk.Entry(frame, textvariable=variable, width=30)
            field.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
            
            if "default" in options:
                variable.set(options["default"])
                
        elif field_type == "combobox":
            field = ttk.Combobox(frame, textvariable=variable, width=30)
            field["values"] = options.get("values", [])
            field.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
            
            if "default" in options:
                variable.set(options["default"])
                
        elif field_type == "checkbox":
            field = ttk.Checkbutton(frame, variable=variable)
            field.pack(side=tk.LEFT, padx=5)
            
            if "default" in options:
                variable.set(options["default"])
                
        elif field_type == "spinbox":
            field = ttk.Spinbox(
                frame, 
                from_=options.get("from_", 0), 
                to=options.get("to", 100), 
                textvariable=variable,
                width=10
            )
            field.pack(side=tk.LEFT, padx=5)
            
            if "default" in options:
                variable.set(options["default"])
                
        elif field_type == "scale":
            frame.pack_forget()  # 重新布局
            
            frame = ttk.Frame(parent)
            frame.pack(fill=tk.X, padx=5, pady=8)
            
            label = ttk.Label(frame, text=label_text, width=20, anchor=tk.W)
            label.pack(side=tk.LEFT, padx=5)
            
            field = ttk.Scale(
                frame, 
                from_=options.get("from_", 0), 
                to=options.get("to", 1), 
                variable=variable,
                orient=tk.HORIZONTAL,
                length=150
            )
            field.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
            
            value_label = ttk.Label(frame, width=5)
            value_label.pack(side=tk.LEFT, padx=5)
            
            def update_value_label(*args):
                value_label.config(text=f"{variable.get():.2f}")
            
            variable.trace_add("write", update_value_label)
            
            if "default" in options:
                variable.set(options["default"])
                update_value_label()
                
        elif field_type == "dir_entry":
            entry_frame = ttk.Frame(frame)
            entry_frame.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
            
            field = ttk.Entry(entry_frame, textvariable=variable)
            field.pack(side=tk.LEFT, fill=tk.X, expand=True)
            
            browse_button = ttk.Button(
                entry_frame, 
                text="...",
                width=3,
                command=lambda: self._browse_directory(variable)
            )
            browse_button.pack(side=tk.RIGHT, padx=(5, 0))
            
            if "default" in options:
                variable.set(options["default"])
                
        elif field_type == "color_entry":
            entry_frame = ttk.Frame(frame)
            entry_frame.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
            
            field = ttk.Entry(entry_frame, textvariable=variable, width=10)
            field.pack(side=tk.LEFT)
            
            color_preview = tk.Frame(
                entry_frame, 
                width=20, 
                height=20, 
                bg=options.get("default", "#FFFFFF")
            )
            color_preview.pack(side=tk.LEFT, padx=5)
            
            browse_button = ttk.Button(
                entry_frame, 
                text="选择",
                width=5,
                command=lambda: self._choose_color(variable, color_preview)
            )
            browse_button.pack(side=tk.LEFT, padx=5)
            
            if "default" in options:
                variable.set(options["default"])
                color_preview.config(bg=options["default"])
        
        return {
            "frame": frame,
            "label": label,
            "field": field,
            "var": variable
        }
    
    def _browse_directory(self, var):
        """
        浏览选择目录
        
        Args:
            var: 存储目录路径的变量
        """
        current_dir = var.get() if var.get() and os.path.exists(var.get()) else os.getcwd()
        dir_path = filedialog.askdirectory(initialdir=current_dir)
        
        if dir_path:
            var.set(dir_path)
            self.has_changes = True
    
    def _choose_color(self, var, preview_widget):
        """
        选择颜色
        
        Args:
            var: 存储颜色的变量
            preview_widget: 预览框架
        """
        current_color = var.get() if var.get() else "#FFFFFF"
        color = colorchooser.askcolor(color=current_color, title="选择颜色")
        
        if color[1]:  # 用户未取消
            var.set(color[1])
            preview_widget.config(bg=color[1])
            self.has_changes = True
    
    def _load_settings(self):
        """加载设置"""
        for key, setting in self.settings.items():
            section = setting["section"]
            config_key = setting["key"]
            
            # 获取配置值
            default_value = setting["var"].get()
            value = get_config(section, config_key, default_value)
            
            # 设置变量值
            setting["var"].set(value)
        
        # 重置更改标记
        self.has_changes = False
    
    def _apply_settings(self):
        """应用设置"""
        for key, setting in self.settings.items():
            section = setting["section"]
            config_key = setting["key"]
            value = setting["var"].get()
            
            # 保存到配置
            set_config(section, config_key, value)
        
        # 保存配置到文件
        save_config()
        
        # 重置更改标记
        self.has_changes = False
        
        # 更新共享数据
        if hasattr(self.controller, 'on_shared_data_changed'):
            self.controller.on_shared_data_changed('settings_updated', True)
        
        self.show_message("设置已成功保存", "info")
        logger.info("应用设置已保存")
    
    def _reset_to_default(self):
        """恢复默认设置"""
        # 确认对话框
        self.confirm_action(
            "确定要恢复默认设置吗？当前的设置将会丢失。",
            self._do_reset_to_default
        )
    
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
        self.settings["app_name"]["var"].set("智能题目推荐系统")
        self.settings["version"]["var"].set("1.0.0")
        self.settings["log_level"]["var"].set("INFO")
        self.settings["thread_pool_size"]["var"].set(4)
        self.settings["cache_size"]["var"].set(100)
        self.settings["random_seed"]["var"].set(42)
        self.settings["default_count"]["var"].set(25000)
        self.settings["test_size"]["var"].set(0.2)
        self.settings["cv_folds"]["var"].set(5)
        
        # 标记更改
        self.has_changes = True
        
        self.show_message("已恢复默认设置，点击应用以保存更改", "info")
    
    def _on_back(self):
        """返回按钮处理"""
        if self.has_changes:
            self.confirm_action(
                "有未保存的更改，确定要离开吗？",
                lambda: self.controller.show_page("TrainPage")
            )
        else:
            self.controller.show_page("TrainPage")
    
    def refresh(self):
        """刷新页面内容"""
        # 可以在这里添加刷新逻辑
        pass
    
    def on_show(self):
        """页面显示时的回调"""
        logger.debug("设置页面显示")
        self._load_settings()  # 重新加载设置


# 测试代码
if __name__ == "__main__":
    root = tk.Tk()
    root.title("设置页面测试")
    root.geometry("800x600")
    
    # 模拟控制器
    class MockController:
        def __init__(self):
            self.shared_data = {}
        
        def show_page(self, page_name):
            print(f"切换到页面: {page_name}")
        
        def on_shared_data_changed(self, key, value):
            print(f"数据已更改: {key}={value}")
    
    # 创建页面
    page = SettingsPage(root, MockController())
    page.show()
    
    # 显示窗口
    root.mainloop() 