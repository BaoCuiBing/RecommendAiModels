"""
数据生成页面模块
用于生成训练数据和导入外部数据集
"""

import os
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import pandas as pd
import numpy as np
import threading
import logging
from typing import Dict, Any, List, Optional, Tuple
import time
from datetime import datetime

# 导入基础页面类
from ui.pages.base_page import BasePage

# 导入数据生成器模块
from data.data_generator import (
    generate_test_data, get_generation_progress, 
    DataGenerator
)

# 导入线程池工具
from utils.thread_pool import submit_task, get_task_status

# 导入配置
from utils.config import get_config, set_config, get_data_dir

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("data_page")


class DataPage(BasePage):
    """数据生成和管理页面类"""
    
    def __init__(self, parent, controller=None, **kwargs):
        """
        初始化数据页面
        
        Args:
            parent: 父级窗口或Frame
            controller: 页面控制器
            **kwargs: 传递给BasePage的参数
        """
        # 任务ID
        self.generation_task_id = None
        
        # 数据生成设置
        self.data_settings = {
            'users_count': tk.IntVar(value=500),
            'questions_count': tk.IntVar(value=2000),
            'attempts_count': tk.IntVar(value=20000),
            'history_count': tk.IntVar(value=3000),
        }
        
        # 初始化基类
        super().__init__(parent, controller, **kwargs)
        logger.debug("数据页面已初始化")
    
    def _create_widgets(self):
        """创建页面小部件"""
        # 设置页面标题
        self.title_label = ttk.Label(
            self, 
            text="数据生成与管理", 
            font=("Helvetica", 16, "bold")
        )
        self.title_label.pack(pady=(10, 5))
        
        # 创建主框架
        main_frame = ttk.Frame(self)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        
        # 左侧：数据生成配置
        self.settings_frame = ttk.LabelFrame(main_frame, text="数据生成配置")
        self.settings_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10), pady=5)
        
        # 生成设置
        self._create_generation_settings()
        
        # 右侧：数据管理
        management_frame = ttk.LabelFrame(main_frame, text="数据管理")
        management_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(10, 0), pady=5)
        
        # 数据导入导出
        self._create_data_management(management_frame)
        
        # 底部：生成按钮和进度条
        button_frame = ttk.Frame(self)
        button_frame.pack(fill=tk.X, padx=20, pady=10)
        
        # 数据生成按钮
        self.generate_button = ttk.Button(
            button_frame,
            text="生成测试数据",
            command=self._generate_data,
            width=20
        )
        self.generate_button.pack(side=tk.LEFT, padx=5)
        
        # 停止按钮
        self.stop_button = ttk.Button(
            button_frame,
            text="停止生成",
            command=self._stop_generation,
            width=15,
            state=tk.DISABLED
        )
        self.stop_button.pack(side=tk.LEFT, padx=5)
        
        # 进度条框架
        progress_frame = ttk.Frame(button_frame)
        progress_frame.pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=5)
        
        # 进度条
        self.progress_var = tk.DoubleVar(value=0)
        self.progress = ttk.Progressbar(
            progress_frame,
            orient=tk.HORIZONTAL,
            variable=self.progress_var,
            mode='determinate',
            length=200
        )
        self.progress.pack(side=tk.TOP, fill=tk.X, pady=2)
        
        # 进度标签
        self.progress_label = ttk.Label(
            progress_frame,
            text="就绪",
            anchor=tk.W
        )
        self.progress_label.pack(side=tk.TOP, fill=tk.X)
        
        # 创建统计信息框架
        stats_frame = ttk.LabelFrame(self, text="数据统计")
        stats_frame.pack(fill=tk.X, padx=20, pady=5)
        
        # 统计信息文本框
        self.stats_text = tk.Text(stats_frame, wrap=tk.WORD, height=6, width=40)
        self.stats_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        stats_scrollbar = ttk.Scrollbar(stats_frame, orient=tk.VERTICAL, command=self.stats_text.yview)
        stats_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.stats_text.config(yscrollcommand=stats_scrollbar.set)
        self.stats_text.config(state=tk.DISABLED)
    
    def _create_generation_settings(self):
        """创建数据生成设置"""
        # 加载默认值
        default_settings = get_config('data_generator', None, {})
        
        if default_settings:
            for key, var in self.data_settings.items():
                if key in default_settings:
                    var.set(default_settings[key])
        
        # 创建设置控件
        settings_inner = ttk.Frame(self.settings_frame)
        settings_inner.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # 用户数量
        user_frame = ttk.Frame(settings_inner)
        user_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(user_frame, text="用户数量:", width=12).pack(side=tk.LEFT)
        user_spin = ttk.Spinbox(
            user_frame, 
            from_=10, 
            to=10000, 
            textvariable=self.data_settings['users_count'],
            width=10
        )
        user_spin.pack(side=tk.LEFT, padx=5)
        
        ttk.Scale(
            user_frame,
            from_=10,
            to=10000,
            variable=self.data_settings['users_count'],
            orient=tk.HORIZONTAL
        ).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        
        # 题目数量
        question_frame = ttk.Frame(settings_inner)
        question_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(question_frame, text="题目数量:", width=12).pack(side=tk.LEFT)
        question_spin = ttk.Spinbox(
            question_frame, 
            from_=100, 
            to=50000, 
            textvariable=self.data_settings['questions_count'],
            width=10
        )
        question_spin.pack(side=tk.LEFT, padx=5)
        
        ttk.Scale(
            question_frame,
            from_=100,
            to=50000,
            variable=self.data_settings['questions_count'],
            orient=tk.HORIZONTAL
        ).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        
        # 答题记录数量
        attempt_frame = ttk.Frame(settings_inner)
        attempt_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(attempt_frame, text="答题记录:", width=12).pack(side=tk.LEFT)
        attempt_spin = ttk.Spinbox(
            attempt_frame, 
            from_=500, 
            to=500000, 
            textvariable=self.data_settings['attempts_count'],
            width=10
        )
        attempt_spin.pack(side=tk.LEFT, padx=5)
        
        ttk.Scale(
            attempt_frame,
            from_=500,
            to=500000,
            variable=self.data_settings['attempts_count'],
            orient=tk.HORIZONTAL
        ).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        
        # 答题历史数量
        history_frame = ttk.Frame(settings_inner)
        history_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(history_frame, text="答题历史:", width=12).pack(side=tk.LEFT)
        history_spin = ttk.Spinbox(
            history_frame, 
            from_=100, 
            to=100000, 
            textvariable=self.data_settings['history_count'],
            width=10
        )
        history_spin.pack(side=tk.LEFT, padx=5)
        
        ttk.Scale(
            history_frame,
            from_=100,
            to=100000,
            variable=self.data_settings['history_count'],
            orient=tk.HORIZONTAL
        ).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        
        # 提示信息
        ttk.Label(
            settings_inner,
            text="注意: 生成大量数据可能需要较长时间，请耐心等待。",
            foreground="blue"
        ).pack(pady=10)
        
        # 预设按钮
        presets_frame = ttk.Frame(settings_inner)
        presets_frame.pack(fill=tk.X, pady=5)
        
        ttk.Button(
            presets_frame,
            text="小数据集",
            command=lambda: self._apply_preset('small')
        ).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(
            presets_frame,
            text="中数据集",
            command=lambda: self._apply_preset('medium')
        ).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(
            presets_frame,
            text="大数据集",
            command=lambda: self._apply_preset('large')
        ).pack(side=tk.LEFT, padx=5)
        
        # 保存设置按钮
        ttk.Button(
            settings_inner,
            text="保存为默认配置",
            command=self._save_settings
        ).pack(pady=10)
    
    def _create_data_management(self, parent):
        """创建数据管理控件"""
        # 创建内部框架
        management_inner = ttk.Frame(parent)
        management_inner.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # 文件列表框架
        files_frame = ttk.LabelFrame(management_inner, text="数据文件")
        files_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # 文件列表
        self.files_listbox = tk.Listbox(files_frame, selectmode=tk.EXTENDED, height=10)
        self.files_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # 滚动条
        scrollbar = ttk.Scrollbar(files_frame, orient=tk.VERTICAL, command=self.files_listbox.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.files_listbox.config(yscrollcommand=scrollbar.set)
        
        # 刷新文件列表
        self._refresh_file_list()
        
        # 按钮框架
        buttons_frame = ttk.Frame(management_inner)
        buttons_frame.pack(fill=tk.X, pady=10)
        
        # 添加数据管理按钮
        ttk.Button(
            buttons_frame,
            text="刷新列表",
            command=self._refresh_file_list,
            width=15
        ).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(
            buttons_frame,
            text="导入数据",
            command=self._import_data,
            width=15
        ).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(
            buttons_frame,
            text="导出数据",
            command=self._export_data,
            width=15
        ).pack(side=tk.LEFT, padx=5)
        
        # 添加合并数据按钮
        ttk.Button(
            buttons_frame,
            text="合并CSV文件",
            command=self._merge_csv_files,
            width=15
        ).pack(side=tk.LEFT, padx=5)
        
        # 创建预览框架
        preview_frame = ttk.LabelFrame(management_inner, text="数据预览")
        preview_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # 预览文本框
        self.preview_text = tk.Text(preview_frame, wrap=tk.WORD, width=40, height=10)
        self.preview_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        preview_scrollbar = ttk.Scrollbar(preview_frame, orient=tk.VERTICAL, command=self.preview_text.yview)
        preview_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.preview_text.config(yscrollcommand=preview_scrollbar.set)
        
        # 绑定文件列表选择事件
        self.files_listbox.bind('<<ListboxSelect>>', self._on_file_select)
    
    def _apply_preset(self, preset: str):
        """
        应用数据生成预设
        
        Args:
            preset: 预设名称 ('small', 'medium', 'large')
        """
        if preset == 'small':
            self.data_settings['users_count'].set(50)
            self.data_settings['questions_count'].set(200)
            self.data_settings['attempts_count'].set(1000)
            self.data_settings['history_count'].set(300)
        elif preset == 'medium':
            self.data_settings['users_count'].set(500)
            self.data_settings['questions_count'].set(2000)
            self.data_settings['attempts_count'].set(20000)
            self.data_settings['history_count'].set(3000)
        elif preset == 'large':
            self.data_settings['users_count'].set(5000)
            self.data_settings['questions_count'].set(20000)
            self.data_settings['attempts_count'].set(200000)
            self.data_settings['history_count'].set(30000)
    
    def _save_settings(self):
        """保存当前设置为默认配置"""
        # 获取当前设置
        settings = {
            'users_count': self.data_settings['users_count'].get(),
            'questions_count': self.data_settings['questions_count'].get(),
            'attempts_count': self.data_settings['attempts_count'].get(),
            'history_count': self.data_settings['history_count'].get(),
        }
        
        # 更新配置
        for key, value in settings.items():
            set_config('data_generator', key, value)
        
        # 显示消息
        self.show_message("设置已保存为默认配置", "info")
    
    def _generate_data(self):
        """生成测试数据"""
        # 获取设置
        users_count = self.data_settings['users_count'].get()
        questions_count = self.data_settings['questions_count'].get()
        attempts_count = self.data_settings['attempts_count'].get()
        history_count = self.data_settings['history_count'].get()
        
        # 确认是否生成
        message = f"即将生成以下测试数据:\n\n" \
                  f"用户数量: {users_count}\n" \
                  f"题目数量: {questions_count}\n" \
                  f"答题记录: {attempts_count}\n" \
                  f"答题历史: {history_count}\n\n" \
                  f"生成大量数据可能需要较长时间，是否继续？"
        
        self.confirm_action(message, lambda: self._start_data_generation(
            users_count, questions_count, attempts_count, history_count
        ))
    
    def _start_data_generation(self, users_count, questions_count, attempts_count, history_count):
        """
        开始数据生成
        
        Args:
            users_count: 用户数量
            questions_count: 题目数量
            attempts_count: 答题记录数量
            history_count: 答题历史数量
        """
        # 更新UI状态
        self.generate_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)
        self.progress_var.set(0)
        self.progress_label.config(text="正在启动数据生成...")
        
        # 通知控制器
        if self.controller:
            self.controller.is_generating_data = True
            self.controller.update_status("正在生成数据...")
        
        # 创建生成任务
        data_dir = get_data_dir()
        
        # 创建任务
        self.generation_task_id = submit_task(
            generate_test_data,
            users_count=users_count,
            questions_count=questions_count,
            attempts_count=attempts_count,
            history_count=history_count,
            data_dir=data_dir
        )
        
        # 启动进度监控
        self._monitor_generation_progress()
    
    def _monitor_generation_progress(self):
        """监控数据生成进度"""
        if not hasattr(self, 'generation_task_id') or self.generation_task_id is None:
            return
        
        # 检查任务状态
        task_status = get_task_status(self.generation_task_id)
        if not task_status['exists']:
            self._generation_completed()
            return
        
        # 如果任务已完成
        if task_status['completed']:
            self._generation_completed()
            return
        
        # 如果任务被取消
        if task_status['cancelled']:
            self._generation_stopped()
            return
        
        # 获取进度
        progress = get_generation_progress()
        if progress:
            # 更新进度条
            self.progress_var.set(progress['total'])
            
            # 更新进度文本
            progress_text = f"总进度: {progress['total']}% | "
            progress_text += f"用户: {progress['users']}% | "
            progress_text += f"题目: {progress['questions']}% | "
            progress_text += f"答题: {progress['attempts']}% | "
            progress_text += f"历史: {progress['history']}%"
            self.progress_label.config(text=progress_text)
            
            # 更新控制器进度
            if self.controller:
                self.controller.update_progress(
                    progress['total'], 
                    f"数据生成: {progress['total']}%"
                )
        
        # 继续监控
        self.after(500, self._monitor_generation_progress)
    
    def _generation_completed(self):
        """数据生成完成处理"""
        # 更新UI状态
        self.generate_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)
        self.progress_var.set(100)
        self.progress_label.config(text="数据生成完成")
        
        # 通知控制器
        if self.controller:
            self.controller.is_generating_data = False
            self.controller.update_status("就绪")
            self.controller.update_progress(0, "")
        
        # 显示消息
        self.show_message("测试数据生成完成", "info")
        
        # 更新数据统计
        self._update_data_stats()
        
        # 清除任务ID
        self.generation_task_id = None
    
    def _generation_stopped(self):
        """数据生成中止处理"""
        # 更新UI状态
        self.generate_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)
        self.progress_var.set(0)
        self.progress_label.config(text="数据生成已停止")
        
        # 通知控制器
        if self.controller:
            self.controller.is_generating_data = False
            self.controller.update_status("就绪")
            self.controller.update_progress(0, "")
        
        # 清除任务ID
        self.generation_task_id = None
    
    def _stop_generation(self):
        """停止数据生成"""
        if not hasattr(self, 'generation_task_id') or self.generation_task_id is None:
            return
        
        # 确认是否停止
        self.confirm_action(
            "确定要停止数据生成吗？当前进度将丢失。",
            self._confirm_stop_generation
        )
    
    def _confirm_stop_generation(self):
        """确认停止数据生成"""
        from utils.thread_pool import cancel_task
        
        if self.generation_task_id:
            # 取消任务
            cancel_task(self.generation_task_id)
            
            # 更新状态
            self._generation_stopped()
    
    def _import_data(self):
        """导入数据"""
        # 选择文件
        file_path = filedialog.askopenfilename(
            title="选择要导入的数据文件",
            filetypes=[("CSV Files", "*.csv"), ("All Files", "*.*")]
        )
        
        if not file_path:
            return
        
        # 确认导入
        self.confirm_action(
            f"确定要导入 {file_path} 吗？如果目标文件已存在，将被覆盖。",
            lambda: self._confirm_import_data(file_path)
        )
    
    def _confirm_import_data(self, source_path: str):
        """确认导入数据"""
        # 目标路径
        data_dir = get_data_dir()
        target_path = os.path.join(data_dir, os.path.basename(source_path))
        
        try:
            # 读取源文件
            df = pd.read_csv(source_path)
            
            # 确保目录存在
            if not os.path.exists(data_dir):
                os.makedirs(data_dir)
            
            # 保存到目标文件
            df.to_csv(target_path, index=False)
            
            # 显示消息
            self.show_message(
                f"成功导入 {len(df)} 条记录到 {target_path}",
                "info"
            )
            
            # 更新数据统计
            self._update_data_stats()
            
        except Exception as e:
            # 显示错误
            self.show_message(
                f"导入数据失败: {str(e)}",
                "error"
            )
            logger.error(f"导入数据失败: {str(e)}")
    
    def _update_data_stats(self):
        """更新数据统计信息"""
        data_dir = get_data_dir()
        stats_text = "数据文件统计：\n\n"
        
        # 检查数据文件
        data_files = {
            'users.csv': '用户数据',
            'questions.csv': '题目数据',
            'quiz_attempts.csv': '答题记录',
            'quiz_history.csv': '答题历史'
        }
        
        for filename, description in data_files.items():
            file_path = os.path.join(data_dir, filename)
            
            if os.path.exists(file_path):
                try:
                    df = pd.read_csv(file_path)
                    stats_text += f"{description}: {len(df)} 条记录\n"
                    
                    # 添加文件大小和修改时间
                    size_mb = os.path.getsize(file_path) / (1024 * 1024)
                    mod_time = datetime.fromtimestamp(os.path.getmtime(file_path))
                    
                    stats_text += f"  文件大小: {size_mb:.2f} MB\n"
                    stats_text += f"  修改时间: {mod_time.strftime('%Y-%m-%d %H:%M:%S')}\n\n"
                    
                except Exception as e:
                    stats_text += f"{description}: 读取失败 ({str(e)})\n\n"
            else:
                stats_text += f"{description}: 文件不存在\n\n"
        
        # 更新统计文本
        self.stats_text.config(state=tk.NORMAL)
        self.stats_text.delete("1.0", tk.END)
        self.stats_text.insert("1.0", stats_text)
        self.stats_text.config(state=tk.DISABLED)
    
    def _open_data_folder(self):
        """打开数据文件夹"""
        data_dir = get_data_dir()
        
        # 确保目录存在
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
        
        # 使用系统默认程序打开文件夹
        import platform
        import subprocess
        
        if platform.system() == "Windows":
            os.startfile(data_dir)
        elif platform.system() == "Darwin":  # macOS
            subprocess.Popen(["open", data_dir])
        else:  # Linux
            subprocess.Popen(["xdg-open", data_dir])
    
    def _backup_data(self):
        """备份数据文件"""
        data_dir = get_data_dir()
        
        # 检查是否有数据文件
        data_files = ['users.csv', 'questions.csv', 'quiz_attempts.csv', 'quiz_history.csv']
        existing_files = [f for f in data_files if os.path.exists(os.path.join(data_dir, f))]
        
        if not existing_files:
            self.show_message("没有找到数据文件，无需备份", "warning")
            return
        
        # 选择备份目录
        backup_dir = filedialog.askdirectory(title="选择备份目录")
        if not backup_dir:
            return
        
        # 创建备份
        try:
            # 备份文件名包含时间戳
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # 备份每个文件
            for filename in existing_files:
                source_path = os.path.join(data_dir, filename)
                target_path = os.path.join(backup_dir, f"{filename[:-4]}_{timestamp}.csv")
                
                # 复制文件
                import shutil
                shutil.copy2(source_path, target_path)
            
            # 显示成功消息
            self.show_message(f"成功备份 {len(existing_files)} 个数据文件到 {backup_dir}", "info")
            
        except Exception as e:
            self.show_message(f"备份失败: {str(e)}", "error")
            logger.error(f"备份数据失败: {str(e)}")
    
    def _clear_data(self):
        """清空数据文件"""
        # 确认操作
        self.confirm_action(
            "确定要清空所有数据文件吗？此操作不可恢复！",
            self._confirm_clear_data
        )
    
    def _confirm_clear_data(self):
        """确认清空数据"""
        data_dir = get_data_dir()
        data_files = ['users.csv', 'questions.csv', 'quiz_attempts.csv', 'quiz_history.csv']
        
        deleted_count = 0
        
        for filename in data_files:
            file_path = os.path.join(data_dir, filename)
            
            if os.path.exists(file_path):
                try:
                    os.remove(file_path)
                    deleted_count += 1
                except Exception as e:
                    logger.error(f"删除文件 {filename} 失败: {str(e)}")
        
        # 显示结果
        if deleted_count > 0:
            self.show_message(f"已清空 {deleted_count} 个数据文件", "info")
        else:
            self.show_message("没有找到数据文件", "warning")
        
        # 更新统计
        self._update_data_stats()
    
    def on_show(self):
        """页面显示时的回调"""
        super().on_show()
        
        # 更新数据统计
        self._update_data_stats()
    
    def _merge_csv_files(self):
        """合并CSV文件"""
        try:
            # 显示文件选择对话框
            file_paths = filedialog.askopenfilenames(
                title="选择要合并的CSV文件",
                filetypes=[("CSV文件", "*.csv")],
                initialdir=get_data_dir()
            )
            
            if not file_paths:
                return
                
            # 确认合并操作
            files_str = "\n".join([os.path.basename(f) for f in file_paths])
            confirm = messagebox.askokcancel(
                "确认合并", 
                f"确定要合并以下文件吗？\n\n{files_str}\n\n合并后将创建一个新的CSV文件。"
            )
            
            if not confirm:
                return
                
            # 选择保存位置
            save_path = filedialog.asksaveasfilename(
                title="保存合并的CSV文件",
                defaultextension=".csv",
                filetypes=[("CSV文件", "*.csv")],
                initialdir=get_data_dir()
            )
            
            if not save_path:
                return
            
            # 展示进度对话框
            progress_window = tk.Toplevel(self)
            progress_window.title("合并进度")
            progress_window.geometry("500x200")
            progress_window.resizable(False, False)
            progress_window.transient(self)
            
            # 设置对话框位置居中
            progress_window.update_idletasks()
            width = progress_window.winfo_width()
            height = progress_window.winfo_height()
            x = (progress_window.winfo_screenwidth() // 2) - (width // 2)
            y = (progress_window.winfo_screenheight() // 2) - (height // 2)
            progress_window.geometry('{}x{}+{}+{}'.format(width, height, x, y))
            
            # 进度标签
            progress_label = ttk.Label(
                progress_window, 
                text="正在合并文件...",
                font=("Helvetica", 10)
            )
            progress_label.pack(pady=(20, 10))
            
            # 进度条
            progress_var = tk.DoubleVar(value=0)
            progress_bar = ttk.Progressbar(
                progress_window,
                orient=tk.HORIZONTAL,
                variable=progress_var,
                mode='determinate',
                length=300
            )
            progress_bar.pack(pady=10)
            
            # 当前处理文件标签
            file_label = ttk.Label(
                progress_window, 
                text="",
                font=("Helvetica", 9)
            )
            file_label.pack(pady=5)
            
            # 开始合并线程
            threading.Thread(
                target=self._do_merge_csv_files,
                args=(file_paths, save_path, progress_var, progress_label, file_label, progress_window),
                daemon=True
            ).start()
            
        except Exception as e:
            logger.error(f"合并CSV文件失败: {str(e)}")
            messagebox.showerror("错误", f"合并CSV文件失败: {str(e)}")
            
    def _do_merge_csv_files(self, file_paths, save_path, progress_var, progress_label, file_label, progress_window):
        """执行CSV文件合并"""
        try:
            # 读取所有CSV文件
            all_dfs = []
            total_files = len(file_paths)
            
            for i, file_path in enumerate(file_paths):
                # 更新进度
                progress_var.set((i / total_files) * 90)  # 保留最后10%用于写入文件
                file_name = os.path.basename(file_path)
                file_label.config(text=f"处理文件: {file_name}")
                
                # 读取CSV文件
                df = pd.read_csv(file_path)
                
                # 检查是否有重复列，添加文件名作为前缀以区分
                if len(all_dfs) > 0:
                    # 检查同名列但不同结构的情况
                    common_cols = set(df.columns) & set(all_dfs[0].columns)
                    if common_cols and len(common_cols) != len(df.columns):
                        # 为非公共列添加前缀
                        prefix = os.path.splitext(file_name)[0] + "_"
                        rename_cols = {col: prefix + col for col in df.columns if col not in common_cols}
                        df = df.rename(columns=rename_cols)
                
                all_dfs.append(df)
                
                # 简单延迟，让UI有时间更新
                time.sleep(0.1)
            
            # 检查合并类型
            if len(file_paths) == 4:  # 判断是否为标准4个文件
                # 检查文件名是否符合标准数据集模式
                filenames = [os.path.basename(f).lower() for f in file_paths]
                is_standard_set = (
                    any('user' in f for f in filenames) and
                    any('question' in f for f in filenames) and
                    any('attempt' in f or 'quiz' in f for f in filenames) and
                    any('history' in f for f in filenames)
                )
                
                if is_standard_set:
                    # 提示用户选择合并类型
                    progress_window.withdraw()  # 先隐藏进度窗口
                    merge_message = """检测到标准数据集格式（用户、题目、答题、历史）。

您想要：
- 点击"是"创建训练数据集（推荐）
- 点击"否"简单合并所有文件"""
                    merge_type = messagebox.askquestion("选择合并方式", merge_message)
                    progress_window.deiconify()  # 重新显示进度窗口
                    
                    if merge_type == "yes":
                        # 创建训练数据集
                        progress_label.config(text="正在创建训练数据集...")
                        merged_df = self._create_training_dataset(all_dfs, filenames)
                    else:
                        # 简单合并
                        progress_label.config(text="正在简单合并所有文件...")
                        merged_df = pd.concat(all_dfs, axis=0, ignore_index=True)
                else:
                    # 简单合并
                    progress_label.config(text="正在合并所有文件...")
                    merged_df = pd.concat(all_dfs, axis=0, ignore_index=True)
            else:
                # 简单合并
                progress_label.config(text="正在合并所有文件...")
                merged_df = pd.concat(all_dfs, axis=0, ignore_index=True)
            
            # 保存合并后的文件
            progress_label.config(text="正在保存合并文件...")
            progress_var.set(95)
            merged_df.to_csv(save_path, index=False)
            
            # 完成
            progress_var.set(100)
            progress_label.config(text="合并完成！")
            file_label.config(text=f"文件已保存: {os.path.basename(save_path)}")
            
            # 关闭按钮
            ttk.Button(
                progress_window,
                text="关闭",
                command=progress_window.destroy
            ).pack(pady=10)
            
            # 刷新文件列表
            self._refresh_file_list()
            
            # 显示成功消息
            messagebox.showinfo("成功", f"文件合并成功！\n已保存到: {save_path}")
            
        except Exception as e:
            logger.error(f"执行CSV文件合并失败: {str(e)}")
            progress_label.config(text=f"合并失败: {str(e)}")
            
            # 关闭按钮
            ttk.Button(
                progress_window,
                text="关闭",
                command=progress_window.destroy
            ).pack(pady=10)
            
            # 显示错误消息
            messagebox.showerror("错误", f"合并CSV文件失败: {str(e)}")
            
    def _create_training_dataset(self, dataframes, filenames):
        """创建用于训练的数据集"""
        # 找到各个数据集
        user_df = next((df for df, name in zip(dataframes, filenames) if 'user' in name.lower()), None)
        question_df = next((df for df, name in zip(dataframes, filenames) if 'question' in name.lower()), None)
        attempt_df = next((df for df, name in zip(dataframes, filenames) 
                          if 'attempt' in name.lower() or 'quiz' in name.lower()), None)
        history_df = next((df for df, name in zip(dataframes, filenames) if 'history' in name.lower()), None)
        
        # 合并相关字段创建训练数据集
        if attempt_df is not None:
            result_df = attempt_df.copy()
            
            # 添加用户特征
            if user_df is not None and 'user_id' in result_df.columns and 'user_id' in user_df.columns:
                # 选择要包含的用户列
                user_cols = [col for col in user_df.columns 
                            if col not in ['user_id'] or col == 'user_id']
                result_df = pd.merge(
                    result_df, 
                    user_df[user_cols], 
                    on='user_id', 
                    how='left'
                )
            
            # 添加题目特征
            if question_df is not None and 'question_id' in result_df.columns and 'question_id' in question_df.columns:
                # 选择要包含的题目列
                question_cols = [col for col in question_df.columns 
                                if col not in ['question_id'] or col == 'question_id']
                result_df = pd.merge(
                    result_df, 
                    question_df[question_cols], 
                    on='question_id', 
                    how='left'
                )
            
            # 添加历史特征
            if history_df is not None and 'user_id' in result_df.columns and 'user_id' in history_df.columns:
                # 计算每个用户的历史统计特征
                history_stats = history_df.groupby('user_id').agg({
                    'attempts_count': 'mean',
                    'correct_count': 'mean',
                    'score': 'mean',
                    'learning_time': 'sum'
                }).reset_index()
                
                # 重命名列以避免冲突
                history_stats = history_stats.rename(columns={
                    'attempts_count': 'hist_attempts',
                    'correct_count': 'hist_correct',
                    'score': 'hist_score',
                    'learning_time': 'hist_learning_time'
                })
                
                # 合并到结果中
                result_df = pd.merge(
                    result_df,
                    history_stats,
                    on='user_id',
                    how='left'
                )
            
            return result_df
        else:
            # 如果找不到答题记录，则简单合并所有DataFrame
            return pd.concat(dataframes, axis=0, ignore_index=True)

    def _refresh_file_list(self):
        """刷新数据文件列表"""
        # 清空列表
        self.files_listbox.delete(0, tk.END)
        
        # 获取数据目录
        data_dir = get_data_dir()
        
        # 列出所有CSV文件
        try:
            files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
            files.sort()  # 按文件名排序
            
            # 添加到列表框
            for file in files:
                self.files_listbox.insert(tk.END, file)
            
            # 显示文件数量
            count_text = f"找到 {len(files)} 个数据文件"
            self.controller.update_status(count_text) if self.controller else None
            
        except Exception as e:
            logger.error(f"读取数据目录失败: {str(e)}")
            self.controller.update_status(f"读取数据目录失败: {str(e)}") if self.controller else None
    
    def _on_file_select(self, event):
        """文件选择事件处理"""
        # 获取选中项
        selection = self.files_listbox.curselection()
        if not selection:
            return
        
        # 获取选中的文件名
        file_name = self.files_listbox.get(selection[0])
        file_path = os.path.join(get_data_dir(), file_name)
        
        # 预览文件内容
        self._preview_file(file_path)
    
    def _preview_file(self, file_path):
        """预览文件内容"""
        try:
            # 清空预览框
            self.preview_text.delete("1.0", tk.END)
            
            # 读取CSV文件
            df = pd.read_csv(file_path)
            
            # 构建预览信息
            file_info = f"文件: {os.path.basename(file_path)}\n"
            file_info += f"大小: {os.path.getsize(file_path) / 1024:.2f} KB\n"
            file_info += f"行数: {len(df)}\n"
            file_info += f"列数: {len(df.columns)}\n\n"
            
            # 列信息
            file_info += "列名:\n"
            for col in df.columns:
                file_info += f"- {col}\n"
            
            file_info += "\n数据预览 (前5行):\n"
            
            # 添加预览信息
            self.preview_text.insert("1.0", file_info)
            
            # 添加数据预览
            self.preview_text.insert(tk.END, df.head().to_string())
            
        except Exception as e:
            logger.error(f"预览文件失败: {str(e)}")
            self.preview_text.insert("1.0", f"预览文件失败: {str(e)}")
    
    def _export_data(self):
        """导出数据文件"""
        # 获取选中的文件
        selection = self.files_listbox.curselection()
        if not selection:
            messagebox.showerror("错误", "请先选择要导出的文件")
            return
        
        # 获取选中的文件名
        file_name = self.files_listbox.get(selection[0])
        source_path = os.path.join(get_data_dir(), file_name)
        
        # 选择导出目标
        target_path = filedialog.asksaveasfilename(
            title="导出数据文件",
            initialfile=file_name,
            defaultextension=".csv",
            filetypes=[("CSV文件", "*.csv"), ("所有文件", "*.*")]
        )
        
        if not target_path:
            return
        
        try:
            # 复制文件
            import shutil
            shutil.copy2(source_path, target_path)
            
            # 显示成功消息
            messagebox.showinfo("成功", f"文件导出成功: {target_path}")
            
        except Exception as e:
            logger.error(f"导出文件失败: {str(e)}")
            messagebox.showerror("错误", f"导出文件失败: {str(e)}")


# 测试代码
if __name__ == "__main__":
    root = tk.Tk()
    root.title("数据生成测试")
    root.geometry("1000x700")
    
    # 模拟控制器
    class MockController:
        def __init__(self):
            self.shared_data = {}
            self.is_generating_data = False
        
        def show_message(self, message, message_type):
            print(f"消息 ({message_type}): {message}")
        
        def confirm_action(self, message, callback):
            print(f"确认: {message}")
            if callable(callback):
                callback()
        
        def update_status(self, status):
            print(f"状态: {status}")
        
        def update_progress(self, value, task):
            print(f"进度: {value}%, 任务: {task}")
    
    # 创建数据页面
    page = DataPage(root, MockController())
    
    # 显示窗口
    root.mainloop() 