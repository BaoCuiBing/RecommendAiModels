"""
性能图表模块
提供模型训练性能图表和可视化功能
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.figure import Figure
from typing import List, Dict, Any, Optional, Tuple, Union
import logging

# 配置日志
logger = logging.getLogger("performance_charts")

def create_accuracy_chart(figure: Figure, data: List[Dict[str, Any]], metric: str = "accuracy") -> None:
    """
    创建准确率/指标曲线图
    
    Args:
        figure: matplotlib图表对象
        data: 性能数据列表
        metric: 要绘制的指标
    """
    if not data:
        logger.warning("没有数据用于创建图表")
        return
    
    # 清除图表
    figure.clear()
    ax = figure.add_subplot(111)
    
    # 提取数据
    iterations = [point.get("iteration", i+1) for i, point in enumerate(data)]
    
    # 映射指标名称到数据键
    metric_map = {
        "准确率": "accuracy",
        "损失函数": "loss",
        "精确率": "precision",
        "召回率": "recall",
        "F1分数": "f1",
        "AUC": "auc",
        "训练时间": "time"
    }
    
    metric_key = metric_map.get(metric, metric)
    values = [point.get(metric_key, 0) for point in data]
    
    # 绘制曲线
    ax.plot(iterations, values, 'b-', linewidth=1.5, marker='o', markersize=4)
    
    # 设置图表属性
    ax.set_title(f"{metric}趋势")
    ax.set_xlabel("迭代次数")
    ax.set_ylabel(metric)
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # 自动调整布局
    figure.tight_layout()

def create_metrics_comparison(figure: Figure, data: Dict[str, float]) -> None:
    """
    创建多指标对比柱状图
    
    Args:
        figure: matplotlib图表对象
        data: 指标数据字典
    """
    if not data:
        logger.warning("没有数据用于创建对比图表")
        return
    
    # 清除图表
    figure.clear()
    ax = figure.add_subplot(111)
    
    # 绘制柱状图
    metrics = list(data.keys())
    values = list(data.values())
    
    bars = ax.bar(metrics, values, color='skyblue', alpha=0.7)
    
    # 在柱子上显示数值
    for bar, value in zip(bars, values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{value:.4f}',
                ha='center', va='bottom', rotation=0, fontsize=8)
    
    # 设置图表属性
    ax.set_title("性能指标对比")
    ax.set_ylabel("值")
    ax.set_ylim(0, 1.0)
    ax.grid(True, linestyle='--', alpha=0.7, axis='y')
    
    # 自动调整布局
    figure.tight_layout()

def create_correlation_heatmap(figure: Figure, data: List[Dict[str, Any]]) -> None:
    """
    创建相关性热力图
    
    Args:
        figure: matplotlib图表对象
        data: 性能数据列表
    """
    if not data or len(data) < 5:  # 需要足够的数据点
        logger.warning("没有足够的数据用于创建相关性图表")
        return
    
    # 清除图表
    figure.clear()
    ax = figure.add_subplot(111)
    
    # 创建数据框
    df = pd.DataFrame(data)
    
    # 选择要分析的列
    columns = ["accuracy", "precision", "recall", "f1", "auc"]
    available_columns = [col for col in columns if col in df.columns]
    
    if not available_columns:
        logger.warning("没有有效的列用于相关性分析")
        return
    
    corr_df = df[available_columns].corr()
    
    # 绘制热力图
    sns.heatmap(corr_df, annot=True, cmap='coolwarm', linewidths=.5, ax=ax, fmt='.2f', cbar_kws={"shrink": .8})
    
    # 设置图表属性
    ax.set_title("指标相关性分析")
    
    # 自动调整布局
    figure.tight_layout()

def create_distribution_chart(figure: Figure, data: List[Dict[str, Any]], key: str = "difficulty") -> None:
    """
    创建分布图表
    
    Args:
        figure: matplotlib图表对象
        data: 数据列表
        key: 要分析分布的键
    """
    if not data:
        logger.warning("没有数据用于创建分布图表")
        return
    
    # 清除图表
    figure.clear()
    ax = figure.add_subplot(111)
    
    # 提取数据
    values = [item.get(key, 0) for item in data if key in item]
    
    if not values:
        logger.warning(f"没有'{key}'键的数据用于创建分布图表")
        return
    
    # 绘制直方图
    ax.hist(values, bins=10, alpha=0.7, color='skyblue', edgecolor='black')
    
    # 设置图表属性
    ax.set_title(f"{key}分布")
    ax.set_xlabel(key)
    ax.set_ylabel("频率")
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # 自动调整布局
    figure.tight_layout()

def create_time_series_chart(figure: Figure, data: List[Dict[str, Any]], 
                           x_key: str = "timestamp", y_key: str = "accuracy") -> None:
    """
    创建时间序列图表
    
    Args:
        figure: matplotlib图表对象
        data: 数据列表
        x_key: 时间或序列键
        y_key: 值键
    """
    if not data:
        logger.warning("没有数据用于创建时间序列图表")
        return
    
    # 清除图表
    figure.clear()
    ax = figure.add_subplot(111)
    
    # 提取数据
    x_values = [item.get(x_key, i) for i, item in enumerate(data) if y_key in item]
    y_values = [item.get(y_key, 0) for item in data if y_key in item]
    
    if not x_values or not y_values:
        logger.warning(f"没有'{x_key}'或'{y_key}'键的数据用于创建时间序列图表")
        return
    
    # 绘制线图
    ax.plot(x_values, y_values, 'b-', linewidth=1.5, marker='o', markersize=4)
    
    # 设置图表属性
    ax.set_title(f"{y_key}随{x_key}变化")
    ax.set_xlabel(x_key)
    ax.set_ylabel(y_key)
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # 如果x轴标签太多，旋转它们
    if len(x_values) > 5:
        plt.xticks(rotation=45)
    
    # 自动调整布局
    figure.tight_layout()

def create_radar_chart(figure: Figure, data: Dict[str, float], title: str = "性能雷达图") -> None:
    """
    创建雷达图
    
    Args:
        figure: matplotlib图表对象
        data: 数据字典 {指标名: 值}
        title: 图表标题
    """
    if not data:
        logger.warning("没有数据用于创建雷达图")
        return
    
    # 清除图表
    figure.clear()
    ax = figure.add_subplot(111, polar=True)
    
    # 准备数据
    categories = list(data.keys())
    values = list(data.values())
    
    # 计算角度
    N = len(categories)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]  # 闭合雷达图
    
    # 添加数据
    values += values[:1]  # 闭合数据
    
    # 绘制雷达图
    ax.plot(angles, values, 'b-', linewidth=1.5)
    ax.fill(angles, values, 'skyblue', alpha=0.4)
    
    # 设置刻度和标签
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories)
    
    # 设置y轴限制
    ax.set_ylim(0, max(values) * 1.1)
    
    # 设置图表标题
    ax.set_title(title)
    
    # 自动调整布局
    figure.tight_layout()

# 创建组合图表，显示多种指标
def create_combined_metrics_chart(figure: Figure, data: List[Dict[str, Any]]) -> None:
    """
    创建组合指标图表，在一个图中显示多个指标
    
    Args:
        figure: matplotlib图表对象
        data: 性能数据列表
    """
    if not data:
        logger.warning("没有数据用于创建组合指标图表")
        return
    
    # 清除图表
    figure.clear()
    ax = figure.add_subplot(111)
    
    # 提取数据
    iterations = [point.get("iteration", i+1) for i, point in enumerate(data)]
    
    # 准备要绘制的指标
    metrics = {
        "accuracy": {"color": "blue", "marker": "o", "label": "准确率"},
        "precision": {"color": "green", "marker": "s", "label": "精确率"},
        "recall": {"color": "red", "marker": "^", "label": "召回率"},
        "f1": {"color": "purple", "marker": "d", "label": "F1分数"}
    }
    
    # 绘制多条线
    for metric, props in metrics.items():
        if metric in data[0]:
            values = [point.get(metric, 0) for point in data]
            ax.plot(iterations, values, 
                   color=props["color"], 
                   marker=props["marker"], 
                   markersize=4,
                   linewidth=1.5,
                   label=props["label"])
    
    # 设置图表属性
    ax.set_title("多指标性能趋势")
    ax.set_xlabel("迭代次数")
    ax.set_ylabel("值")
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend(loc="lower right")
    
    # 自动调整布局
    figure.tight_layout() 