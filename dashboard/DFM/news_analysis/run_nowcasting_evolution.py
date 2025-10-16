# -*- coding: utf-8 -*-
import sys
import os
import io

# 修复Windows控制台编码问题
if sys.platform == 'win32':
    # 设置标准输出为UTF-8编码
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# Get the directory of the current file (run_nowcasting_evolution.py, which is in news_analysis folder)
current_script_dir = os.path.dirname(os.path.abspath(__file__))
# Get the DFM directory (parent of news_analysis)
dfm_directory = os.path.abspath(os.path.join(current_script_dir, '..'))
# Get the project root directory (parent of DFM)
project_root_dir = os.path.abspath(os.path.join(dfm_directory, '..', '..'))
# Get the dashboard directory (parent of DFM)
dashboard_actual_dir = os.path.abspath(os.path.join(dfm_directory, '..'))

# Add project root directory to sys.path
if project_root_dir not in sys.path:
    sys.path.insert(0, project_root_dir)
    print(f"[run_nowcasting_evolution] Added project_root ('{project_root_dir}') to sys.path for local modules.")

import train_model.DynamicFactorModel as DynamicFactorModel
import train_model.DiscreteKalmanFilter as DiscreteKalmanFilter
sys.modules['DynamicFactorModel'] = DynamicFactorModel
sys.modules['DiscreteKalmanFilter'] = DiscreteKalmanFilter
print("[run_nowcasting_evolution] 模块别名已设置，可兼容旧的joblib文件")

# Add DFM directory to sys.path for potential imports from DFM or other subdirectories like news_analysis
if dfm_directory not in sys.path:
    sys.path.insert(0, dfm_directory)
    # print(f"[run_nowcasting_evolution] Added dfm_directory ('{dfm_directory}') to sys.path.")

# Add dashboard directory to sys.path
if dashboard_actual_dir not in sys.path:
    sys.path.insert(0, dashboard_actual_dir)

# 处理配置文件导入（之前有相关逻辑）
# 使用本地配置，项目已完全迁移到 dashboard 体系

# 从重构后的dashboard结构导入
from train_model.DynamicFactorModel import DFMEMResultsWrapper
from train_model.DiscreteKalmanFilter import KalmanFilter, KalmanFilterResultsWrapper
from data_prep.data_preparation import prepare_data, load_mappings
from data_prep.data_preparation import apply_stationarity_transforms
print("[run_nowcasting_evolution] 成功从本地模块路径导入")


"""
run_nowcasting_evolution.py

加载最终的 DFM 模型结果，重新运行 Kalman 滤波器以获取中间状态，
计算特定目标月份的 Nowcast 演变和新闻贡献，并生成可视化图表。
"""

import pandas as pd
import numpy as np
import pickle
import joblib # 用于加载模型对象
import tempfile
from datetime import datetime, timedelta
import plotly.graph_objects as go # <<< 新增 Plotly
from plotly.subplots import make_subplots # <<< 新增 Plotly (如果需要)
import plotly.io as pio # <<< 新增 Plotly
import seaborn as sns # No longer needed (comment out or remove if truly unused)
import argparse
from typing import Optional, Dict, List, Tuple
from collections import defaultdict
import unicodedata

# 自身的 Nowcasting 类
# 假设 DFM_Nowcasting.py 与此脚本在同一目录 (news_analysis/)
from DFM_Nowcasting import DFMNowcastModel

# 导入DFM配置模块
from dashboard.DFM import config as dfm_config
print("[run_nowcasting_evolution] 成功导入DFM配置模块")

# 创建配置包装类以保持兼容性
class ConfigWrapper:
    def __init__(self):
        # 从DFM配置文件导入所需的配置
        self.TARGET_VARIABLE = dfm_config.DataDefaults.TARGET_VARIABLE
        self.TARGET_FREQ = 'M'  # 月度频率
        self.SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
        self.NOWCAST_MODEL_INPUT_DIR = 'models'
        self.NOWCAST_MODEL_FILENAME = 'final_dfm_model.joblib'
        self.NOWCAST_METADATA_FILENAME = 'final_model_metadata.pkl'
        self.DEFAULT_MODEL_FREQUENCY_FOR_NEWS = dfm_config.NewsAnalysisDefaults.DEFAULT_MODEL_FREQUENCY
        self.NOWCAST_EVOLUTION_OUTPUT_DIR = tempfile.mkdtemp(prefix="nowcast_evolution_")

config = ConfigWrapper()
print("[run_nowcasting_evolution] 使用配置包装器加载DFM配置")

# 已移除 Matplotlib 相关配置，改用 Plotly

# --- 配置 --- (从 config.py 加载)
DEFAULT_TUNING_OUTPUT_DIR = os.path.join(config.SCRIPT_DIR, config.NOWCAST_MODEL_INPUT_DIR)
MODEL_FILENAME = config.NOWCAST_MODEL_FILENAME
METADATA_FILENAME = config.NOWCAST_METADATA_FILENAME
TARGET_VARIABLE = config.TARGET_VARIABLE
TARGET_FREQ = config.TARGET_FREQ

def get_industry_group(var_name: str, mapping: Dict[str, str]) -> str:
    norm_name = unicodedata.normalize('NFKC', str(var_name)).strip().lower()
    return mapping.get(norm_name, '其他未分类')

def plot_news_decomposition(
    input_dir: str,
    # output_file: str, # 将是 .html 文件 (修改：此参数将作为基础名)
    base_output_filename: str, # 例如：news_analysis_plot_backend (不含扩展名)
    output_dir: str, # 输出目录
    plot_start_date: Optional[str] = None,
    plot_end_date: Optional[str] = None,
    target_variable_name: str = "目标变量"
) -> Dict[str, Optional[str]]: # 返回包含两个图表路径的字典
    print(f"  [Plotting] 开始生成新闻分解图和演变图 (Plotly) 到目录: {output_dir}")
    print(f"  [Plotting] 使用基础文件名: {base_output_filename}")

    output_paths = {
        "evolution_plot_path": None,
        "decomposition_plot_path": None
    }

    evolution_file_path = os.path.join(output_dir, f"{base_output_filename}_evo.html")
    decomposition_file_path = os.path.join(output_dir, f"{base_output_filename}_decomp.html")

    evolution_file = os.path.join(input_dir, 'nowcast_evolution_data_T.csv')
    news_file = os.path.join(input_dir, 'news_decomposition_grouped.csv')

    if not os.path.exists(evolution_file):
        raise FileNotFoundError(f"绘图所需的 Nowcast 演变文件未找到: {evolution_file}")
    if not os.path.exists(news_file):
        # 如果新闻文件不存在，我们仍然可以绘制演变图
        print(f"  [Plotting] 新闻分解文件未找到: {news_file}. 只绘制 Nowcast 演变图。")
        news_df = pd.DataFrame() # 创建空的 DataFrame
    else:
        try:
            news_df = pd.read_csv(news_file, index_col=0, parse_dates=True, encoding='utf-8')
        except UnicodeDecodeError:
            print(f"  [Plotting] UTF-8编码失败，尝试使用gbk编码读取新闻分解CSV...")
            news_df = pd.read_csv(news_file, index_col=0, parse_dates=True, encoding='gbk')


    try:
        nowcast_df = pd.read_csv(evolution_file, index_col=0, parse_dates=True)
    except Exception as e_load_evo:
        # If evolution_file is missing or unreadable, treat as empty for plotting
        print(f"  [Plotting] 警告: 加载 Nowcast 演变 CSV 数据时出错: {e_load_evo}. 将尝试生成空演变图。")
        nowcast_df = pd.DataFrame()

    if not nowcast_df.empty and 'nowcast_descaled' in nowcast_df.columns:
         nowcast_col_name = 'nowcast_descaled'
    elif not nowcast_df.empty and 'nowcast_orig' in nowcast_df.columns:
         nowcast_col_name = 'nowcast_orig'
    else:
         # 如果 nowcast_df 为空，或必要的列不存在，设置一个默认值以避免后续错误
         nowcast_col_name = 'nowcast_descaled' # Fallback, will be empty if df is empty
         if not nowcast_df.empty: # 仅在df非空但列缺失时告警
            print(f"  [Plotting] 警告: 在 Nowcast 演变数据中找不到 'nowcast_descaled' 或 'nowcast_orig' 列。演变图可能不正确。")

    plot_news = not news_df.empty
    
    # 目标月份字符串，用于标题
    min_date_for_title = None
    if not nowcast_df.empty and isinstance(nowcast_df.index, pd.DatetimeIndex) and not nowcast_df.index.empty:
        min_date_for_title = nowcast_df.index.min()
    elif plot_news and not news_df.empty and isinstance(news_df.index, pd.DatetimeIndex) and not news_df.index.empty:
        min_date_for_title = news_df.index.min()
    
    if min_date_for_title is not None:
        inferred_month_str = min_date_for_title.strftime('%Y-%m')
    else:
        inferred_month_str = "目标月份" # 后备标题


    # 日期过滤 - 修复：考虑两个数据集的时间范围以确保时间轴一致
    # min_allowable_date = pd.Timestamp('1900-01-01')
    # max_allowable_date = pd.Timestamp('2200-01-01')

    default_start_dt = None
    default_end_dt = None

    # 收集所有可用数据的时间范围
    all_start_times = []
    all_end_times = []

    # 从 nowcast_df 获取时间范围
    if not nowcast_df.empty and isinstance(nowcast_df.index, pd.DatetimeIndex) and not nowcast_df.index.empty:
        nowcast_min = nowcast_df.index.min()
        nowcast_max = nowcast_df.index.max()
        if pd.notna(nowcast_min):
            all_start_times.append(nowcast_min)
        if pd.notna(nowcast_max):
            all_end_times.append(nowcast_max)

    # 从 news_df 获取时间范围
    if plot_news and not news_df.empty and isinstance(news_df.index, pd.DatetimeIndex) and not news_df.index.empty:
        news_min = news_df.index.min()
        news_max = news_df.index.max()
        if pd.notna(news_min):
            all_start_times.append(news_min)
        if pd.notna(news_max):
            all_end_times.append(news_max)

    # 使用两个数据集的交集时间范围以确保时间轴一致
    if all_start_times and all_end_times:
        default_start_dt = max(all_start_times)  # 使用较晚的开始时间
        default_end_dt = min(all_end_times)      # 使用较早的结束时间
        print(f"  [Plotting] 基于两个数据集的交集确定时间范围: {default_start_dt.strftime('%Y-%m-%d')} 到 {default_end_dt.strftime('%Y-%m-%d')}")

    # 如果经过上述步骤后，默认日期仍然是None (例如两个数据集都为空)
    if default_start_dt is None or default_end_dt is None:
        raise ValueError('[Plotting] 无法从数据集确定默认的绘图时间范围')

    # 确保开始时间不晚于结束时间
    if default_start_dt > default_end_dt:
        print(f"  [Plotting] 警告: 交集时间范围无效 (开始时间晚于结束时间)，使用并集范围")
        if all_start_times and all_end_times:
            default_start_dt = min(all_start_times)  # 使用最早的开始时间
            default_end_dt = max(all_end_times)      # 使用最晚的结束时间
            print(f"  [Plotting] 改用并集时间范围: {default_start_dt.strftime('%Y-%m-%d')} 到 {default_end_dt.strftime('%Y-%m-%d')}")

    start_dt = pd.to_datetime(plot_start_date) if plot_start_date else default_start_dt
    end_dt = pd.to_datetime(plot_end_date) if plot_end_date else default_end_dt

    # 再次确保 start_dt 和 end_dt 是有效的 Timestamp 对象，并且 start_dt <= end_dt
    valid_dates = True
    if not isinstance(start_dt, pd.Timestamp) or pd.isna(start_dt):
        print(f"  [Plotting] 警告: 解析后的 start_dt ('{plot_start_date}') 无效，使用默认值: {default_start_dt.strftime('%Y-%m-%d')}")
        start_dt = default_start_dt
        valid_dates = False
    if not isinstance(end_dt, pd.Timestamp) or pd.isna(end_dt):
        print(f"  [Plotting] 警告: 解析后的 end_dt ('{plot_end_date}') 无效，使用默认值: {default_end_dt.strftime('%Y-%m-%d')}")
        end_dt = default_end_dt
        valid_dates = False

    if start_dt > end_dt:
        print(f"  [Plotting] 警告: start_dt ({start_dt.strftime('%Y-%m-%d')}) 晚于 end_dt ({end_dt.strftime('%Y-%m-%d')})。将交换它们或使用默认范围。")
        # 简单处理：如果无效，都用默认值，或者只交换
        start_dt, end_dt = min(default_start_dt, default_end_dt), max(default_start_dt, default_end_dt) 
        print(f"  [Plotting] 日期范围已重置为: {start_dt.strftime('%Y-%m-%d')} 到 {end_dt.strftime('%Y-%m-%d')}")
        valid_dates = False

    print(f"  [Plotting] 使用的绘图日期范围: {start_dt.strftime('%Y-%m-%d')} 到 {end_dt.strftime('%Y-%m-%d')}")
    print(f"  [Plotting] start_dt 类型: {type(start_dt)}, end_dt 类型: {type(end_dt)}")
    
    nowcast_plot_data = nowcast_df[(nowcast_df.index >= start_dt) & (nowcast_df.index <= end_dt)]
    if plot_news:
        news_plot_data = news_df[(news_df.index >= start_dt) & (news_df.index <= end_dt)]
        if news_plot_data.empty: plot_news = False # 如果过滤后为空，则不绘制新闻
    else:
        news_plot_data = pd.DataFrame()

    if nowcast_plot_data.empty:
         # raise ValueError(f"在指定的日期范围 ({start_dt.strftime('%Y-%m-%d')} 到 {end_dt.strftime('%Y-%m-%d')}) 内没有找到 Nowcast 演变数据。")
         # 确保这里的 start_dt 和 end_dt 是有效的 Timestamp 对象才能调用 strftime
         # 这个打印语句在之前的日志中导致了 AttributeError，因为那时的 start_dt/end_dt 可能是 float
         # 现在的逻辑应该能确保它们是 Timestamp
         print(f"  [Plotting] 警告: 在指定的日期范围 ({start_dt.strftime('%Y-%m-%d')} 到 {end_dt.strftime('%Y-%m-%d')}) 内没有找到 Nowcast 演变数据。演变图可能为空。")
         # 即使数据为空，也尝试生成一个空的演变图，而不是抛出错误导致后续分解图也不生成

    fig_evo = go.Figure()

    if not nowcast_plot_data.empty and nowcast_col_name in nowcast_plot_data.columns:
        fig_evo.add_trace(go.Scatter(
            x=nowcast_plot_data.index,
            y=nowcast_plot_data[nowcast_col_name],
            mode='lines+markers+text',
            name=f'Nowcast ({target_variable_name})',
            line=dict(color='black', width=2),
            marker=dict(size=7),
            text=[f'{val:.2f}' for val in nowcast_plot_data[nowcast_col_name]],
            textposition="top center",
            textfont=dict(size=10, color='black')
        ))
        title_text_evo = f"{inferred_month_str} {target_variable_name} Nowcast 演变"
    else:
        fig_evo.add_annotation(text="没有可用的 Nowcast 演变数据显示。", showarrow=False, font=dict(size=16))
        title_text_evo = f"{inferred_month_str} {target_variable_name} Nowcast 演变 (无数据)"

    fig_evo.update_layout(
        title=dict(text=title_text_evo, font=dict(size=16, color='#333333'), x=0.5, xanchor='center'),
        xaxis_title="信息截止日期 (Vintage t)",
        yaxis_title=f"{target_variable_name} (反标准化值)",
        legend_title_text="图例",
        legend=dict(orientation="h", yanchor="top", y=-0.2, xanchor="center", x=0.5),
        margin=dict(b=120, t=80, l=50, r=50),
        xaxis_showgrid=True, yaxis_showgrid=True,
        # gridwidth=1, gridcolor='LightGrey' # 在轴上设置
    )
    # 确保 start_dt 和 end_dt 是 Pandas Timestamp 类型
    range_start = pd.to_datetime(start_dt) if not isinstance(start_dt, pd.Timestamp) else start_dt
    range_end = pd.to_datetime(end_dt) if not isinstance(end_dt, pd.Timestamp) else end_dt

    print(f"  [Plotting] 演变图 range 设置: [{range_start}] 到 [{range_end}] (类型: {type(range_start)}, {type(range_end)})")

    unified_xaxis_config = dict(
        range=[range_start, range_end],  # 强制使用相同的时间范围
        tickformat="%m-%d",             # 统一的日期格式
        tickangle=0,                    # 统一的刻度角度
        showgrid=True,                  # 显示网格
        gridwidth=1,                    # 网格宽度
        gridcolor='LightGrey',          # 网格颜色
        dtick=7*24*60*60*1000,         # [HOT] 关键：固定7天间隔（毫秒）
        tick0=range_start,              # [HOT] 关键：从开始日期开始刻度
        tickmode='linear',              # [HOT] 关键：线性刻度模式
        nticks=10                       # 最大刻度数
    )

    fig_evo.update_xaxes(**unified_xaxis_config)
    fig_evo.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightGrey')

    try:
        html_content_evo = fig_evo.to_html(full_html=True, include_plotlyjs='cdn')
        with open(evolution_file_path, 'w', encoding='utf-8') as f:
            f.write(html_content_evo)
        print(f"  [Plotting] Plotly 演变图已保存到: {evolution_file_path}")
        output_paths["evolution_plot_path"] = evolution_file_path
    except Exception as e_save_evo:
        print(f"  [Plotting] 保存 Plotly 演变图为 HTML 时出错: {e_save_evo}")
        # 不抛出异常，允许继续生成分解图

    print(f"    Is empty: {news_df.empty}")
    if not news_df.empty:
        print(f"    Shape: {news_df.shape}")
        print(f"    Head:\n{news_df.head().to_string()}")
        print(f"    NaN counts per column:\n{news_df.isna().sum().to_string()}")
        print(f"    Data types:\n{news_df.dtypes.to_string()}")
    print(f"    Is empty: {news_plot_data.empty}")
    if not news_plot_data.empty:
        print(f"    Shape: {news_plot_data.shape}")
        print(f"    Head:\n{news_plot_data.head().to_string()}")
        print(f"    NaN counts per column:\n{news_plot_data.isna().sum().to_string()}")

    fig_decomp = go.Figure()
    if plot_news and not news_plot_data.empty: # <<< 增加对 news_plot_data 是否为空的检查
        group_cols = [col for col in news_plot_data.columns if col != 'total_news' and col != 'residual_news']
        if group_cols: # 确保有列可供绘制
            import plotly.express as px
            # 请确保这里的键名与 news_decomposition_grouped.csv 中的列名完全一致
            color_map = {
                    # 基于美联储Nowcasting截图的颜色推断 (具体类别对应需要根据实际数据调整)
                    "化学化工": "#B22222",  # Firebrick (类似截图中的深红色系)
                    "钢铁": "#4682B4",    # SteelBlue (类似截图中的蓝色系)
                    "煤炭": "#556B2F",    # DarkOliveGreen (类似截图中的绿色系)
                    "有色金属": "#DAA520",  # Goldenrod (类似截图中的黄色/橙色系)
                    "建材": "#708090",    # SlateGray (中性灰色系)
                    
                    "PMI": "#ADD8E6",      # LightBlue (浅蓝色)
                    "工业增加值": "#8FBC8F",  # DarkSeaGreen (一种较柔和的绿色)
                    "油气": "#FFD700",      # Gold (用一个较亮的黄色代表油气)
                    "电力": "#A9A9A9",      # DarkGray (灰色系)
                    "运输": "#BA55D3",      # MediumOrchid (紫色系)
                    
                    "汽车": "#87CEEB",      # SkyBlue (另一种浅蓝色)
                    "橡胶塑料": "#3CB371",  # MediumSeaGreen (中绿色)
                    "化纤": "#6A5ACD",      # SlateBlue (蓝紫色)
                    
                    # 默认颜色
                    "其他未分类": "#D3D3D3",  # LightGray
                    "residual_news": "#F0F0F0" # 非常浅的灰色，接近白色
                }
                # 为数据中存在但 color_map 中没有的列提供一个默认颜色列表循环使用
                default_colors_for_unmapped = px.colors.qualitative.Pastel 

                current_color_index = 0 # 用于从 default_colors_for_unmapped 中取色

                for col_name in group_cols:
                    fig_decomp.add_trace(go.Bar(
                        x=news_plot_data.index,
                        y=news_plot_data[col_name].fillna(0),
                        name=col_name,
                        marker_color=color_map.get(col_name, default_colors_for_unmapped[current_color_index % len(default_colors_for_unmapped)])
                    ))
                    current_color_index += 1

            fig_decomp.update_layout(barmode='relative')
            title_text_decomp = f"{inferred_month_str} {target_variable_name} 新闻贡献分解"
        else:
            fig_decomp.add_annotation(text="没有可供分解的新闻组数据。", showarrow=False, font=dict(size=16))
            title_text_decomp = f"{inferred_month_str} {target_variable_name} 新闻贡献分解 (无数据列)"
    else:
        # 如果没有新闻数据 (plot_news is False 或 news_plot_data 为空)
        fig_decomp.add_annotation(text="没有可用的新闻分解数据。", showarrow=False, font=dict(size=16))
        title_text_decomp = f"{inferred_month_str} {target_variable_name} 新闻贡献分解 (无数据)"

    fig_decomp.update_layout(
        title=dict(text=title_text_decomp, font=dict(size=16, color='#333333'), x=0.5, xanchor='center'),
        xaxis_title="信息截止日期 (Vintage t)",
        yaxis_title="对 Nowcast 的贡献 (反标准化值)", # Y轴含义可能需要根据实际计算调整
        legend_title_text="图例",
        legend=dict(orientation="h", yanchor="top", y=-0.2, xanchor="center", x=0.5),
        margin=dict(b=120, t=80, l=50, r=50),
        xaxis_showgrid=True, yaxis_showgrid=True,
    )
    print(f"  [Plotting] 分解图应用统一的 x 轴配置")

    fig_decomp.update_xaxes(**unified_xaxis_config)

    print(f"  [Plotting] === 时间轴设置验证 ===")
    print(f"  [Plotting] 演变图 x 轴范围: {fig_evo.layout.xaxis.range}")
    print(f"  [Plotting] 分解图 x 轴范围: {fig_decomp.layout.xaxis.range}")
    print(f"  [Plotting] 演变图 dtick: {fig_evo.layout.xaxis.dtick}")
    print(f"  [Plotting] 分解图 dtick: {fig_decomp.layout.xaxis.dtick}")
    print(f"  [Plotting] 演变图 tick0: {fig_evo.layout.xaxis.tick0}")
    print(f"  [Plotting] 分解图 tick0: {fig_decomp.layout.xaxis.tick0}")

    # 检查是否完全一致
    if (fig_evo.layout.xaxis.range == fig_decomp.layout.xaxis.range and
        fig_evo.layout.xaxis.dtick == fig_decomp.layout.xaxis.dtick and
        fig_evo.layout.xaxis.tick0 == fig_decomp.layout.xaxis.tick0):
        print(f"  [Plotting] [OK] 时间轴设置完全一致")
    else:
        print(f"  [Plotting] [ERROR] 时间轴设置不一致，需要检查")
    fig_decomp.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightGrey')

    try:
        html_content_decomp = fig_decomp.to_html(full_html=True, include_plotlyjs='cdn')
        with open(decomposition_file_path, 'w', encoding='utf-8') as f:
            f.write(html_content_decomp)
        print(f"  [Plotting] Plotly 分解图已保存到: {decomposition_file_path}")
        output_paths["decomposition_plot_path"] = decomposition_file_path
    except Exception as e_save_decomp:
        print(f"  [Plotting] 保存 Plotly 分解图为 HTML 时出错: {e_save_decomp}")
        import traceback
        traceback.print_exc()

    return output_paths # 返回两个图表的路径


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='运行 Nowcasting 演变与新闻贡献分析，并自动生成图表。')
    parser.add_argument(
        '--model_files_dir', # 参数名修改，以避免与之前的混淆，并强调其用途
        type=str,
        default=None, # 后端将传递此参数，命令行直接运行时可选（若不提供则使用 DEFAULT_TUNING_OUTPUT_DIR）
        help='指定 DFM 模型 (.joblib) 和元数据 (.pkl) 文件所在的输入目录。'
    )
    parser.add_argument(
        '--evolution_output_dir',
        type=str,
        default=config.NOWCAST_EVOLUTION_OUTPUT_DIR,
        help=f'Nowcasting 计算结果的输出目录 (默认为 config.py 中设置的值: {config.NOWCAST_EVOLUTION_OUTPUT_DIR})'
    )
    parser.add_argument(
        '--target_month',
        type=str,
        default=None,
        help='指定目标月份 (格式 YYYY-MM)。如果未指定，则使用 config.py 中的 NEWS_TARGET_MONTH 或自动确定。'
    )
    parser.add_argument(
        '--plot_output_file',
        type=str,
        default=None, # 将由 news_analysis_backend.py 传递
        help=f'输出图表的 HTML 文件路径 (默认为计算结果目录下的 news_analysis_plot.html)。' # <<< 修改帮助文本
    )
    parser.add_argument(
        '--plot_start_date',
        type=str,
        default=None,
        help='绘图的开始日期 (YYYY-MM-DD)。默认为分析周期的开始日期。'
    )
    parser.add_argument(
        '--plot_end_date',
        type=str,
        default=None,
        help='绘图的结束日期 (YYYY-MM-DD)。默认为分析周期的结束日期。'
    )
    args = parser.parse_args()
    EVOLUTION_OUTPUT_DIR = args.evolution_output_dir
    SPECIFIED_TARGET_MONTH_STR = args.target_month

    kf_results_rerun = None # <<< 确保这是 kf_results_rerun 在 __main__ 块中的唯一初始化位置

    if args.model_files_dir:
        TUNING_OUTPUT_DIR = args.model_files_dir
    else:
        TUNING_OUTPUT_DIR = DEFAULT_TUNING_OUTPUT_DIR # 使用脚本顶部的默认值

    
    print("--- 开始运行 Nowcast 演变与新闻贡献分析 ---")
    # ... (脚本其余主要逻辑保持不变) ...
    print(f"结果将保存到: {EVOLUTION_OUTPUT_DIR}")
    os.makedirs(EVOLUTION_OUTPUT_DIR, exist_ok=True)
    
    # 这确保了即使 joblib 文件中有 dashboard.DFM 的引用也能正确加载
    import sys
    if 'dashboard' not in sys.modules:
        # 创建一个假的 dashboard 模块
        import types
        dashboard = types.ModuleType('dashboard')
        sys.modules['dashboard'] = dashboard
        
        # 创建 dashboard.DFM
        dfm = types.ModuleType('DFM')
        dashboard.DFM = dfm
        sys.modules['dashboard.DFM'] = dfm
        
        # 设置必要的子模块别名
        sys.modules['dashboard.DFM.train_model'] = sys.modules.get('train_model')
        sys.modules['dashboard.DFM.data_prep'] = sys.modules.get('data_prep')
        
        print("  [Main Logic] 已设置 dashboard.DFM 模块别名以兼容 joblib 文件")
    
    print(f"--- 步骤 1: 从 '{TUNING_OUTPUT_DIR}' 加载模型结果和元数据 ---")
    model_load_path = os.path.join(TUNING_OUTPUT_DIR, MODEL_FILENAME)
    metadata_load_path = os.path.join(TUNING_OUTPUT_DIR, METADATA_FILENAME)
    try:
        print(f"  [Main Logic] 正在加载模型: {model_load_path}")
        final_dfm_results = joblib.load(model_load_path)
        print(f"  [Main Logic] 正在加载元数据: {metadata_load_path}")
        with open(metadata_load_path, 'rb') as f:
            metadata = pickle.load(f)
        print("  [Main Logic] 模型和元数据加载成功。")
        # ... (提取 A, Lambda, Q, R, x0, P0 等)
        # 确保提取 TARGET_VARIABLE_FROM_METADATA (如果元数据中有)
        TARGET_VARIABLE_FROM_METADATA = metadata.get('target_variable', TARGET_VARIABLE)
        print(f"  [Main Logic] 使用的目标变量 (来自元数据或配置): {TARGET_VARIABLE_FROM_METADATA}")

        z_for_rerun = metadata.get('processed_data_for_model')
        if z_for_rerun is None:
            raise ValueError("  [Main Logic] 缺少 'processed_data_for_model' 数据，无法重新运行卡尔曼滤波")
        if not isinstance(z_for_rerun.index, pd.DatetimeIndex):
            try:
                z_for_rerun.index = pd.to_datetime(z_for_rerun.index)
            except Exception as e_dt_z:
                raise ValueError("  [Main Logic] 无法将 z_for_rerun 的索引转换为 DatetimeIndex") from e_dt_z
            obs_names_from_lambda = Lambda_from_model.index.tolist()
            Lambda_np = Lambda_from_model.to_numpy()
        elif isinstance(Lambda_from_model, np.ndarray):
            Lambda_np = Lambda_from_model
            if best_variables_from_metadata and len(best_variables_from_metadata) == Lambda_np.shape[0]:
                obs_names_from_lambda = best_variables_from_metadata
            else:
                print("  [Main Logic] 错误: Lambda 是 NumPy 数组，但无法从元数据确定其对应的观测变量顺序。脚本将退出。")
                sys.exit(1)
        else:
            print(f"  [Main Logic] 错误: final_dfm_results.Lambda 的类型无法识别: {type(Lambda_from_model)}。脚本将退出。")
            sys.exit(1)

        n_factors = final_dfm_results.A.shape[0]
        state_names = [f"Factor_{i+1}" for i in range(n_factors)]

    except FileNotFoundError as e_fnf:
        print(f"错误: 找不到模型或元数据文件: {e_fnf}")
        sys.exit(1)
    except Exception as e_load:
        print(f"加载模型/元数据时发生错误: {e_load}")
        sys.exit(1)

    print("--- 步骤 2: 重新运行卡尔曼滤波以获取完整的中间状态和增益 ---")
    try:
        print(f"  [Main Logic] z_for_rerun 的索引类型: {type(z_for_rerun.index)}")

        n_time_kf = len(z_for_rerun.index)
        n_state_kf = final_dfm_results.A.shape[0]
        
        # 创建 U_for_kf: (n_time, 1) DataFrame of zeros
        # 使用一个不太可能与 z_for_rerun.columns冲突的列名
        dummy_col_name_u = 'dummy_ctrl_in'
        if dummy_col_name_u in z_for_rerun.columns: # 进一步避免列名冲突
            dummy_col_name_u = 'alt_dummy_ctrl_in'
        U_for_kf = pd.DataFrame(np.zeros((n_time_kf, 1)), index=z_for_rerun.index, columns=[dummy_col_name_u])

        # 创建 B_for_kf: (n_state, 1) np.array of zeros
        B_for_kf = np.zeros((n_state_kf, 1))

        print(f"  [Main Logic] 调用 KalmanFilter - Z shape: {z_for_rerun.shape}, U shape: {U_for_kf.shape}, A shape: {final_dfm_results.A.shape}, B shape: {B_for_kf.shape}, H shape: {Lambda_np.shape if 'Lambda_np' in locals() and Lambda_np is not None else 'Lambda_np not defined'}, Q shape: {final_dfm_results.Q.shape}, R shape: {final_dfm_results.R.shape}")

        kf_results_rerun = KalmanFilter(
            Z=z_for_rerun,                    # 观测数据
            U=U_for_kf,                       # 控制输入向量 (此处为占位符)
            A=final_dfm_results.A,            # 状态转移矩阵
            B=B_for_kf,                       # 控制输入矩阵 (此处为占位符)
            H=Lambda_np,                      # 观测矩阵
            state_names=state_names,          # 状态名称列表
            x0=final_dfm_results.x0,          # 初始状态估计
            P0=final_dfm_results.P0,          # 初始误差协方差
            Q=final_dfm_results.Q,            # 过程噪声协方差
            R=final_dfm_results.R             # 测量噪声协方差
        )
        print("  [Main Logic] 卡尔曼滤波重新运行完成。")
        if kf_results_rerun is None:
            print("  [Main Logic] 严重错误: kf_rerun.filter(z_for_rerun) 返回了 None。无法继续。")
            sys.exit(1)
        if kf_results_rerun.x_minus is None or kf_results_rerun.Kalman_gain is None:
            print("  [Main Logic] 严重错误: 重新运行的卡尔曼滤波结果中缺少 x_minus 或 Kalman_gain。无法继续。")
            sys.exit(1)
        else:
            print(f"  [Main Logic] KF Rerun x_minus (shape {kf_results_rerun.x_minus.shape}) and Kalman_gain (len {len(kf_results_rerun.Kalman_gain)}) 看起来是有效的。")

    except Exception as e_kf_rerun:
        print(f"重新运行卡尔曼滤波时发生错误: {e_kf_rerun}")
        import traceback
        traceback.print_exc()
        # 即使 KF 重新运行失败，我们也允许脚本继续，但后续的新闻分析可能无法进行
        # kf_results_rerun 将保持为 None (或其初始值)
        print("  [Main Logic] 警告: 卡尔曼滤波重新运行失败，新闻贡献分析将使用模拟数据。")
        kf_results_rerun = None  # 确保设置为 None
        # sys.exit(1) # 改为不退出


    print("--- 步骤 3: 确定目标月份、目标日期 T 和月内分析周期 ---")
    all_available_dates = final_dfm_results.x.index # 假设 final_dfm_results.x 包含所有日期的因子状态
    target_period_m = None
    effective_target_month_str = SPECIFIED_TARGET_MONTH_STR
    target_month_source = "N/A"

    if effective_target_month_str:
        try:
            target_period_m = pd.Period(effective_target_month_str, freq='M')
            target_month_source = "Command Line"
            print(f"  [Main Logic] 使用命令行指定的目标月份: {effective_target_month_str}, 解析为 Period: {target_period_m}")
        except ValueError as e_period:
            print(f"  [Main Logic] 警告：无法将命令行目标月份 '{effective_target_month_str}' 解析为 Period: {e_period}。请检查输入后重试。")
            raise ValueError("  [Main Logic] 命令行提供的目标月份无效")
    
    if target_period_m is None and hasattr(config, 'NEWS_TARGET_MONTH') and config.NEWS_TARGET_MONTH:
        effective_target_month_str = config.NEWS_TARGET_MONTH
        try:
            target_period_m = pd.Period(effective_target_month_str, freq='M')
            target_month_source = "Config File"
            print(f"  [Main Logic] 使用配置文件中的目标月份: {effective_target_month_str}, 解析为 Period: {target_period_m}")
        except ValueError as e_period_config:
            raise ValueError(f"  [Main Logic] 配置文件中的目标月份无法解析: {e_period_config}") from e_period_config

    if target_period_m is None: # 自动确定逻辑
        target_month_source = "Auto (Derived)"
        last_date_in_data = all_available_dates.max()
        # 简单的自动逻辑：取数据中最新日期所在月份的前一个月作为目标，如果最新日期是当前月份的话
        # 或者直接取最新数据所在的月份 (需要模型支持预测当月)
        target_period_m = last_date_in_data.to_period('M')
        print(f"  [Main Logic] 自动确定目标月份 Period: {target_period_m} (基于数据最新日期: {last_date_in_data.strftime('%Y-%m-%d')})")

    # ... (这部分逻辑需要非常小心，确保它基于 target_period_m 工作)
    print(f"  [Main Logic] **最终用于计算的目标月份 Period: {target_period_m} (来源: {target_month_source})**")

    target_month_start = target_period_m.start_time
    target_month_end = target_period_m.end_time
    print(f"  [Main Logic] 计算用目标月份范围: {target_month_start.strftime('%Y-%m-%d')} to {target_month_end.strftime('%Y-%m-%d')}")

    # 实际用于计算和数据提取的日期范围，应严格限制在目标月份内或由 analysis_period_dates 定义
    # 我们已经有了 analysis_period_dates，它是目标月份内的有效周五
    # 确保所有后续的数据访问都基于这个范围

    fridays_in_target_month = pd.date_range(start=target_month_start, end=target_month_end, freq='W-FRI')
    # Filter fridays that are actually in the model's data index
    analysis_period_dates_all_model_data = all_available_dates[all_available_dates.dayofweek == 4] # 所有模型数据中的周五
    analysis_period_dates = fridays_in_target_month[fridays_in_target_month.isin(analysis_period_dates_all_model_data)]

    if analysis_period_dates.empty:
        print(f"  [Main Logic] 错误: 在目标月份 {target_period_m.strftime('%Y-%m')} ({target_month_source}) 内，数据中找不到任何周五作为分析时点。")
        # 创建空的CSV以避免绘图函数因文件不存在而失败
        pd.DataFrame().to_csv(os.path.join(EVOLUTION_OUTPUT_DIR, "nowcast_evolution_data_T.csv"))
        pd.DataFrame().to_csv(os.path.join(EVOLUTION_OUTPUT_DIR, "news_decomposition_grouped.csv"))
        print(f"  [Main Logic] 已创建空的CSV文件，图表将为空或提示无数据。")
        # 这里可以选择 sys.exit(1) 或者让它继续生成空图
    else:
        print(f"  [Main Logic] 月内分析 Vintage 日期 (基于目标月份 {target_period_m.strftime('%Y-%m')}): {len(analysis_period_dates)} 个点, 从 {analysis_period_dates.min().strftime('%Y-%m-%d')} 到 {analysis_period_dates.max().strftime('%Y-%m-%d')}")

    # target_prediction_date 通常是 target_period_m 的最后一个周五 (如果数据存在)
    if not analysis_period_dates.empty:
        target_prediction_date = analysis_period_dates.max()
        print(f"  [Main Logic] 预测目标日期 T (基于目标月份的最后一个可用Vintage): {target_prediction_date.strftime('%Y-%m-%d')}")
    else:
        raise ValueError('  [Main Logic] 无可用Vintage，无法确定预测目标日期 T')

    # 在此之前，确保核心数据对象 (如 final_filtered_x, x_minus_df, z_observed) 被适当地筛选
    # 以匹配 analysis_period_dates 或至少 target_month_start/end

    # 假设 final_dfm_results.x (即 final_filtered_x), kf_results_rerun.x_minus, 和 z_for_rerun (即 z_observed)
    # 已经是加载的完整数据集，我们需要从这里筛选出目标月份的数据
    # 注意：筛选应基于 analysis_period_dates 的最小和最大日期，或 target_month_start/end
    # 为确保与 analysis_period_dates 一致，使用其范围
    if not analysis_period_dates.empty:
        calc_start_date = analysis_period_dates.min()
        calc_end_date = analysis_period_dates.max()
    else: # 如果没有分析日期，使用月份范围，后续计算会产生空结果
        calc_start_date = target_month_start
        calc_end_date = target_month_end

    print(f"  [Main Logic] 用于核心计算的数据筛选范围: {calc_start_date.strftime('%Y-%m-%d')} to {calc_end_date.strftime('%Y-%m-%d')}")

    # 筛选核心数据帧
    # final_filtered_x 是 final_dfm_results.x
    current_month_final_filtered_x = final_dfm_results.x[(final_dfm_results.x.index >= calc_start_date) & (final_dfm_results.x.index <= calc_end_date)]
    
    if kf_results_rerun and hasattr(kf_results_rerun, 'x_minus') and kf_results_rerun.x_minus is not None:
        current_month_x_minus_df = kf_results_rerun.x_minus[(kf_results_rerun.x_minus.index >= calc_start_date) & (kf_results_rerun.x_minus.index <= calc_end_date)]
    else:
        print("  [Main Logic] 警告: kf_results_rerun 无效或缺少 x_minus，无法筛选 current_month_x_minus_df。将使用空DataFrame。")
        current_month_x_minus_df = pd.DataFrame(index=pd.to_datetime([])) # 创建一个空的带DatetimeIndex的DataFrame

    # z_observed 是 z_for_rerun
    current_month_z_observed = z_for_rerun[(z_for_rerun.index >= calc_start_date) & (z_for_rerun.index <= calc_end_date)]
    
    # Kalman_gain_list 需要特殊处理，因为它是一个列表，其长度应与 z_observed 的行数对应
    # 如果 z_observed 被截断，Kalman_gain_list 也需要相应截断
    # 假设原始 kalman_gain_list 对应原始 z_for_rerun
    original_z_dates = z_for_rerun.index
    kalman_gain_list_for_month = []
    if not current_month_z_observed.empty:
        if kf_results_rerun and hasattr(kf_results_rerun, 'Kalman_gain') and kf_results_rerun.Kalman_gain is not None:
            for date_in_month in current_month_z_observed.index:
                try:
                    original_idx = original_z_dates.get_loc(date_in_month)
                    if original_idx < len(kf_results_rerun.Kalman_gain):
                         kalman_gain_list_for_month.append(kf_results_rerun.Kalman_gain[original_idx])
                    else:
                        print(f"  [Main Logic] 警告: 无法为日期 {date_in_month} 找到对应的Kalman增益 (索引超出范围)")
                except KeyError:
                    print(f"  [Main Logic] 警告: 日期 {date_in_month} 不在原始 z_for_rerun.index 中，无法获取Kalman增益。")
        else:
            print("  [Main Logic] 警告: kf_results_rerun 无效或缺少 Kalman_gain，无法填充 kalman_gain_list_for_month。")
    else:
        print("  [Main Logic] current_month_z_observed 为空，kalman_gain_list_for_month 将为空。")


    print(f"--- 步骤 4: 计算目标日期 {target_prediction_date.strftime('%Y-%m-%d')} (针对月份 {target_period_m.strftime('%Y-%m')}) 的 Nowcast 演变序列 ---")
    nowcast_evolution_list = []
    A_pow_k_cache = {} # 在循环外定义
    lambda_target_row = metadata.get('lambda_target_row') # 假设这个从元数据中预存或计算
    if lambda_target_row is None:
        # 从 Lambda_np (如果已加载) 中获取
        if 'Lambda_np' in locals() and 'TARGET_VARIABLE_FROM_METADATA' in locals() and 'obs_names_from_lambda' in locals():
            try:
                target_var_idx = obs_names_from_lambda.index(TARGET_VARIABLE_FROM_METADATA)
                lambda_target_row = Lambda_np[target_var_idx, :]
            except (ValueError, IndexError) as e_lambda_idx:
                print(f"  [Main Logic] 错误: 无法从Lambda中找到目标变量 '{TARGET_VARIABLE_FROM_METADATA}' 的行: {e_lambda_idx}")
                lambda_target_row = np.array([]) # 设置为空，后续会安全失败
        else:
            print("  [Main Logic] 错误: Lambda_np 或相关变量未定义，无法获取 lambda_target_row.")
            lambda_target_row = np.array([])

    target_mean_original = metadata.get('target_mean_original')
    target_std_original = metadata.get('target_std_original')
    n_factors = final_dfm_results.A.shape[0] # 从加载的A矩阵获取
    A_matrix = final_dfm_results.A # 从加载的A矩阵获取

    if lambda_target_row.size > 0 and target_mean_original is not None and target_std_original is not None:
        for t_vintage in analysis_period_dates: # 这里的 analysis_period_dates 已经是目标月份的日期
            days_diff = (target_prediction_date - t_vintage).days
            weeks_diff = int(np.round(days_diff / 7))
            weeks_diff = max(0, weeks_diff)

            if t_vintage in current_month_final_filtered_x.index:
                x_t = current_month_final_filtered_x.loc[t_vintage].values

                if weeks_diff == 0:
                    A_pow_k = np.eye(n_factors)
                elif weeks_diff in A_pow_k_cache:
                    A_pow_k = A_pow_k_cache[weeks_diff]
                else:
                    A_pow_k = np.linalg.matrix_power(A_matrix, weeks_diff)
                    A_pow_k_cache[weeks_diff] = A_pow_k

                x_T_given_t = A_pow_k @ x_t
                nowcast_std_val = lambda_target_row @ x_T_given_t
                nowcast_orig_val = (nowcast_std_val * target_std_original + target_mean_original) if pd.notna(target_std_original) and pd.notna(target_mean_original) and target_std_original != 0 else nowcast_std_val

                nowcast_evolution_list.append({
                    'date': t_vintage,
                    'target_prediction_date': target_prediction_date,
                    'nowcast_orig': nowcast_orig_val,
                    'nowcast_std': nowcast_std_val,
                    'forecast_horizon_weeks': weeks_diff
                })
            else:
                # 即使数据不可用，也添加一个占位符记录以保持时间点一致性
                print(f"  [Main Logic] 警告 (演变计算): Vintage日期 {t_vintage.strftime('%Y-%m-%d')} 不在筛选后的因子数据中，添加NaN占位符。")
                nowcast_evolution_list.append({
                    'date': t_vintage,
                    'target_prediction_date': target_prediction_date,
                    'nowcast_orig': np.nan,
                    'nowcast_std': np.nan,
                    'forecast_horizon_weeks': weeks_diff
                })
    else:
        print("  [Main Logic] 演变计算所需参数 (lambda_target_row, mean, std) 不完整，创建占位符数据以保持时间点一致性。")
        # 即使参数不完整，也为所有时间点创建占位符记录
        for t_vintage in analysis_period_dates:
            days_diff = (target_prediction_date - t_vintage).days
            weeks_diff = int(np.round(days_diff / 7))
            weeks_diff = max(0, weeks_diff)
            nowcast_evolution_list.append({
                'date': t_vintage,
                'target_prediction_date': target_prediction_date,
                'nowcast_orig': np.nan,
                'nowcast_std': np.nan,
                'forecast_horizon_weeks': weeks_diff
            })

    nowcast_forecast_df = pd.DataFrame(nowcast_evolution_list).set_index('date') if nowcast_evolution_list else pd.DataFrame(columns=['target_prediction_date', 'nowcast_orig', 'nowcast_std', 'forecast_horizon_weeks'])

    nowcast_output_path = os.path.join(EVOLUTION_OUTPUT_DIR, "nowcast_evolution_data_T.csv")
    if 'nowcast_forecast_df' in locals() and not nowcast_forecast_df.empty:
        nowcast_forecast_df.to_csv(nowcast_output_path, index_label='date')
        print(f"  [Main Logic] Nowcast 演变数据已保存到: {nowcast_output_path} (针对目标月份: {target_period_m.strftime('%Y-%m')})")
    else:
        pd.DataFrame().to_csv(nowcast_output_path) # 保存空文件以防绘图出错
        print(f"  [Main Logic] Nowcast 演变数据为空，已保存空CSV到: {nowcast_output_path}")

    print(f"--- 步骤 5: 计算目标日期 {target_prediction_date.strftime('%Y-%m-%d')} (针对月份 {target_period_m.strftime('%Y-%m')}) 的新闻贡献 ---")
    news_decomposition_list = []
    var_industry_map = metadata.get('var_industry_map', {})
    industry_groups = list(set(var_industry_map.values())) + ['其他未分类']
    var_to_group = {var: get_industry_group(var, var_industry_map) for var in obs_names_from_lambda} if 'obs_names_from_lambda' in locals() else {}
    n_obs = Lambda_np.shape[0] if 'Lambda_np' in locals() else 0

    if lambda_target_row.size > 0 and target_std_original is not None and 'Lambda_np' in locals():
        print(f"  [Main Logic] 开始新闻贡献计算循环，共 {len(analysis_period_dates)} 个Vintage日期...")
        for t_idx, t_vintage in enumerate(analysis_period_dates):
            print(f"    [News Calc Loop] 处理 Vintage: {t_vintage.strftime('%Y-%m-%d')} (索引 {t_idx}) ")

            # 先创建默认的零贡献记录
            news_data_row = {'date': t_vintage}
            for group in industry_groups:
                news_data_row[group] = 0.0

            # 尝试计算实际的新闻贡献
            if (t_vintage in current_month_x_minus_df.index and \
                t_vintage in current_month_z_observed.index and \
                t_idx < len(kalman_gain_list_for_month)):

                x_t_minus_1 = current_month_x_minus_df.loc[t_vintage].values
                K_t = kalman_gain_list_for_month[t_idx]
                z_t_actual = current_month_z_observed.loc[t_vintage].values

                if K_t is not None and K_t.shape[0] == n_factors and K_t.shape[1] == n_obs:
                    z_t_pred = Lambda_np @ x_t_minus_1
                    nu_t = np.full(n_obs, np.nan)
                    valid_obs_mask = ~np.isnan(z_t_actual)
                    # 修复符号错误：新闻应该是预测值减去实际值，使符号与nowcast变化一致
                    nu_t[valid_obs_mask] = z_t_pred[valid_obs_mask] - z_t_actual[valid_obs_mask]

                    current_vintage_evo_data = nowcast_forecast_df[nowcast_forecast_df.index == t_vintage]
                    if not current_vintage_evo_data.empty:
                        weeks_diff = int(current_vintage_evo_data['forecast_horizon_weeks'].iloc[0])
                    else:
                        days_diff_news = (target_prediction_date - t_vintage).days
                        weeks_diff = int(np.round(days_diff_news / 7))
                        weeks_diff = max(0, weeks_diff)

                    if weeks_diff == 0: A_pow_k = np.eye(n_factors)
                    elif weeks_diff in A_pow_k_cache: A_pow_k = A_pow_k_cache[weeks_diff]
                    else: A_pow_k = np.linalg.matrix_power(A_matrix, weeks_diff); A_pow_k_cache[weeks_diff] = A_pow_k

                    term1 = lambda_target_row @ A_pow_k
                    weights_w = term1 @ K_t
                    news_contributions_i = weights_w * nu_t
                    news_contributions_i_orig = news_contributions_i * target_std_original if pd.notna(target_std_original) and target_std_original != 0 else news_contributions_i

                    grouped_news = defaultdict(float)
                    if 'obs_names_from_lambda' in locals():
                        for i, var_name in enumerate(obs_names_from_lambda):
                            contribution = news_contributions_i_orig[i]
                            if not pd.isna(contribution) and abs(contribution) > 1e-6: # 仅记录非nan且有显著影响的贡献
                                grouped_news[var_to_group.get(var_name, '其他未分类')] += contribution

                    # 更新记录中的实际贡献值
                    for group in industry_groups:
                        news_data_row[group] = grouped_news.get(group, 0.0)

                    # 调试信息：显示是否有显著贡献
                    if grouped_news:
                        print(f"      [News Calc Loop] 为 {t_vintage.strftime('%Y-%m-%d')} 添加了新闻分解数据: {dict(grouped_news)}")
                    else:
                        print(f"      [News Calc Loop] 为 {t_vintage.strftime('%Y-%m-%d')} 添加了零贡献数据（保持时间点一致性）")
                else:
                    print(f"      [News Calc Loop] 警告: Vintage {t_vintage.strftime('%Y-%m-%d')} 的Kalman增益形状不正确或为None，使用零贡献。")
            else:
                print(f"    [News Calc Loop] 警告: Vintage日期 {t_vintage.strftime('%Y-%m-%d')} 数据不完整，使用零贡献保持时间点一致性。")

            # 无论如何都添加记录，确保时间点一致性
            news_decomposition_list.append(news_data_row)
    else:
        print("  [Main Logic] 新闻贡献计算所需参数 (lambda_target_row, std, Lambda_np) 不完整，创建占位符数据以保持时间点一致性。")
        # 即使参数不完整，也为所有时间点创建占位符记录
        for t_vintage in analysis_period_dates:
            news_data_row = {'date': t_vintage}
            for group in industry_groups:
                news_data_row[group] = 0.0
            news_decomposition_list.append(news_data_row)

    news_decomposition_df = pd.DataFrame(news_decomposition_list).set_index('date') if news_decomposition_list else pd.DataFrame(columns=industry_groups)
    if not news_decomposition_df.empty:
        for group in industry_groups: # 确保所有预期的行业组列都存在，即使它们在该月份没有贡献
            if group not in news_decomposition_df.columns:
                news_decomposition_df[group] = 0.0
        news_decomposition_df = news_decomposition_df[industry_groups] # 确保列顺序一致性
        print("  [Main Logic] 生成的 news_decomposition_df (前5行):\n", news_decomposition_df.head())
    else:
        print("  [Main Logic] news_decomposition_df 为空 (没有计算出任何新闻贡献)。")

    news_output_path = os.path.join(EVOLUTION_OUTPUT_DIR, "news_decomposition_grouped.csv")
    if 'news_decomposition_df' in locals() and not news_decomposition_df.empty:
        news_decomposition_df.to_csv(news_output_path, index_label='date', encoding='utf-8')
        print(f"  [Main Logic] 新闻分解数据已保存到: {news_output_path} (针对目标月份: {target_period_m.strftime('%Y-%m')})")
    else:
        pd.DataFrame().to_csv(news_output_path, encoding='utf-8') # 保存空文件
        print(f"  [Main Logic] 新闻分解数据为空，已保存空CSV到: {news_output_path}")


    plot_output_file_arg = args.plot_output_file 
    
    plot_base_name_for_func = "news_analysis_plot" # 默认基础名
    plot_output_dir_for_func = EVOLUTION_OUTPUT_DIR # 默认输出目录

    if plot_output_file_arg:
        plot_output_dir_for_func = os.path.dirname(plot_output_file_arg)
        base_name_with_ext = os.path.basename(plot_output_file_arg)
        plot_base_name_for_func = os.path.splitext(base_name_with_ext)[0]
        # 如果 backend 传来的名字是 news_analysis_plot_backend.html, 那么 plot_base_name_for_func 会是 news_analysis_plot_backend
        print(f"  [Main] 使用来自参数的输出目录: {plot_output_dir_for_func} 和基础文件名: {plot_base_name_for_func}")
    else:
        # 如果后端没有提供明确的文件名 (理论上不应该发生，因为 backend.py 会提供)
        print(f"  [Main] 未指定 --plot_output_file, 将使用默认输出目录: {plot_output_dir_for_func} 和基础文件名: {plot_base_name_for_func}")
    
    # 确保目录存在
    os.makedirs(plot_output_dir_for_func, exist_ok=True)

    print(f"--- 步骤 5: 调用绘图函数生成新闻分解图和演变图 (基础名: {plot_base_name_for_func}) ---")
    target_var_display_name = metadata.get('target_variable_label', TARGET_VARIABLE) \
        if 'metadata' in locals() and metadata else TARGET_VARIABLE

    try:
        # 调用修改后的绘图函数
        generated_plot_paths = plot_news_decomposition(
            input_dir=EVOLUTION_OUTPUT_DIR, 
            # output_file=plot_output_file_arg, # <<< 不再直接使用这个参数
            base_output_filename=plot_base_name_for_func,
            output_dir=plot_output_dir_for_func,
            plot_start_date=args.plot_start_date,
            plot_end_date=args.plot_end_date,
            target_variable_name=target_var_display_name
        )
        # print(f"图表已生成: {plot_output_file_arg}") # 旧的打印信息
        if generated_plot_paths["evolution_plot_path"]:
            print(f"  演变图已生成: {generated_plot_paths['evolution_plot_path']}")
        else:
            print("  演变图生成失败或未保存。")
        if generated_plot_paths["decomposition_plot_path"]:
            print(f"  分解图已生成: {generated_plot_paths['decomposition_plot_path']}")
        else:
            print("  分解图生成失败或未保存。")
            
    except FileNotFoundError as e_plot_fnf:
        print(f"绘图失败: 所需数据文件未找到: {e_plot_fnf}")
        # sys.exit(1) # 根据情况决定是否因为绘图失败而退出
    except ValueError as e_plot_val:
        print(f"绘图失败: 数据值错误或日期范围无效: {e_plot_val}")
    except RuntimeError as e_plot_rt:
        print(f"绘图失败: 运行时错误: {e_plot_rt}")
    except Exception as e_plot:
        print(f"生成图表时发生未知错误: {e_plot}")
        import traceback
        traceback.print_exc()


    print("--- Nowcast 演变与新闻贡献分析完成 ---")
