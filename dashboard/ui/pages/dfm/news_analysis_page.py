# -*- coding: utf-8 -*-
"""
DFM新闻分析页面组件

完全重构版本，与dfm_old_ui/news_analysis_front_end.py保持完全一致
"""

import streamlit as st
import pandas as pd
import os
import sys
import traceback
from datetime import datetime, timedelta
from typing import Optional, Dict, Any

# 添加路径以导入统一状态管理
current_dir = os.path.dirname(os.path.abspath(__file__))
dashboard_root = os.path.abspath(os.path.join(current_dir, '..', '..', '..', '..'))
if dashboard_root not in sys.path:
    sys.path.insert(0, dashboard_root)

# 导入统一状态管理
from dashboard.core import get_global_dfm_manager

print("[DFM News Analysis] [SUCCESS] 统一状态管理器导入成功")


def get_dfm_manager():
    """获取DFM模块管理器实例（使用全局单例）"""
    try:
        dfm_manager = get_global_dfm_manager()
        if dfm_manager is None:
            raise RuntimeError("全局DFM管理器不可用")
        return dfm_manager
    except Exception as e:
        print(f"[DFM News Analysis] Error getting DFM manager: {e}")
        raise RuntimeError(f"DFM管理器获取失败: {e}")


def get_dfm_state(key, default=None):
    """获取DFM状态值（简化版本）"""
    dfm_manager = get_dfm_manager()

    # 数据相关的键从data_prep命名空间获取
    data_keys = [
        'dfm_prepared_data_df',
        'dfm_transform_log_obj',
        'dfm_industry_map_obj',
        'dfm_removed_vars_log_obj',
        'dfm_var_type_map_obj'
    ]

    if key in data_keys:
        return dfm_manager.get_dfm_state('data_prep', key, default)

    # 模型文件相关从model_analysis获取
    model_keys = ['dfm_model_file_indep', 'dfm_metadata_file_indep']
    if key in model_keys:
        return dfm_manager.get_dfm_state('model_analysis', key, default)

    # 其他键从news_analysis命名空间获取
    return dfm_manager.get_dfm_state('news_analysis', key, default)


def set_dfm_state(key, value):
    """设置DFM状态值（只使用统一状态管理）"""
    dfm_manager = get_dfm_manager()

    success = dfm_manager.set_dfm_state('news_analysis', key, value)
    if not success:
        print(f"[DFM News Analysis] [WARNING] 统一状态管理器设置失败: {key}")
    return success


# 导入后端执行函数
from dashboard.models.DFM.decomp.news_analysis_backend import execute_news_analysis
backend_available = True
print("[News Frontend] Successfully imported news_analysis_backend.execute_news_analysis")


def render_news_analysis_tab(st_instance):
    """
    渲染新闻分析标签页

    Args:
        st_instance: Streamlit实例
    """

    if not backend_available:
        st_instance.error("新闻分析后端不可用，请检查模块导入。")
        return

    # === 参数设置区域 ===
    st_instance.markdown("##### 分析参数设置")

    # 目标月份选择
    target_month_date_selected = st_instance.date_input(
        "目标月份",
        value=datetime.now().replace(day=1),  # 默认当月第一天
        min_value=datetime(2000, 1, 1),      # 合理的最小可选日期
        max_value=datetime.now().replace(day=1) + timedelta(days=365*5),  # 限制可选的最大日期
        key="news_target_month_date_selector_frontend",
        help="选择您希望进行新闻归因分析的目标月份。"
    )
    st_instance.caption("选择目标月份后，程序将自动使用该年和月份进行分析。")

    # === 文件检查区域 ===
    st_instance.markdown("##### 模型文件检查")

    model_file = get_dfm_state('dfm_model_file_indep', None)
    metadata_file = get_dfm_state('dfm_metadata_file_indep', None)

    # 显示文件状态
    col_file1, col_file2 = st_instance.columns(2)
    with col_file1:
        if model_file is not None:
            file_name = getattr(model_file, 'name', '未知文件')
            st_instance.success(f"模型文件: {file_name}")
        else:
            st_instance.error("未找到模型文件")

    with col_file2:
        if metadata_file is not None:
            file_name = getattr(metadata_file, 'name', '未知文件')
            st_instance.success(f"元数据文件: {file_name}")
        else:
            st_instance.error("未找到元数据文件")

    if model_file is None or metadata_file is None:
        st_instance.warning("请先在 **模型分析** 标签页上传必要的模型文件和元数据文件。")
        st_instance.info("提示：模型文件通常为 .joblib 格式，元数据文件通常为 .pkl 格式。")
        return

    # === 执行按钮 ===
    st_instance.markdown("---")

    # === 执行按钮 ===
    analysis_triggered = st_instance.button(
        "运行新闻分析和生成图表",
        key="run_news_analysis_frontend_button",
        type="primary",
        use_container_width=True
    )

    if analysis_triggered:
        _execute_news_analysis(st_instance, target_month_date_selected, model_file, metadata_file)

    # === 统一的结果显示区域 ===
    # 检查是否有分析结果需要显示（无论是刚完成的还是之前缓存的）
    if get_dfm_state('news_analysis_completed', False):
        st_instance.markdown("---")

        # 如果是刚刚触发的分析，显示完成信息
        if analysis_triggered:
            st_instance.success("新闻分析执行完成！")
        else:
            st_instance.info("显示之前的分析结果")

        # 构建结果数据
        cached_backend_results = {
            "returncode": 0,
            "evo_csv_path": get_dfm_state('news_analysis_evo_csv_path'),
            "news_csv_path": get_dfm_state('news_analysis_news_csv_path'),
            "evolution_plot_path": get_dfm_state('news_analysis_evolution_plot_path'),
            "decomposition_plot_path": get_dfm_state('news_analysis_decomposition_plot_path')
        }

        # 显示图表（下载按钮已集成在图表中）
        _display_charts(st_instance, cached_backend_results)


def _execute_news_analysis(st_instance, target_month_date_selected, model_file, metadata_file):
    """
    执行新闻分析的内部函数

    Args:
        st_instance: Streamlit实例
        target_month_date_selected: 选择的目标月份
        model_file: 模型文件
        metadata_file: 元数据文件
    """

    with st_instance.spinner("正在执行新闻分析，请稍候..."):
        try:
            # 准备参数
            target_month_str = target_month_date_selected.strftime('%Y-%m') if target_month_date_selected else None
            plot_start_date_str = None
            plot_end_date_str = None

            # 调用后端执行
            backend_results = execute_news_analysis(
                dfm_model_file_content=model_file.getbuffer(),
                dfm_metadata_file_content=metadata_file.getbuffer(),
                target_month=target_month_str,
                plot_start_date=plot_start_date_str,
                plot_end_date=plot_end_date_str,
            )

            # 处理结果（只保存状态，不显示结果）
            _save_analysis_results(st_instance, backend_results, target_month_str)

        except Exception as e_call_backend:
            st_instance.error(f"调用后端脚本时发生错误: {e_call_backend}")
            st_instance.error(f"详细错误信息: {traceback.format_exc()}")


def _save_analysis_results(st_instance, backend_results: Dict[str, Any], target_month_str: Optional[str]):
    """
    保存分析结果到状态管理器（不显示结果）

    Args:
        st_instance: Streamlit实例
        backend_results: 后端返回的结果字典
        target_month_str: 目标月份字符串
    """

    if backend_results["returncode"] == 0:
        # 保存文件路径到统一状态管理器，避免页面刷新时丢失
        if backend_results.get("evo_csv_path"):
            set_dfm_state('news_analysis_evo_csv_path', backend_results["evo_csv_path"])
        if backend_results.get("news_csv_path"):
            set_dfm_state('news_analysis_news_csv_path', backend_results["news_csv_path"])
        if backend_results.get("evolution_plot_path"):
            set_dfm_state('news_analysis_evolution_plot_path', backend_results["evolution_plot_path"])
        if backend_results.get("decomposition_plot_path"):
            set_dfm_state('news_analysis_decomposition_plot_path', backend_results["decomposition_plot_path"])

        # 设置标记表示分析已完成
        set_dfm_state('news_analysis_completed', True)
        set_dfm_state('news_analysis_target_month', target_month_str)

    else:
        st_instance.error(f"后端脚本执行失败 (返回码: {backend_results['returncode']})")

        if backend_results.get("error_message"):
            st_instance.error(f"错误详情: {backend_results['error_message']}")

        # 显示调试信息
        if backend_results.get("stderr"):
            with st_instance.expander("查看错误日志", expanded=False):
                st_instance.code(backend_results["stderr"], language="text")


def _display_analysis_results(st_instance, backend_results: Dict[str, Any], target_month_str: Optional[str]):
    """
    显示分析结果
    
    Args:
        st_instance: Streamlit实例
        backend_results: 后端返回的结果字典
        target_month_str: 目标月份字符串
    """
    
    if backend_results["returncode"] == 0:
        st_instance.success("后端脚本执行成功！")

        # === 结果展示区域 ===
        st_instance.markdown("---")

        # 图表展示（下载按钮已集成在图表中）
        _display_charts(st_instance, backend_results)
        
    else:
        st_instance.error(f"后端脚本执行失败 (返回码: {backend_results['returncode']})")
        
        if backend_results.get("error_message"):
            st_instance.error(f"错误详情: {backend_results['error_message']}")
        
        # 显示调试信息
        if backend_results.get("stderr"):
            with st_instance.expander("查看错误日志", expanded=False):
                st_instance.code(backend_results["stderr"], language="text")


def _display_charts(st_instance, backend_results: Dict[str, Any]):
    """
    显示图表
    
    Args:
        st_instance: Streamlit实例
        backend_results: 后端返回的结果字典
    """
    
    col_left_chart, col_right_chart = st_instance.columns(2)
    
    # 先预加载和缓存所有文件数据，确保数据持久化
    evo_csv_path = get_dfm_state('news_analysis_evo_csv_path') or backend_results.get("evo_csv_path")
    news_csv_path = get_dfm_state('news_analysis_news_csv_path') or backend_results.get("news_csv_path")
    _prepare_download_data(evo_csv_path, news_csv_path)

    # 左列：演变图
    with col_left_chart:
        st_instance.markdown("##### Nowcast 演变图")
        # 优先从状态管理器获取，避免页面刷新时丢失
        evo_plot_path = get_dfm_state('news_analysis_evolution_plot_path') or backend_results.get("evolution_plot_path")

        if evo_plot_path and os.path.exists(evo_plot_path):
            if evo_plot_path.lower().endswith(".html"):
                try:
                    with open(evo_plot_path, 'r', encoding='utf-8') as f_html_evo:
                        html_content_evo = f_html_evo.read()
                    st_instance.components.v1.html(html_content_evo, height=500, scrolling=True)
                    print(f"[News Frontend] Displayed Evolution HTML plot: {evo_plot_path}")
                except Exception as e_html_evo:
                    st_instance.error(f"读取或显示演变图时出错: {e_html_evo}")
            else:
                st_instance.warning(f"演变图文件格式未知: {evo_plot_path}")
        else:
            st_instance.warning("未找到 Nowcast 演变图文件或路径无效。")

        # 在图表下方直接添加下载按钮
        _render_single_download(st_instance, evo_csv_path, "evo", "Nowcast 演变数据")
    
    # 右列：分解图
    with col_right_chart:
        st_instance.markdown("##### 新闻贡献分解图")
        # 优先从状态管理器获取，避免页面刷新时丢失
        decomp_plot_path = get_dfm_state('news_analysis_decomposition_plot_path') or backend_results.get("decomposition_plot_path")

        if decomp_plot_path and os.path.exists(decomp_plot_path):
            if decomp_plot_path.lower().endswith(".html"):
                try:
                    with open(decomp_plot_path, 'r', encoding='utf-8') as f_html_decomp:
                        html_content_decomp = f_html_decomp.read()
                    st_instance.components.v1.html(html_content_decomp, height=500, scrolling=True)
                    print(f"[News Frontend] Displayed Decomposition HTML plot: {decomp_plot_path}")
                except Exception as e_html_decomp:
                    st_instance.error(f"读取或显示分解图时出错: {e_html_decomp}")
            else:
                st_instance.warning(f"分解图文件格式未知: {decomp_plot_path}")
        else:
            st_instance.warning("未找到新闻贡献分解图文件或路径无效。")

        # 在图表下方直接添加下载按钮
        news_csv_path = get_dfm_state('news_analysis_news_csv_path') or backend_results.get("news_csv_path")
        _render_single_download(st_instance, news_csv_path, "news", "新闻分解数据")


def _display_download_section(st_instance, backend_results: Dict[str, Any]):
    """
    显示下载区域

    Args:
        st_instance: Streamlit实例
        backend_results: 后端返回的结果字典
    """

    # 优先从状态管理器获取文件路径，避免页面刷新时丢失
    evo_csv_path = get_dfm_state('news_analysis_evo_csv_path') or backend_results.get("evo_csv_path")
    news_csv_path = get_dfm_state('news_analysis_news_csv_path') or backend_results.get("news_csv_path")

    st_instance.markdown("##### 数据下载")
    
    # 预加载和缓存所有文件数据，确保数据持久化
    _prepare_download_data(evo_csv_path, news_csv_path)
    
    # 使用expander来包装下载区域，减少页面刷新的影响
    with st_instance.expander("下载选项", expanded=True):
        col_dl1, col_dl2 = st_instance.columns(2)

        with col_dl1:
            _render_single_download(st_instance, evo_csv_path, "evo", "Nowcast 演变数据")

        with col_dl2:
            _render_single_download(st_instance, news_csv_path, "news", "新闻分解数据")


def _prepare_download_data(evo_csv_path: Optional[str], news_csv_path: Optional[str]):
    """
    预加载和缓存下载数据
    
    Args:
        evo_csv_path: 演变数据文件路径
        news_csv_path: 新闻数据文件路径
    """
    
    print(f"[News Analysis Debug] _prepare_download_data called with:")
    print(f"[News Analysis Debug]   evo_csv_path: {evo_csv_path}")
    print(f"[News Analysis Debug]   news_csv_path: {news_csv_path}")
    
    # 处理演变数据文件
    if evo_csv_path and os.path.exists(evo_csv_path):
        try:
            print(f"[News Analysis Debug] Processing evo file: {evo_csv_path}")
            cached_evo_data = get_dfm_state('news_analysis_evo_csv_data')
            cached_evo_path = get_dfm_state('news_analysis_evo_csv_cached_path')
            
            print(f"[News Analysis Debug] Cached evo path: {cached_evo_path}")
            print(f"[News Analysis Debug] Cached evo data exists: {cached_evo_data is not None}")
            
            if cached_evo_path != evo_csv_path or cached_evo_data is None:
                print(f"[News Analysis Debug] Caching new evo data...")
                with open(evo_csv_path, "rb") as fp_evo:
                    file_data = fp_evo.read()
                print(f"[News Analysis Debug] Read evo file size: {len(file_data)} bytes")
                success = set_dfm_state('news_analysis_evo_csv_data', file_data)
                set_dfm_state('news_analysis_evo_csv_cached_path', evo_csv_path)
                print(f"[News Analysis Debug] Evo data cached successfully: {success}")
            else:
                print(f"[News Analysis Debug] Using existing cached evo data")
        except Exception as e:
            print(f"[News Analysis] Error caching evo data: {e}")
            import traceback
            traceback.print_exc()
    else:
        print(f"[News Analysis Debug] Evo file not found or invalid path")
    
    # 处理新闻数据文件
    if news_csv_path and os.path.exists(news_csv_path):
        try:
            print(f"[News Analysis Debug] Processing news file: {news_csv_path}")
            cached_news_data = get_dfm_state('news_analysis_news_csv_data')
            cached_news_path = get_dfm_state('news_analysis_news_csv_cached_path')
            
            print(f"[News Analysis Debug] Cached news path: {cached_news_path}")
            print(f"[News Analysis Debug] Cached news data exists: {cached_news_data is not None}")
            
            if cached_news_path != news_csv_path or cached_news_data is None:
                print(f"[News Analysis Debug] Caching new news data...")
                with open(news_csv_path, "rb") as fp_news:
                    file_data = fp_news.read()
                print(f"[News Analysis Debug] Read news file size: {len(file_data)} bytes")
                success = set_dfm_state('news_analysis_news_csv_data', file_data)
                set_dfm_state('news_analysis_news_csv_cached_path', news_csv_path)
                print(f"[News Analysis Debug] News data cached successfully: {success}")
            else:
                print(f"[News Analysis Debug] Using existing cached news data")
        except Exception as e:
            print(f"[News Analysis] Error caching news data: {e}")
            import traceback
            traceback.print_exc()
    else:
        print(f"[News Analysis Debug] News file not found or invalid path")


def _render_single_download(st_instance, file_path: Optional[str], file_type: str, description: str):
    """
    渲染单个下载区域
    
    Args:
        st_instance: Streamlit实例
        file_path: 文件路径
        file_type: 文件类型标识 ("evo" 或 "news")
        description: 文件描述
    """
    
    st_instance.markdown(f"**{description}**")
    
    if not file_path or not os.path.exists(file_path):
        st_instance.caption("CSV 文件未生成")
        return
    
    try:
        print(f"[News Analysis Debug] _render_single_download called for {file_type}")
        print(f"[News Analysis Debug]   file_path: {file_path}")
        print(f"[News Analysis Debug]   file_exists: {os.path.exists(file_path) if file_path else False}")
        
        # 从缓存获取文件数据
        cached_data_key = f'news_analysis_{file_type}_csv_data'
        print(f"[News Analysis Debug]   cached_data_key: {cached_data_key}")
        
        file_data = get_dfm_state(cached_data_key)
        print(f"[News Analysis Debug]   cached_data_exists: {file_data is not None}")
        if file_data:
            print(f"[News Analysis Debug]   cached_data_size: {len(file_data)} bytes")
        
        if file_data is None:
            st_instance.error(f"文件数据未缓存，请重新运行分析")
            print(f"[News Analysis Debug] ERROR: No cached data found for {file_type}")
            return
            
        # 显示文件信息
        file_size_mb = len(file_data) / (1024 * 1024)
        st_instance.caption(f"{os.path.basename(file_path)} ({file_size_mb:.2f} MB)")
        
        # 使用稳定的键，基于文件路径和类型
        # 避免使用时间戳，确保键的稳定性
        stable_key = f"download_{file_type}_csv_{abs(hash(file_path + file_type)) % 99999}"
        
        # 直接使用download_button，但使用稳定的键
        st_instance.download_button(
            label=f"下载 CSV",
            data=file_data,
            file_name=os.path.basename(file_path),
            mime="text/csv",
            key=stable_key,
            use_container_width=True,
            help=f"下载 {description} CSV 文件"
        )
            
    except Exception as e:
        st_instance.error(f"准备下载文件时出错: {e}")
        print(f"[News Analysis] Download error for {file_type}: {e}")


def render_dfm_news_analysis_page(st_module: Any) -> Dict[str, Any]:
    """
    渲染DFM新闻分析页面

    Args:
        st_module: Streamlit模块

    Returns:
        Dict[str, Any]: 渲染结果
    """
    try:
        # 调用主要的UI渲染函数
        render_news_analysis_tab(st_module)

        return {
            'status': 'success',
            'page': 'news_analysis',
            'components': ['parameter_setting', 'file_check', 'analysis_execution', 'results_display']
        }

    except Exception as e:
        st_module.error(f"新闻分析页面渲染失败: {str(e)}")
        return {
            'status': 'error',
            'page': 'news_analysis',
            'error': str(e)
        }


def render_dfm_news_analysis_tab(st_module: Any) -> Dict[str, Any]:
    """
    兼容性接口：渲染DFM新闻分析标签页

    Args:
        st_module: Streamlit模块

    Returns:
        Dict[str, Any]: 渲染结果
    """
    return render_dfm_news_analysis_page(st_module)
