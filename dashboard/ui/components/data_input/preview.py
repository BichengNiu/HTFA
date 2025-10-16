# -*- coding: utf-8 -*-
"""
数据预览组件
提供数据预览、筛选和基本统计功能
"""

import streamlit as st
import pandas as pd
import numpy as np
import io
from typing import List, Dict, Any, Optional, Tuple
import logging
from datetime import datetime, date

from dashboard.ui.components.data_input.base import DataInputComponent

logger = logging.getLogger(__name__)


class DataPreviewComponent(DataInputComponent):
    """数据预览组件 - 优化版本，支持缓存和分页"""

    def __init__(self):
        super().__init__("preview", "数据预览")

        # 性能优化配置
        self.max_preview_rows = 1000  # 最大预览行数
        self.page_size = 100  # 分页大小
        self.cache_enabled = True  # 启用缓存
        self.large_file_threshold = 50 * 1024 * 1024  # 50MB大文件阈值
    
    def render_basic_info(self, st_obj, df: pd.DataFrame):
        """渲染基本数据信息 - 优化版本"""

        col1, col2, col3, col4 = st_obj.columns(4)

        with col1:
            st_obj.metric("总行数", f"{df.shape[0]:,}")

        with col2:
            st_obj.metric("总列数", df.shape[1])

        with col3:
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            st_obj.metric("数值列", len(numeric_cols))

        with col4:
            memory_usage = df.memory_usage(deep=True).sum() / 1024 / 1024
            st_obj.metric("内存使用", f"{memory_usage:.2f} MB")

        # 大文件警告
        if memory_usage > 50:  # 50MB
            st_obj.warning(f"[WARNING] 大文件警告：数据占用 {memory_usage:.2f} MB 内存，建议使用分页预览")

    def get_cached_preview(self, df: pd.DataFrame, cache_key: str) -> Optional[pd.DataFrame]:
        """获取缓存的预览数据"""
        if not self.cache_enabled:
            return None

        try:
            # 使用组件状态管理缓存
            cached_data = self.get_state(f"preview_cache_{cache_key}")
            if cached_data is not None:
                return cached_data
        except Exception as e:
            logger.warning(f"获取缓存失败: {e}")

        return None

    def set_cached_preview(self, df: pd.DataFrame, cache_key: str, preview_data: pd.DataFrame):
        """设置缓存的预览数据"""
        if not self.cache_enabled:
            return

        try:
            # 只缓存小于10MB的数据
            memory_usage = preview_data.memory_usage(deep=True).sum() / 1024 / 1024
            if memory_usage < 10:
                self.set_state(f"preview_cache_{cache_key}", preview_data)
        except Exception as e:
            logger.warning(f"设置缓存失败: {e}")

    def render_paginated_preview(self, st_obj, df: pd.DataFrame, page_size: int = None) -> pd.DataFrame:
        """渲染分页预览"""
        if page_size is None:
            page_size = self.page_size

        total_rows = len(df)
        total_pages = (total_rows + page_size - 1) // page_size

        if total_pages <= 1:
            # 数据量小，直接显示
            st_obj.dataframe(df, use_container_width=True)
            return df

        # 分页控制
        col1, col2, col3 = st_obj.columns([1, 2, 1])

        with col2:
            current_page = st_obj.selectbox(
                f"页面 (共 {total_pages} 页，每页 {page_size} 行)",
                options=list(range(1, total_pages + 1)),
                key=f"{self.component_name}_page_selector"
            )

        # 计算当前页数据范围
        start_idx = (current_page - 1) * page_size
        end_idx = min(start_idx + page_size, total_rows)

        # 显示当前页数据
        current_page_data = df.iloc[start_idx:end_idx]

        st_obj.info(f"显示第 {start_idx + 1} - {end_idx} 行 (共 {total_rows:,} 行)")
        st_obj.dataframe(current_page_data, use_container_width=True)

        return current_page_data
    
    def render_column_info(self, st_obj, df: pd.DataFrame):
        """渲染列信息"""
        
        column_info = []
        for col in df.columns:
            col_data = df[col]
            info = {
                '列名': col,
                '数据类型': str(col_data.dtype),
                '非空值': col_data.notna().sum(),
                '缺失值': col_data.isna().sum(),
                '缺失率(%)': round((col_data.isna().sum() / len(col_data)) * 100, 2)
            }
            
            # 对于数值列，添加统计信息
            if pd.api.types.is_numeric_dtype(col_data):
                info.update({
                    '最小值': col_data.min(),
                    '最大值': col_data.max(),
                    '平均值': round(col_data.mean(), 2),
                    '标准差': round(col_data.std(), 2)
                })
            
            column_info.append(info)
        
        column_df = pd.DataFrame(column_info)
        st_obj.dataframe(column_df, use_container_width=True)
    
    def render_data_filter(self, st_obj, df: pd.DataFrame) -> pd.DataFrame:
        """渲染数据筛选器"""
        
        filtered_df = df.copy()
        
        # 时间范围筛选（如果有时间列）
        time_col = None
        for col in df.columns:
            try:
                pd.to_datetime(df[col], errors='raise')
                time_col = col
                break
            except:
                continue
        
        if time_col:
            st_obj.markdown("**时间范围筛选：**")
            
            time_series = pd.to_datetime(df[time_col])
            min_date = time_series.min().date()
            max_date = time_series.max().date()
            
            col1, col2 = st_obj.columns(2)
            
            with col1:
                start_date = st_obj.date_input(
                    "开始日期",
                    value=min_date,
                    min_value=min_date,
                    max_value=max_date,
                    key=f"{self.component_name}_start_date"
                )
            
            with col2:
                end_date = st_obj.date_input(
                    "结束日期",
                    value=max_date,
                    min_value=min_date,
                    max_value=max_date,
                    key=f"{self.component_name}_end_date"
                )
            
            # 应用时间筛选
            if start_date and end_date and start_date <= end_date:
                mask = (time_series.dt.date >= start_date) & (time_series.dt.date <= end_date)
                filtered_df = filtered_df[mask]
                
                if len(filtered_df) != len(df):
                    st_obj.info(f"时间筛选后：{len(filtered_df)} 行（原始：{len(df)} 行）")
        
        # 列选择筛选
        st_obj.markdown("**列选择：**")
        
        all_columns = df.columns.tolist()
        selected_columns = st_obj.multiselect(
            "选择要显示的列",
            options=all_columns,
            default=all_columns,
            key=f"{self.component_name}_column_selector"
        )
        
        if selected_columns:
            filtered_df = filtered_df[selected_columns]
        
        # 行数限制
        st_obj.markdown("**显示行数：**")
        max_rows = st_obj.slider(
            "最大显示行数",
            min_value=10,
            max_value=min(1000, len(filtered_df)),
            value=min(100, len(filtered_df)),
            step=10,
            key=f"{self.component_name}_max_rows"
        )
        
        if max_rows < len(filtered_df):
            filtered_df = filtered_df.head(max_rows)
            st_obj.info(f"显示前 {max_rows} 行（共 {len(df)} 行）")
        
        return filtered_df
    
    def render_statistical_summary(self, st_obj, df: pd.DataFrame):
        """渲染统计摘要"""
        
        numeric_df = df.select_dtypes(include=[np.number])
        
        if numeric_df.empty:
            st_obj.info("没有数值列可进行统计分析")
            return
        
        # 基本统计信息
        st_obj.markdown("**数值列统计摘要：**")
        summary_stats = numeric_df.describe()
        st_obj.dataframe(summary_stats, use_container_width=True)
        
        # 相关性矩阵（如果列数不太多）
        if len(numeric_df.columns) <= 10:
            st_obj.markdown("**相关性矩阵：**")
            corr_matrix = numeric_df.corr()
            st_obj.dataframe(corr_matrix, use_container_width=True)
        
        # 缺失值统计
        missing_stats = numeric_df.isnull().sum()
        if missing_stats.sum() > 0:
            st_obj.markdown("**缺失值统计：**")
            missing_df = pd.DataFrame({
                '列名': missing_stats.index,
                '缺失数量': missing_stats.values,
                '缺失比例(%)': (missing_stats.values / len(numeric_df) * 100).round(2)
            })
            missing_df = missing_df[missing_df['缺失数量'] > 0]
            if not missing_df.empty:
                st_obj.dataframe(missing_df, use_container_width=True)
    
    def render_data_download(self, st_obj, df: pd.DataFrame, filename_prefix: str = "data"):
        """渲染数据下载功能"""
        
        st_obj.markdown("**数据下载：**")
        
        col1, col2 = st_obj.columns(2)
        
        with col1:
            # CSV下载，使用utf-8-sig编码避免中文乱码
            csv_string = df.to_csv(index=False, encoding='utf-8-sig')
            csv_data = csv_string.encode('utf-8-sig')
            st_obj.download_button(
                label="下载为CSV",
                data=csv_data,
                file_name=f"{filename_prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                key=f"{self.component_name}_download_csv"
            )
        
        with col2:
            # Excel下载
            excel_buffer = io.BytesIO()
            with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
                df.to_excel(writer, sheet_name='Data', index=False)
            excel_data = excel_buffer.getvalue()

            st_obj.download_button(
                label="下载为Excel",
                data=excel_data,
                file_name=f"{filename_prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                key=f"{self.component_name}_download_excel"
            )
    
    def render_input_section(self, st_obj, **kwargs) -> Optional[pd.DataFrame]:
        """渲染数据预览输入部分"""
        
        data = kwargs.get('data')
        
        if data is None or data.empty:
            st_obj.info("没有可预览的数据")
            return None
        
        # 选择预览模式
        preview_mode = st_obj.selectbox(
            "预览模式",
            options=["基本预览", "详细信息", "统计摘要", "数据筛选"],
            key=f"{self.component_name}_mode_selector"
        )
        
        if preview_mode == "基本预览":
            # 基本信息
            self.render_basic_info(st_obj, data)

            # 数据预览 - 优化版本
            st_obj.markdown("**数据预览：**")

            # 检查是否为大文件
            memory_usage = data.memory_usage(deep=True).sum() / 1024 / 1024

            if memory_usage > 50 or len(data) > self.max_preview_rows:
                # 大文件使用分页预览
                st_obj.info("[LOADING] 检测到大文件，使用分页预览模式")

                # 分页预览选项
                use_pagination = st_obj.checkbox(
                    "启用分页预览 (推荐用于大文件)",
                    value=True,
                    key=f"{self.component_name}_use_pagination"
                )

                if use_pagination:
                    self.render_paginated_preview(st_obj, data)
                else:
                    # 限制预览行数
                    max_rows = min(self.max_preview_rows, len(data))
                    preview_rows = st_obj.slider(
                        f"预览行数 (最大 {max_rows})",
                        min_value=5,
                        max_value=max_rows,
                        value=min(100, max_rows),
                        key=f"{self.component_name}_preview_rows"
                    )

                    # 使用缓存
                    cache_key = f"basic_preview_{hash(str(data.shape))}"
                    cached_preview = self.get_cached_preview(data, cache_key)

                    if cached_preview is not None and len(cached_preview) >= preview_rows:
                        preview_data = cached_preview.head(preview_rows)
                    else:
                        preview_data = data.head(preview_rows)
                        self.set_cached_preview(data, cache_key, preview_data)

                    st_obj.dataframe(preview_data, use_container_width=True)
            else:
                # 小文件直接预览
                preview_rows = st_obj.slider(
                    "预览行数",
                    min_value=5,
                    max_value=min(50, len(data)),
                    value=min(10, len(data)),
                    key=f"{self.component_name}_preview_rows"
                )
                st_obj.dataframe(data.head(preview_rows), use_container_width=True)
        
        elif preview_mode == "详细信息":
            # 基本信息
            self.render_basic_info(st_obj, data)
            
            # 列信息
            st_obj.markdown("**列详细信息：**")
            self.render_column_info(st_obj, data)
        
        elif preview_mode == "统计摘要":
            # 统计摘要
            self.render_statistical_summary(st_obj, data)
        
        elif preview_mode == "数据筛选":
            # 数据筛选
            filtered_data = self.render_data_filter(st_obj, data)
            
            # 显示筛选后的数据
            st_obj.markdown("**筛选后的数据：**")
            st_obj.dataframe(filtered_data, use_container_width=True)
            
            # 下载功能
            self.render_data_download(st_obj, filtered_data, "filtered_data")
            
            return filtered_data
        
        # 下载功能（除了筛选模式外的其他模式）
        if preview_mode != "数据筛选":
            self.render_data_download(st_obj, data, "preview_data")
        
        return data

    def render_final_data_generation(self, st_obj, data: pd.DataFrame, **kwargs) -> Optional[pd.DataFrame]:
        """渲染最终数据生成界面"""

        st_obj.markdown("#### 最终数据生成与处理")
        st_obj.caption("对数据进行最终处理，包括列选择、时间筛选和数据完整性检查")

        if data is None or data.empty:
            st_obj.info("请先提供要处理的数据")
            return None

        # 列选择
        col1, col2 = st_obj.columns(2)

        with col1:
            selected_columns = st_obj.multiselect(
                "选择要保留的列:",
                options=list(data.columns),
                default=list(data.columns),
                key=f"{self.component_name}_final_columns"
            )

        with col2:
            # 时间列识别
            time_column = None
            for col in data.columns:
                try:
                    pd.to_datetime(data[col], errors='raise')
                    time_column = col
                    break
                except:
                    continue

            if time_column:
                st_obj.success(f"[SUCCESS] 识别到时间列: {time_column}")

                # 时间范围筛选
                time_series = pd.to_datetime(data[time_column])
                min_date = time_series.min().date()
                max_date = time_series.max().date()

                date_range = st_obj.date_input(
                    "选择时间范围:",
                    value=(min_date, max_date),
                    min_value=min_date,
                    max_value=max_date,
                    key=f"{self.component_name}_date_range"
                )
            else:
                st_obj.warning("[WARNING] 未识别到时间列")
                date_range = None

        # 生成最终数据按钮
        if st_obj.button("生成最终数据", key=f"{self.component_name}_generate_final"):
            if not selected_columns:
                st_obj.warning("请至少选择一列")
                return data

            # 处理数据
            final_data = data[selected_columns].copy()

            # 应用时间筛选
            if time_column and date_range and len(date_range) == 2:
                start_date, end_date = date_range
                time_mask = (pd.to_datetime(final_data[time_column]).dt.date >= start_date) & \
                           (pd.to_datetime(final_data[time_column]).dt.date <= end_date)
                final_data = final_data[time_mask]

            # 保存到状态
            self.set_state('final_data', final_data)
            self.set_state('final_columns', selected_columns)
            self.set_state('final_time_range', date_range)

            st_obj.success(f"[SUCCESS] 最终数据生成成功！形状: {final_data.shape}")

            # 显示最终数据预览
            st_obj.markdown("**最终数据预览：**")
            st_obj.dataframe(final_data.head(10), use_container_width=True)

            # 数据统计
            col1, col2, col3 = st_obj.columns(3)
            with col1:
                st_obj.metric("总行数", final_data.shape[0])
            with col2:
                st_obj.metric("总列数", final_data.shape[1])
            with col3:
                numeric_cols = final_data.select_dtypes(include=[np.number]).columns
                st_obj.metric("数值列", len(numeric_cols))

            # 下载功能
            self.render_data_download(st_obj, final_data, "final_data")

            return final_data

        # 显示当前最终数据（如果存在）
        current_final_data = self.get_state('final_data')
        if current_final_data is not None:
            st_obj.markdown("**当前最终数据：**")
            st_obj.dataframe(current_final_data.head(10), use_container_width=True)

            col1, col2, col3 = st_obj.columns(3)
            with col1:
                st_obj.metric("总行数", current_final_data.shape[0])
            with col2:
                st_obj.metric("总列数", current_final_data.shape[1])
            with col3:
                numeric_cols = current_final_data.select_dtypes(include=[np.number]).columns
                st_obj.metric("数值列", len(numeric_cols))

            return current_final_data

        return data


__all__ = ['DataPreviewComponent']
