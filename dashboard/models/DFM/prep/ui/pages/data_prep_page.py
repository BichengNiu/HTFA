# -*- coding: utf-8 -*-
"""
DFM数据准备页面

严格遵循前后端分离架构：
- UI层只负责用户交互和展示
- 所有业务逻辑通过API调用
- 使用统一的data_input组件
"""

import streamlit as st
import pandas as pd
from datetime import datetime, date
import logging
import io

from dashboard.core.ui.components.data_input import UnifiedDataUploadComponent
from dashboard.preview.ui import DataPreviewComponent
from dashboard.models.DFM.prep.api import (
    prepare_dfm_data,
    validate_preparation_parameters
)

logger = logging.getLogger(__name__)


def _get_state(key: str, default=None):
    """获取状态值（命名空间: data_prep）"""
    full_key = f'data_prep.{key}'
    return st.session_state.get(full_key, default)


def _set_state(key: str, value):
    """设置状态值（命名空间: data_prep）"""
    full_key = f'data_prep.{key}'
    st.session_state[full_key] = value


def _sync_dates_to_train_model():
    """将数据准备的日期设置同步到模型训练模块"""
    try:
        data_start = _get_state('param_data_start_date')
        data_end = _get_state('param_data_end_date')

        if data_start:
            st.session_state['train_model.training_start_date'] = data_start
        if data_end:
            st.session_state['train_model.validation_end_date'] = data_end

        return True
    except Exception as e:
        logger.error(f"日期同步失败: {e}")
        return False


def _render_parameter_config(st_obj):
    """渲染参数配置区域"""

    st_obj.markdown("### 数据准备参数配置")

    col1, col2 = st_obj.columns(2)

    with col1:
        st_obj.markdown("**时间范围设置**")

        default_start = date(2010, 1, 31)
        data_start_date = st_obj.date_input(
            "数据起始日期",
            value=_get_state('param_data_start_date', default_start),
            key="dfm_prep_start_date_input"
        )
        _set_state('param_data_start_date', data_start_date)

        default_end = date.today()
        data_end_date = st_obj.date_input(
            "数据结束日期",
            value=_get_state('param_data_end_date', default_end),
            key="dfm_prep_end_date_input"
        )
        _set_state('param_data_end_date', data_end_date)

    with col2:
        st_obj.markdown("**高级设置**")

        target_freq = st_obj.selectbox(
            "目标数据频率",
            options=["W-FRI", "W", "D", "M"],
            index=0,
            help="将所有数据转换为统一的频率",
            key="dfm_prep_target_freq"
        )
        _set_state('param_target_freq', target_freq)

        nan_threshold = st_obj.number_input(
            "连续NaN阈值",
            min_value=1,
            max_value=50,
            value=_get_state('param_nan_threshold', 10),
            help="允许的最大连续缺失值数量",
            key="dfm_prep_nan_threshold"
        )
        _set_state('param_nan_threshold', nan_threshold)

    st_obj.markdown("---")

    param_validation = validate_preparation_parameters(
        data_start_date=str(data_start_date),
        data_end_date=str(data_end_date),
        target_freq=target_freq
    )

    if not param_validation['is_valid']:
        for error in param_validation['errors']:
            st_obj.error(f"参数错误: {error}")
        return False

    return True


def _render_preparation_result(st_obj, result: dict):
    """渲染数据准备结果"""

    if result['status'] == 'error':
        st_obj.error(f"数据准备失败: {result['message']}")
        return

    st_obj.success(result['message'])

    metadata = result['metadata']

    col1, col2, col3, col4 = st_obj.columns(4)

    with col1:
        st_obj.metric("数据形状", f"{metadata['data_shape'][0]} × {metadata['data_shape'][1]}")

    with col2:
        time_start, time_end = metadata['time_range']
        st_obj.metric("时间范围", f"{time_start[:10]} 至 {time_end[:10]}")

    with col3:
        st_obj.metric("处理耗时", metadata['processing_time'])

    with col4:
        transform_count = len(metadata['transform_log'])
        st_obj.metric("转换变量数", transform_count)

    with st_obj.expander("查看详细信息", expanded=False):

        if metadata['transform_log']:
            st_obj.markdown("**平稳性转换日志:**")
            transform_df = pd.DataFrame([
                {'变量名': var, **info}
                for var, info in metadata['transform_log'].items()
            ])
            st_obj.dataframe(transform_df, use_container_width=True)

        if metadata['removal_log']:
            st_obj.markdown("**移除变量日志:**")
            st_obj.dataframe(pd.DataFrame(metadata['removal_log']), use_container_width=True)

        st_obj.markdown("**处理参数:**")
        st_obj.json(metadata['parameters'])


def _render_data_export(st_obj, prepared_data: pd.DataFrame):
    """渲染数据导出区域"""

    st_obj.markdown("### 数据导出")

    col1, col2 = st_obj.columns(2)

    with col1:
        csv_data = prepared_data.to_csv(encoding='utf-8-sig').encode('utf-8-sig')
        st_obj.download_button(
            label="下载为 CSV",
            data=csv_data,
            file_name=f"dfm_prepared_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            key="dfm_prep_download_csv"
        )

    with col2:
        excel_buffer = io.BytesIO()
        with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
            prepared_data.to_excel(writer, sheet_name='PreparedData')
        excel_data = excel_buffer.getvalue()

        st_obj.download_button(
            label="下载为 Excel",
            data=excel_data,
            file_name=f"dfm_prepared_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            key="dfm_prep_download_excel"
        )


def render_dfm_data_prep_page(st_obj):
    """
    渲染DFM数据准备页面

    页面流程：
    1. 文件上传（使用UnifiedDataUploadComponent）
    2. 参数配置
    3. 执行数据准备（调用后端API）
    4. 结果展示
    5. 数据导出
    """

    st_obj.markdown("## DFM 数据准备")
    st_obj.caption("上传数据文件，配置处理参数，准备用于DFM训练的数据")

    st_obj.markdown("---")

    st_obj.markdown("### 1. 数据文件上传")

    uploader = UnifiedDataUploadComponent(
        accepted_types=['xlsx', 'xls'],
        help_text="请上传包含时间序列数据的Excel文件",
        show_data_source_selector=False,
        show_staging_data_option=False,
        component_id="dfm_prep_upload",
        return_file_object=True
    )

    uploaded_file = uploader.render_file_upload_section(
        st_obj,
        show_overview=False,
        show_preview=False
    )

    if uploaded_file is None:
        st_obj.info("请上传Excel数据文件以开始数据准备")
        return

    st_obj.markdown("---")

    params_valid = _render_parameter_config(st_obj)

    if not params_valid:
        st_obj.warning("请修正参数配置错误")
        return

    st_obj.markdown("### 2. 执行数据准备")

    col1, col2, col3 = st_obj.columns([2, 1, 1])

    with col1:
        if st_obj.button("开始数据准备", type="primary", key="dfm_prep_execute_btn"):
            _set_state('execute_preparation', True)

    with col2:
        if st_obj.button("重置参数", key="dfm_prep_reset_btn"):
            for key in ['param_data_start_date', 'param_data_end_date', 'param_target_freq', 'param_nan_threshold']:
                if f'data_prep.{key}' in st.session_state:
                    del st.session_state[f'data_prep.{key}']
            st_obj.rerun()

    with col3:
        if st_obj.button("清除结果", key="dfm_prep_clear_btn"):
            _set_state('prepared_data', None)
            _set_state('preparation_result', None)
            st_obj.rerun()

    if _get_state('execute_preparation', False):
        _set_state('execute_preparation', False)

        with st_obj.spinner("正在处理数据，请稍候..."):

            result = prepare_dfm_data(
                uploaded_file=uploaded_file,
                data_start_date=str(_get_state('param_data_start_date')),
                data_end_date=str(_get_state('param_data_end_date')),
                target_freq=_get_state('param_target_freq', 'W-FRI'),
                consecutive_nan_threshold=_get_state('param_nan_threshold', 10)
            )

            _set_state('preparation_result', result)
            if result['status'] == 'success':
                _set_state('prepared_data', result['data'])
                _sync_dates_to_train_model()

    st_obj.markdown("---")

    preparation_result = _get_state('preparation_result')

    if preparation_result:
        st_obj.markdown("### 3. 处理结果")

        _render_preparation_result(st_obj, preparation_result)

        if preparation_result['status'] == 'success':
            prepared_data = _get_state('prepared_data')

            if prepared_data is not None:
                st_obj.markdown("---")

                st_obj.markdown("### 4. 数据预览")

                preview_comp = DataPreviewComponent()
                preview_comp.render_input_section(
                    st_obj,
                    data=prepared_data
                )

                st_obj.markdown("---")

                _render_data_export(st_obj, prepared_data)

                st.session_state['train_model.prepared_data'] = prepared_data
                st.session_state['train_model.transform_log'] = preparation_result['metadata']['transform_log']

                st_obj.success("数据已准备完成，可以前往「模型训练」模块开始训练")


__all__ = ['render_dfm_data_prep_page']
