# -*- coding: utf-8 -*-
"""
DFM日期范围组件

提供训练期和验证期日期选择、验证和自动计算功能
"""

import streamlit as st
import pandas as pd
import logging
from typing import Dict, Any, Optional, List, Tuple
from datetime import date, datetime, timedelta

from dashboard.ui.components.dfm.base import DFMComponent, DFMServiceManager
from dashboard.core import get_global_dfm_manager


logger = logging.getLogger(__name__)


class DateRangeComponent(DFMComponent):
    """DFM日期范围组件"""
    
    def __init__(self, service_manager: Optional[DFMServiceManager] = None):
        """
        初始化日期范围组件
        
        Args:
            service_manager: DFM服务管理器
        """
        super().__init__(service_manager)
        self._default_training_years = 5  # 默认训练期年数
        self._default_validation_months = 6  # 默认验证期月数
    
    def get_component_id(self) -> str:
        """获取组件ID"""
        return "date_range"
    
    def get_state_keys(self) -> list:
        """
        获取组件相关的状态键
        
        Returns:
            List[str]: 状态键列表
        """
        return [
            'dfm_training_start_date',
            'dfm_validation_start_date',
            'dfm_validation_end_date'
        ]
    
    def validate_input(self, data: Dict) -> bool:
        """
        验证输入数据
        
        Args:
            data: 输入数据字典，包含training_data, data_prep_dates
            
        Returns:
            bool: 验证是否通过
        """
        try:
            # 检查训练数据
            training_data = data.get('training_data')
            if training_data is None:
                logger.warning("缺少训练数据")
                return False
            
            if not isinstance(training_data, pd.DataFrame):
                logger.warning("训练数据不是DataFrame格式")
                return False
            
            if training_data.empty:
                logger.warning("训练数据为空")
                return False
            
            # 检查是否有时间索引
            if not isinstance(training_data.index, pd.DatetimeIndex):
                logger.warning("训练数据没有时间索引")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"输入验证失败: {e}")
            return False
    
    def handle_service_error(self, error: Exception) -> None:
        """
        处理服务错误
        
        Args:
            error: 异常对象
        """
        error_msg = f"日期范围服务错误: {str(error)}"
        logger.error(error_msg)
        st.error(error_msg)
    
    def render(self, st_obj, training_data: pd.DataFrame, 
               data_prep_dates: Dict[str, date] = None) -> Optional[Dict[str, Any]]:
        """
        渲染日期范围组件
        
        Args:
            st_obj: Streamlit对象
            training_data: 训练数据
            data_prep_dates: 数据准备页面的日期设置
            
        Returns:
            日期范围设置结果字典或None
        """
        try:
            # 验证输入
            input_data = {
                'training_data': training_data,
                'data_prep_dates': data_prep_dates or {}
            }
            
            if not self.validate_input(input_data):
                st_obj.error("[ERROR] 输入数据验证失败，请检查训练数据是否有效。")
                return None
            
            # === 与老代码第1126-1292行完全一致的日期设置逻辑 ===

            # 计算基于数据的智能默认值 - 与老代码第1131-1214行一致
            date_defaults = self._get_data_based_date_defaults_legacy(training_data, data_prep_dates)

            # 使用统一状态管理器检查是否有数据并获取数据 - 与老代码第1219-1257行一致
            self._update_date_states_from_data_legacy(training_data, date_defaults)

            # 执行日期参数一致性验证 - 与老代码第1262-1265行一致
            self._validate_date_consistency_legacy()

            # 渲染日期输入控件 - 与老代码第1267-1292行完全一致
            date_result = self._render_date_inputs_legacy(st_obj, date_defaults)

            return date_result

        except Exception as e:
            self.handle_service_error(e)
            return None

    # === 与老代码完全一致的辅助方法 ===

    def _get_data_based_date_defaults_legacy(self, training_data: pd.DataFrame,
                                           data_prep_dates: Dict[str, Any]) -> Dict[str, Any]:
        """计算基于数据的智能默认值 - 与老代码第1131-1214行一致"""
        try:
            # 从统一状态管理器获取数据准备阶段的日期
            data_start_date = self._get_state('dfm_param_data_start_date')
            data_end_date = self._get_state('dfm_param_data_end_date')

            # 如果没有从状态管理器获取到日期，尝试从参数获取
            if not data_start_date and data_prep_dates:
                data_start_date = data_prep_dates.get('start_date')
            if not data_end_date and data_prep_dates:
                data_end_date = data_prep_dates.get('end_date')

            # 如果仍然没有日期，尝试从数据中推断
            if not data_start_date or not data_end_date:
                if training_data is not None and not training_data.empty:
                    # 查找日期列
                    date_columns = [col for col in training_data.columns if 'date' in col.lower()]
                    if date_columns:
                        date_col = date_columns[0]
                        try:
                            training_data[date_col] = pd.to_datetime(training_data[date_col])
                            if not data_start_date:
                                data_start_date = training_data[date_col].min().date()
                            if not data_end_date:
                                data_end_date = training_data[date_col].max().date()
                        except Exception as e:
                            logger.warning(f"无法解析日期列 {date_col}: {e}")

            # 设置默认值
            if data_start_date and data_end_date:
                # 计算训练期和验证期的默认分割点
                total_days = (data_end_date - data_start_date).days
                training_days = int(total_days * 0.8)  # 80%用于训练

                training_start_date = data_start_date
                validation_start_date = data_start_date + timedelta(days=training_days)
                validation_end_date = data_end_date
            else:
                # 如果无法获取数据日期，使用固定默认值
                from datetime import date
                training_start_date = date(2020, 1, 1)
                validation_start_date = date(2023, 1, 1)
                validation_end_date = date(2023, 12, 31)

            return {
                'training_start_date': training_start_date,
                'validation_start_date': validation_start_date,
                'validation_end_date': validation_end_date,
                'data_start_date': data_start_date,
                'data_end_date': data_end_date
            }

        except Exception as e:
            logger.error(f"计算默认日期失败: {e}")
            # 返回固定默认值
            from datetime import date
            return {
                'training_start_date': date(2020, 1, 1),
                'validation_start_date': date(2023, 1, 1),
                'validation_end_date': date(2023, 12, 31),
                'data_start_date': None,
                'data_end_date': None
            }

    def _update_date_states_from_data_legacy(self, training_data: pd.DataFrame,
                                           date_defaults: Dict[str, Any]):
        """使用统一状态管理器检查是否有数据并获取数据 - 与老代码第1219-1257行一致"""
        try:
            # 检查是否有训练数据
            if training_data is not None and not training_data.empty:
                # 初始化日期状态（如果尚未设置）
                if self._get_state('dfm_training_start_date') is None:
                    self._set_state('dfm_training_start_date', date_defaults['training_start_date'])

                if self._get_state('dfm_validation_start_date') is None:
                    self._set_state('dfm_validation_start_date', date_defaults['validation_start_date'])

                if self._get_state('dfm_validation_end_date') is None:
                    self._set_state('dfm_validation_end_date', date_defaults['validation_end_date'])

        except Exception as e:
            logger.error(f"更新日期状态失败: {e}")

    def _validate_date_consistency_legacy(self):
        """执行日期参数一致性验证 - 与老代码第1262-1265行一致"""
        try:
            training_start = self._get_state('dfm_training_start_date')
            validation_start = self._get_state('dfm_validation_start_date')
            validation_end = self._get_state('dfm_validation_end_date')

            # 基本验证逻辑
            if training_start and validation_start and training_start >= validation_start:
                logger.warning("训练开始日期应早于验证开始日期")

            if validation_start and validation_end and validation_start >= validation_end:
                logger.warning("验证开始日期应早于验证结束日期")

        except Exception as e:
            logger.error(f"日期一致性验证失败: {e}")

    def _render_date_inputs_legacy(self, st_obj, date_defaults: Dict[str, Any]) -> Dict[str, Any]:
        """渲染日期输入控件 - 与老代码第1267-1292行完全一致"""
        try:
            # 获取当前状态
            current_training_start = self._get_state('dfm_training_start_date', date_defaults['training_start_date'])
            current_validation_start = self._get_state('dfm_validation_start_date', date_defaults['validation_start_date'])
            current_validation_end = self._get_state('dfm_validation_end_date', date_defaults['validation_end_date'])

            # 渲染训练开始日期
            training_start_date = st_obj.date_input(
                "训练开始日期",
                value=current_training_start,
                key="new_ss_dfm_training_start_date",
                help="选择模型训练的开始日期"
            )
            self._set_state('dfm_training_start_date', training_start_date)

            # 渲染验证期日期
            validation_start_date = st_obj.date_input(
                "验证开始日期",
                value=current_validation_start,
                key="new_ss_dfm_validation_start_date",
                help="选择模型验证的开始日期"
            )
            self._set_state('dfm_validation_start_date', validation_start_date)

            validation_end_date = st_obj.date_input(
                "验证结束日期",
                value=current_validation_end,
                key="new_ss_dfm_validation_end_date",
                help="选择模型验证的结束日期"
            )
            self._set_state('dfm_validation_end_date', validation_end_date)

            return {
                'training_start_date': training_start_date,
                'validation_start_date': validation_start_date,
                'validation_end_date': validation_end_date
            }

        except Exception as e:
            logger.error(f"渲染日期输入控件失败: {e}")
            return {}

    def _calculate_default_dates(self, training_data: pd.DataFrame,
                               data_prep_dates: Dict[str, date]) -> Dict[str, date]:
        """
        计算默认日期
        
        Args:
            training_data: 训练数据
            data_prep_dates: 数据准备页面的日期设置
            
        Returns:
            默认日期字典
        """
        try:
            today = datetime.now().date()
            
            # 获取数据的时间范围
            data_start = training_data.index.min().date()
            data_end = training_data.index.max().date()
            
            # 优先使用数据准备页面设置的边界
            prep_start = data_prep_dates.get('data_start_date')
            prep_end = data_prep_dates.get('data_end_date')
            
            # 计算训练开始日期
            if prep_start:
                training_start = prep_start
            else:
                # 使用数据开始日期，但不早于合理的历史范围
                reasonable_start = datetime(2020, 1, 1).date()
                training_start = max(data_start, reasonable_start)
            
            # 计算验证期日期
            if prep_start and prep_end:
                # 基于数据准备边界计算：80%用于训练，20%用于验证
                total_days = (prep_end - prep_start).days
                training_days = int(total_days * 0.8)
                validation_start = prep_start + timedelta(days=training_days)
                validation_end = prep_end
            else:
                # 基于实际数据计算
                total_days = (data_end - data_start).days
                training_days = int(total_days * 0.8)
                validation_start = data_start + timedelta(days=training_days)
                validation_end = data_end
            
            # 确保验证期不包含未来日期
            if validation_start > today:
                validation_start = today - timedelta(days=90)  # 3个月前
            
            if validation_end > today:
                validation_end = datetime(2024, 12, 31).date()  # 强制使用2024年底
            
            # 验证日期逻辑的合理性
            if validation_start >= validation_end:
                validation_end = datetime(2024, 12, 31).date()
                validation_start = validation_end - timedelta(days=90)  # 验证期3个月
            
            return {
                'training_start': training_start,
                'validation_start': validation_start,
                'validation_end': validation_end
            }
            
        except Exception as e:
            logger.error(f"计算默认日期失败: {e}")
            # 返回静态默认值
            today = datetime.now().date()
            return {
                'training_start': datetime(today.year - self._default_training_years, 1, 1).date(),
                'validation_start': datetime(2024, 7, 1).date(),
                'validation_end': datetime(2024, 12, 31).date()
            }
    
    def _validate_and_correct_dates(self, date_defaults: Dict[str, date], 
                                  data_prep_dates: Dict[str, date]) -> None:
        """
        验证并修正现有的日期设置
        
        Args:
            date_defaults: 默认日期
            data_prep_dates: 数据准备页面的日期设置
        """
        try:
            # 检查并更新训练开始日期
            current_training_start = self._get_state('dfm_training_start_date')
            if (current_training_start is None or 
                current_training_start == datetime(2010, 1, 1).date()):
                self._set_state('dfm_training_start_date', date_defaults['training_start'])
            
            # 检查并更新验证开始日期
            current_validation_start = self._get_state('dfm_validation_start_date')
            if (current_validation_start is None or 
                current_validation_start == datetime(2020, 12, 31).date()):
                self._set_state('dfm_validation_start_date', date_defaults['validation_start'])
            
            # 检查并更新验证结束日期
            current_validation_end = self._get_state('dfm_validation_end_date')
            if (current_validation_end is None or 
                current_validation_end == datetime(2022, 12, 31).date()):
                self._set_state('dfm_validation_end_date', date_defaults['validation_end'])
                
        except Exception as e:
            logger.error(f"验证和修正日期失败: {e}")
    
    def _render_data_range_info(self, st_obj, training_data: pd.DataFrame) -> None:
        """
        渲染数据范围信息
        
        Args:
            st_obj: Streamlit对象
            training_data: 训练数据
        """
        try:
            data_start = training_data.index.min().strftime('%Y-%m-%d')
            data_end = training_data.index.max().strftime('%Y-%m-%d')
            data_count = len(training_data.index)
            
            st_obj.info(f"[DATA] 可用数据范围: {data_start} 至 {data_end} ({data_count} 个时间点)")
            
        except Exception as e:
            logger.error(f"渲染数据范围信息失败: {e}")
    
    def _render_training_start_date(self, st_obj, default_date: date) -> date:
        """
        渲染训练开始日期选择
        
        Args:
            st_obj: Streamlit对象
            default_date: 默认日期
            
        Returns:
            选择的训练开始日期
        """
        current_value = self._get_state('dfm_training_start_date', default_date)
        
        training_start = st_obj.date_input(
            "训练期开始日期 (Training Start Date)",
            value=current_value,
            key=f"{self.get_state_key_prefix()}_training_start_input",
            help="选择模型训练数据的起始日期。默认为数据的第一期。"
        )
        
        # 更新状态
        self._set_state('dfm_training_start_date', training_start)
        
        return training_start
    
    def _render_validation_dates(self, st_obj, date_defaults: Dict[str, date]) -> Tuple[date, date]:
        """
        渲染验证期日期选择
        
        Args:
            st_obj: Streamlit对象
            date_defaults: 默认日期字典
            
        Returns:
            (验证开始日期, 验证结束日期)
        """
        # 验证开始日期
        validation_start_value = self._get_state(
            'dfm_validation_start_date', 
            date_defaults['validation_start']
        )
        
        validation_start = st_obj.date_input(
            "验证期开始日期 (Validation Start Date)",
            value=validation_start_value,
            key=f"{self.get_state_key_prefix()}_validation_start_input",
            help="选择验证期开始日期。默认为训练期结束后。"
        )
        
        # 验证结束日期
        validation_end_value = self._get_state(
            'dfm_validation_end_date', 
            date_defaults['validation_end']
        )
        
        validation_end = st_obj.date_input(
            "验证期结束日期 (Validation End Date)",
            value=validation_end_value,
            key=f"{self.get_state_key_prefix()}_validation_end_input",
            help="选择验证期结束日期。默认为数据的最后一期。"
        )
        
        # 更新状态
        self._set_state('dfm_validation_start_date', validation_start)
        self._set_state('dfm_validation_end_date', validation_end)
        
        return validation_start, validation_end

    def _validate_date_consistency(self, dates: Dict[str, date],
                                 data_prep_dates: Dict[str, date]) -> Tuple[bool, List[str]]:
        """
        验证日期一致性

        Args:
            dates: 当前日期设置
            data_prep_dates: 数据准备页面的日期设置

        Returns:
            (是否有效, 错误列表)
        """
        errors = []

        try:
            training_start = dates['training_start']
            validation_start = dates['validation_start']
            validation_end = dates['validation_end']

            # 1. 基本逻辑验证
            if training_start >= validation_start:
                errors.append(f"训练开始日期 ({training_start}) 必须早于验证开始日期 ({validation_start})")

            if validation_start >= validation_end:
                errors.append(f"验证开始日期 ({validation_start}) 必须早于验证结束日期 ({validation_end})")

            # 2. 验证日期边界
            boundary_valid, boundary_errors = self._validate_date_boundaries(dates, data_prep_dates)
            errors.extend(boundary_errors)

            # 3. 验证期长度检查
            if validation_start < validation_end:
                validation_days = (validation_end - validation_start).days
                if validation_days < 30:
                    errors.append(f"验证期过短 ({validation_days} 天)，建议至少30天")
                elif validation_days > 730:  # 2年
                    errors.append(f"验证期过长 ({validation_days} 天)，建议不超过2年")

            # 4. 训练期长度检查
            if training_start < validation_start:
                training_days = (validation_start - training_start).days
                if training_days < 365:  # 1年
                    errors.append(f"训练期过短 ({training_days} 天)，建议至少1年")

            return len(errors) == 0, errors

        except Exception as e:
            logger.error(f"日期一致性验证失败: {e}")
            return False, [f"日期验证过程中发生错误: {e}"]

    def _validate_date_boundaries(self, dates: Dict[str, date],
                                data_prep_dates: Dict[str, date]) -> Tuple[bool, List[str]]:
        """
        验证日期边界

        Args:
            dates: 当前日期设置
            data_prep_dates: 数据准备页面的日期设置

        Returns:
            (是否有效, 错误列表)
        """
        errors = []

        try:
            prep_start = data_prep_dates.get('data_start_date')
            prep_end = data_prep_dates.get('data_end_date')

            if prep_start:
                if dates['training_start'] < prep_start:
                    errors.append(
                        f"训练开始日期 ({dates['training_start']}) "
                        f"不能早于数据准备页面设置的开始边界 ({prep_start})"
                    )

                if dates['validation_start'] < prep_start:
                    errors.append(
                        f"验证开始日期 ({dates['validation_start']}) "
                        f"不能早于数据准备页面设置的开始边界 ({prep_start})"
                    )

            if prep_end:
                if dates['validation_end'] > prep_end:
                    errors.append(
                        f"验证结束日期 ({dates['validation_end']}) "
                        f"不能晚于数据准备页面设置的结束边界 ({prep_end})"
                    )

                if dates['training_start'] > prep_end:
                    errors.append(
                        f"训练开始日期 ({dates['training_start']}) "
                        f"不能晚于数据准备页面设置的结束边界 ({prep_end})"
                    )

            return len(errors) == 0, errors

        except Exception as e:
            logger.error(f"日期边界验证失败: {e}")
            return False, [f"边界验证过程中发生错误: {e}"]

    def _auto_correct_dates(self, dates: Dict[str, date]) -> Dict[str, date]:
        """
        自动修正日期设置

        Args:
            dates: 当前日期设置

        Returns:
            修正后的日期设置
        """
        try:
            corrected = dates.copy()

            # 确保训练开始日期早于验证开始日期
            if corrected['training_start'] >= corrected['validation_start']:
                # 将验证开始日期设置为训练开始日期后6个月
                corrected['validation_start'] = corrected['training_start'] + timedelta(days=180)

            # 确保验证开始日期早于验证结束日期
            if corrected['validation_start'] >= corrected['validation_end']:
                # 将验证结束日期设置为验证开始日期后3个月
                corrected['validation_end'] = corrected['validation_start'] + timedelta(days=90)

            # 更新状态
            for key, value in corrected.items():
                self._set_state(f'dfm_{key}', value)

            logger.info(f"日期已自动修正: {corrected}")
            return corrected

        except Exception as e:
            logger.error(f"自动修正日期失败: {e}")
            return dates

    def _render_date_summary(self, st_obj, dates: Dict[str, date]) -> None:
        """
        渲染日期摘要

        Args:
            st_obj: Streamlit对象
            dates: 日期设置
        """
        try:
            st_obj.markdown("---")
            st_obj.markdown("**[INFO] 日期设置摘要**")

            # 基本信息
            st_obj.text(f"训练期开始: {dates['training_start']}")
            st_obj.text(f"验证期开始: {dates['validation_start']}")
            st_obj.text(f"验证期结束: {dates['validation_end']}")

            # 计算期间长度
            training_days = self._calculate_training_period_length(dates)
            validation_days = self._calculate_validation_period_length(dates)

            st_obj.text(f"训练期长度: {training_days} 天 ({training_days // 365} 年 {(training_days % 365) // 30} 个月)")
            st_obj.text(f"验证期长度: {validation_days} 天 ({validation_days // 30} 个月)")

            # 显示训练/验证比例
            total_days = training_days + validation_days
            if total_days > 0:
                training_ratio = training_days / total_days * 100
                validation_ratio = validation_days / total_days * 100
                st_obj.text(f"训练/验证比例: {training_ratio:.1f}% / {validation_ratio:.1f}%")

        except Exception as e:
            logger.error(f"渲染日期摘要失败: {e}")

    def _calculate_training_period_length(self, dates: Dict[str, date]) -> int:
        """
        计算训练期长度（天数）

        Args:
            dates: 日期设置

        Returns:
            训练期天数
        """
        try:
            return (dates['validation_start'] - dates['training_start']).days
        except Exception as e:
            logger.error(f"计算训练期长度失败: {e}")
            return 0

    def _calculate_validation_period_length(self, dates: Dict[str, date]) -> int:
        """
        计算验证期长度（天数）

        Args:
            dates: 日期设置

        Returns:
            验证期天数
        """
        try:
            return (dates['validation_end'] - dates['validation_start']).days
        except Exception as e:
            logger.error(f"计算验证期长度失败: {e}")
            return 0

    def _get_state(self, key: str, default: Any = None) -> Any:
        """获取状态值"""
        try:
            dfm_manager = get_global_dfm_manager()
            if dfm_manager:
                value = dfm_manager.get_dfm_state('train_model', key, None)
                if value is not None:
                    return value

            # 如果DFM管理器不可用，返回默认值
            logger.warning(f"DFM状态管理器不可用，返回默认值: {key}")
            return default

        except Exception as e:
            logger.warning(f"获取状态失败: {e}")
            return default

    def _set_state(self, key: str, value: Any) -> None:
        """设置状态值"""
        try:
            dfm_manager = get_global_dfm_manager()
            if dfm_manager:
                success = dfm_manager.set_dfm_state('train_model', key, value)
                if not success:
                    logger.error(f"设置DFM状态失败: {key}")
            else:
                logger.error(f"DFM统一状态管理器不可用，无法设置状态: {key}")

        except Exception as e:
            logger.error(f"设置状态失败: {e}")
