"""
DFM数据预处理核心处理器（简化版）

设计原则：
1. 单一流程，7个清晰步骤
2. 映射表只加载一次
3. 智能缺失值检测（根据频率关系选择检测时机）
4. 消除重复代码和多次合并
5. 遵循KISS、DRY、SRP原则

重构时间：2025-11-13
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Set
from datetime import datetime
import logging

from dashboard.models.DFM.prep.modules.data_loader import DataLoader
from dashboard.models.DFM.prep.modules.data_aligner import DataAligner
from dashboard.models.DFM.prep.modules.data_cleaner import DataCleaner, clean_dataframe
from dashboard.models.DFM.prep.utils.date_utils import standardize_date
from dashboard.models.DFM.utils.text_utils import normalize_text

logger = logging.getLogger(__name__)


class DataPreparationProcessor:
    """DFM数据预处理核心处理器

    实现7步简化流程：
    1. 读取映射表（由API层完成，作为参数传入）
    2. 统计所有数据的时间范围
    3. 读取UI配置，框定时间范围
    4. 按原始频率检测缺失值并剔除变量（智能检测）
    5. 对齐到目标频率
    6. 合并数据形成最终表
    7. 输出处理结果
    """

    # 频率优先级顺序（数字越小频率越高）
    FREQ_ORDER = {
        'D': 1,     # 日度
        'W': 2,     # 周度
        '10D': 2.5, # 旬度（2025-11-15新增）
        'M': 3,     # 月度
        'Q': 4,     # 季度
        'Y': 5      # 年度
    }

    def __init__(
        self,
        excel_path: str,
        target_variable_name: str,
        var_industry_map: Dict[str, str],
        var_frequency_map: Dict[str, str],
        target_freq: str = 'W-FRI',
        consecutive_nan_threshold: int = 10,
        data_start_date: Optional[str] = None,
        data_end_date: Optional[str] = None,
        enable_borrowing: bool = True,
        enable_freq_alignment: bool = True,
        zero_handling: str = 'missing',
        var_publication_lag_map: Dict[str, int] = None,
        enable_publication_calibration: bool = False
    ):
        """初始化处理器

        Args:
            excel_path: Excel文件路径
            target_variable_name: 目标变量名称（可选，用于将其放在第一列）
            var_industry_map: 变量-行业映射字典（从指标体系加载）
            var_frequency_map: 变量-频率映射字典（从指标体系加载）
            target_freq: 目标频率，默认'W-FRI'
            consecutive_nan_threshold: 连续缺失值阈值
            data_start_date: 数据起始日期
            data_end_date: 数据结束日期
            enable_borrowing: 是否启用数据借调，默认True
            enable_freq_alignment: 是否启用频率对齐，默认True。选择False时保留原始日期
            zero_handling: 零值处理方式，'none'不处理，'missing'转为缺失值，'adjust'调正为1
            var_publication_lag_map: 变量-发布日期滞后映射（从指标体系加载）
            enable_publication_calibration: 是否启用发布日期校准，默认False
        """
        # 仅在启用频率对齐时验证目标频率
        if enable_freq_alignment and not target_freq.upper().endswith('-FRI'):
            raise ValueError(f"当前仅支持周五对齐 (W-FRI)，提供的频率 '{target_freq}' 无效")

        self.excel_path = excel_path
        self.target_variable_name = target_variable_name
        self.var_industry_map = var_industry_map
        self.var_frequency_map = var_frequency_map
        self.target_freq = target_freq
        self.consecutive_nan_threshold = consecutive_nan_threshold
        self.data_start_date = standardize_date(data_start_date)
        self.data_end_date = standardize_date(data_end_date)
        self.enable_borrowing = enable_borrowing
        self.enable_freq_alignment = enable_freq_alignment
        self.zero_handling = zero_handling
        self.var_publication_lag_map = var_publication_lag_map or {}
        self.enable_publication_calibration = enable_publication_calibration

        # 初始化组件
        self.data_loader = DataLoader(
            reference_industry_map=var_industry_map,
            reference_frequency_map=var_frequency_map
        )
        # DataAligner仅在启用频率对齐���使用
        if enable_freq_alignment:
            self.data_aligner = DataAligner(target_freq, enable_borrowing=enable_borrowing)
        else:
            self.data_aligner = None
        self.data_cleaner = DataCleaner()

        # 日志记录
        self.removal_log = []
        self.transform_log = {}

        logger.info(f"[Processor] 初始化完成: 目标变量={target_variable_name}, 频率对齐={'启用' if enable_freq_alignment else '禁用'}")
        logger.info(f"[Processor] 零值处理={zero_handling}, 发布日期校准={'启用' if enable_publication_calibration else '禁用'}")

    def _apply_global_preprocessing(self, df: pd.DataFrame) -> pd.DataFrame:
        """应用全局零值处理

        在数据加载后、频率对齐前统一应用，确保所有后续处理都基于相同的预处理结果。

        Args:
            df: 输入DataFrame

        Returns:
            pd.DataFrame: 预处理后的DataFrame
        """
        if df is None or df.empty:
            return df

        result = df.copy()

        # 零值处理
        if self.zero_handling == 'missing':
            zero_mask = result == 0
            zero_count = zero_mask.sum().sum()
            result = result.replace(0, np.nan)
            if zero_count > 0:
                logger.info(f"  [全局预处理] 零值处理(缺失值): {zero_count}个零值转为NaN")
        elif self.zero_handling == 'adjust':
            zero_mask = result == 0
            zero_count = zero_mask.sum().sum()
            result = result.replace(0, 1)
            if zero_count > 0:
                logger.info(f"  [全局预处理] 零值处理(调正): {zero_count}个零值调正为1")
        # 'none' 不处理

        return result

    def _apply_publication_date_calibration(self, df: pd.DataFrame, freq_type: str) -> pd.DataFrame:
        """应用发布日期校准

        根据频率类型采用不同的校准逻辑：
        - 日度/周度：数据日期 + 滞后天数 = 实际发布日期
        - 月度：对齐到当月最后一个周五（周度目标）或当月最后一日（日度目标）
        - 季度：对齐到当季最后一个周五（周度目标）或当季最后一日（日度目标）
        - 年度：对齐到当年最后一个周五（周度目标）或当年最后一日（日度目标）

        Args:
            df: 输入DataFrame，索引为数据日期
            freq_type: 频率类型 ('daily', 'weekly', 'dekad', 'monthly', 'quarterly', 'yearly')

        Returns:
            pd.DataFrame: 索引为校准后日期的DataFrame
        """
        if not self.enable_publication_calibration or not self.var_publication_lag_map:
            return df

        if df is None or df.empty:
            return df

        logger.info(f"  应用发布日期校准 ({freq_type})...")
        calibration_count = 0

        # 创建结果DataFrame
        result_columns = {}

        for col in df.columns:
            col_norm = normalize_text(col)
            lag_days = self.var_publication_lag_map.get(col_norm)

            if lag_days is not None and lag_days != 0:
                # 提取该列数据
                series = df[col].dropna()
                if series.empty:
                    result_columns[col] = df[col]
                    continue

                # 根据频率类型计算校准后的日期
                if freq_type in ['daily', 'weekly', 'dekad']:
                    # 日度/周度/旬度：直接加滞后天数
                    new_index = series.index + pd.Timedelta(days=lag_days)
                elif freq_type == 'monthly':
                    # 月度：对齐到当月最后一个周五
                    new_index = series.index.map(lambda d: self._get_month_end_friday(d, lag_days))
                elif freq_type == 'quarterly':
                    # 季度：对齐到当季最后一个周五
                    new_index = series.index.map(lambda d: self._get_quarter_end_friday(d, lag_days))
                elif freq_type == 'yearly':
                    # 年度：对齐到当年最后一个周五
                    new_index = series.index.map(lambda d: self._get_year_end_friday(d, lag_days))
                else:
                    # 未知频率：直接加滞后天数
                    new_index = series.index + pd.Timedelta(days=lag_days)

                calibrated_series = pd.Series(series.values, index=new_index, name=col)
                result_columns[col] = calibrated_series
                calibration_count += 1
                logger.debug(f"    {col}: 滞后{lag_days}天 ({freq_type})")
            else:
                # 不需要校准的变量，保持原样
                result_columns[col] = df[col]

        if calibration_count > 0:
            # 合并所有列，自动扩展索引
            result_df = pd.DataFrame(result_columns)
            # 重新排序索引
            result_df = result_df.sort_index()
            logger.info(f"  发布日期校准完成: {calibration_count}个变量 ({freq_type})")
            return result_df
        else:
            return df

    def _get_month_end_friday(self, data_date: pd.Timestamp, lag_days: int) -> pd.Timestamp:
        """获取发布日期附近的周五（在发布月份内，不跨月）

        逻辑：
        1. 数据日期 + 滞后天数 = 发布日期
        2. 对齐到发布日期最近的周五
        3. 如果最近周五跨月，则调整到发布月份内的周五

        Args:
            data_date: 数据日期（如月度数据的月末日期）
            lag_days: 滞后天数（如15表示下月15日发布）

        Returns:
            pd.Timestamp: 发布日期附近的周五（在发布月份内）
        """
        from calendar import monthrange

        # 实际发布日期 = 数据日期 + 滞后天数
        pub_date = data_date + pd.Timedelta(days=lag_days)
        year, month = pub_date.year, pub_date.month
        weekday = pub_date.weekday()

        # 计算最近周五
        if weekday <= 4:  # 周一到周五
            days_to_friday = 4 - weekday
        else:  # 周六日
            days_to_friday = -(weekday - 4)

        target_friday = pub_date + pd.Timedelta(days=days_to_friday)

        # 检查是否跨月，如果跨月则调整到发布月份内的周五
        if target_friday.month != month:
            if target_friday.month > month or (target_friday.month == 1 and month == 12):
                # 跨到下个月 -> 发布月份的最后一个周五
                last_day = monthrange(year, month)[1]
                last_date = pd.Timestamp(year, month, last_day)
                last_wd = last_date.weekday()
                if last_wd >= 4:
                    days_back = last_wd - 4
                else:
                    days_back = last_wd + 3
                target_friday = last_date - pd.Timedelta(days=days_back)
            else:
                # 跨到上个月 -> 发布月份的第一个周五
                first_date = pd.Timestamp(year, month, 1)
                first_wd = first_date.weekday()
                if first_wd <= 4:
                    days_forward = 4 - first_wd
                else:
                    days_forward = 11 - first_wd
                target_friday = first_date + pd.Timedelta(days=days_forward)

        return target_friday

    def _get_quarter_end_friday(self, data_date: pd.Timestamp, lag_days: int) -> pd.Timestamp:
        """获取发布日期附近的周五（在发布季度内，不跨季）

        Args:
            data_date: 数据日期
            lag_days: 滞后天数

        Returns:
            pd.Timestamp: 发布日期附近的周五（在发布季度内）
        """
        from calendar import monthrange

        # 实际发布日期 = 数据日期 + 滞后天数
        pub_date = data_date + pd.Timedelta(days=lag_days)
        year = pub_date.year
        month = pub_date.month
        quarter = (month - 1) // 3 + 1
        quarter_start_month = (quarter - 1) * 3 + 1
        quarter_end_month = quarter * 3
        weekday = pub_date.weekday()

        # 计算最近周五
        if weekday <= 4:
            days_to_friday = 4 - weekday
        else:
            days_to_friday = -(weekday - 4)

        target_friday = pub_date + pd.Timedelta(days=days_to_friday)
        target_quarter = (target_friday.month - 1) // 3 + 1

        # 检查是否跨季
        if target_friday.year != year or target_quarter != quarter:
            if (target_friday.year > year) or (target_friday.year == year and target_quarter > quarter):
                # 跨到下个季度 -> 发布季度的最后一个周五
                last_day = monthrange(year, quarter_end_month)[1]
                last_date = pd.Timestamp(year, quarter_end_month, last_day)
                last_wd = last_date.weekday()
                if last_wd >= 4:
                    days_back = last_wd - 4
                else:
                    days_back = last_wd + 3
                target_friday = last_date - pd.Timedelta(days=days_back)
            else:
                # 跨到上个季度 -> 发布季度的第一个周五
                first_date = pd.Timestamp(year, quarter_start_month, 1)
                first_wd = first_date.weekday()
                if first_wd <= 4:
                    days_forward = 4 - first_wd
                else:
                    days_forward = 11 - first_wd
                target_friday = first_date + pd.Timedelta(days=days_forward)

        return target_friday

    def _get_year_end_friday(self, data_date: pd.Timestamp, lag_days: int) -> pd.Timestamp:
        """获取发布日期附近的周五（在发布年份内，不跨年）

        逻辑：
        1. 数据日期 + 滞后天数 = 发布日期
        2. 对齐到发布日期最近的周五
        3. 如果最近周五跨年，则调整到发布年份内的周五

        Args:
            data_date: 数据日期
            lag_days: 滞后天数

        Returns:
            pd.Timestamp: 发布日期附近的周五（在发布年份内）
        """
        # 实际发布日期 = 数据日期 + 滞后天数
        pub_date = data_date + pd.Timedelta(days=lag_days)
        year = pub_date.year
        weekday = pub_date.weekday()

        # 计算最近周五
        if weekday <= 4:  # 周一到周五
            days_to_friday = 4 - weekday
        else:  # 周六日
            days_to_friday = -(weekday - 4)

        target_friday = pub_date + pd.Timedelta(days=days_to_friday)

        # 检查是否跨年，如果跨年则调整到发布年份内的周五
        if target_friday.year != year:
            if target_friday.year > year:
                # 跨到下一年 -> 发布年份的最后一个周五
                year_end = pd.Timestamp(year, 12, 31)
                year_end_wd = year_end.weekday()
                if year_end_wd >= 4:
                    days_back = year_end_wd - 4
                else:
                    days_back = year_end_wd + 3
                target_friday = year_end - pd.Timedelta(days=days_back)
            else:
                # 跨到上一年 -> 发布年份的第一个周五
                year_start = pd.Timestamp(year, 1, 1)
                year_start_wd = year_start.weekday()
                if year_start_wd <= 4:
                    days_forward = 4 - year_start_wd
                else:
                    days_forward = 11 - year_start_wd
                target_friday = year_start + pd.Timedelta(days=days_forward)

        return target_friday

    def execute(self) -> Tuple[pd.DataFrame, Dict[str, str], Dict, List[Dict]]:
        """执行完整的7步数据准备流程

        Returns:
            Tuple[pd.DataFrame, Dict, Dict, List]:
                - 处理后的DataFrame
                - 变量-行业映射
                - 转换日志（包含借调日志）
                - 移除变量日志
        """
        logger.info("\n" + "="*60)
        logger.info("DFM数据预处理流程启动（简化版）")
        logger.info("="*60)

        try:
            # 步骤2: 统计所有数据的时间范围（返回并集）
            logger.info("\n步骤2/7: 统计数据时间范围...")
            time_range = self._step2_collect_time_range()
            logger.info(f"  数据时间范围: {time_range['start']} 至 {time_range['end']}")

            # 步骤3: 应用UI配置的时间范围
            logger.info("\n步骤3/7: 应用配置的时间范围...")
            effective_start = self.data_start_date or time_range['start']
            effective_end = self.data_end_date or time_range['end']
            logger.info(f"  有效时间范围: {effective_start} 至 {effective_end}")

            # 步骤4: 加载所有数据并按原始频率分类
            logger.info("\n步骤4/7: 加载数据并按频率分类...")
            data_by_freq = self._step4_load_and_classify_data()

            # 步骤5: 智能缺失值检测并对齐到目标频率
            logger.info("\n步骤5/7: 智能缺失值检测与频率对齐...")
            aligned_data, borrowing_log = self._step5_smart_missing_detection_and_align(data_by_freq)

            # 步骤6: 合并所有数据形成最终表
            logger.info("\n步骤6/7: 合并数据...")
            final_df = self._step6_merge_all_data(aligned_data, effective_start, effective_end)

            # 步骤7: 生成输出
            logger.info("\n步骤7/7: 生成输出...")
            result = self._step7_generate_output(final_df, borrowing_log)

            logger.info("="*60)
            logger.info(f"数据预处理完成！最终形状: {result[0].shape}")
            logger.info("="*60 + "\n")

            return result

        except Exception as e:
            logger.error(f"数据预处理失败: {e}", exc_info=True)
            raise

    def _step2_collect_time_range(self) -> Dict[str, str]:
        """步骤2: 统计所有数据的时间范围（返回并集）

        Returns:
            Dict: {'start': '2020-01-01', 'end': '2024-12-31'}
        """
        excel_file = pd.ExcelFile(self.excel_path)
        all_dates = []

        for sheet_name in excel_file.sheet_names:
            # 跳过映射表
            if sheet_name == '指标体系':
                continue

            try:
                # 读取第一列作为日期列
                df = pd.read_excel(excel_file, sheet_name=sheet_name, usecols=[0])
                if df.empty:
                    continue

                # 尝试解析为日期
                dates = pd.to_datetime(df.iloc[:, 0], errors='coerce')
                valid_dates = dates.dropna()

                if not valid_dates.empty:
                    all_dates.extend(valid_dates.tolist())

            except Exception as e:
                logger.debug(f"  跳过工作表 '{sheet_name}': {e}")
                continue

        if not all_dates:
            raise ValueError("未能从任何工作表中提取有效日期")

        min_date = min(all_dates)
        max_date = max(all_dates)

        return {
            'start': min_date.strftime('%Y-%m-%d'),
            'end': max_date.strftime('%Y-%m-%d')
        }

    def _step4_load_and_classify_data(self) -> Dict[str, Dict[str, pd.DataFrame]]:
        """步骤4: 加载所有数据并按变量频率分类（统一处理，无特殊逻辑）

        Returns:
            Dict: {
                'daily': {'var1': series1, ...},
                'weekly': {'var2': series2, ...},
                'dekad': {'var3': series3, ...},
                'monthly': {'var4': series4, ...}
            }
        """
        excel_file = pd.ExcelFile(self.excel_path)
        data_by_freq = {
            'daily': {},
            'weekly': {},
            'dekad': {},
            'monthly': {},
            'quarterly': {},
            'yearly': {}
        }

        for sheet_name in excel_file.sheet_names:
            if sheet_name == '指标体系':
                continue

            logger.info(f"  加载工作表: {sheet_name}")

            try:
                # 统一读取所有sheet
                df = pd.read_excel(excel_file, sheet_name=sheet_name, header=0)
                if df.shape[1] < 2:
                    continue

                # 第一列是日期
                date_col = pd.to_datetime(df.iloc[:, 0], errors='coerce')
                valid_mask = date_col.notna()

                # 遍历每个变量列，根据映射表频率分类
                for col_idx in range(1, df.shape[1]):
                    var_name = df.columns[col_idx]
                    norm_var_name = normalize_text(var_name)

                    # 从映射表获取频率
                    freq = self.var_frequency_map.get(norm_var_name, '').lower()

                    if not freq:
                        logger.debug(f"    变量 '{var_name}' 未在映射表中找到频率，跳过")
                        continue

                    # 提取数据
                    values = pd.to_numeric(df.loc[valid_mask, var_name], errors='coerce')
                    series = pd.Series(values.values, index=date_col[valid_mask], name=var_name)

                    # 按频率分类
                    if '日' in freq or 'daily' in freq:
                        data_by_freq['daily'][var_name] = series
                    elif '周' in freq or 'weekly' in freq:
                        data_by_freq['weekly'][var_name] = series
                    elif '旬' in freq or 'dekad' in freq:
                        data_by_freq['dekad'][var_name] = series
                    elif '月' in freq or 'monthly' in freq:
                        data_by_freq['monthly'][var_name] = series
                    elif '季' in freq or 'quarterly' in freq:
                        data_by_freq['quarterly'][var_name] = series
                    elif '年' in freq or 'yearly' in freq or 'annual' in freq:
                        data_by_freq['yearly'][var_name] = series
                    else:
                        logger.warning(f"    变量 '{var_name}' 频率 '{freq}' 无法识别，跳过")

            except Exception as e:
                logger.warning(f"    加载失败: {e}")
                continue

        # 转换为DataFrame格式并应用全局预处理
        for freq_type in ['daily', 'weekly', 'dekad', 'monthly', 'quarterly', 'yearly']:
            if data_by_freq[freq_type]:
                series_list = []
                for var_name, series in data_by_freq[freq_type].items():
                    if series.index.duplicated().any():
                        series = series[~series.index.duplicated(keep='last')]
                    series_list.append(series)
                if series_list:
                    combined_df = pd.concat(series_list, axis=1)
                    # 应用全局零值和负值预处理
                    combined_df = self._apply_global_preprocessing(combined_df)
                    # 应用发布日期校准（传入频率类型）
                    combined_df = self._apply_publication_date_calibration(combined_df, freq_type)
                    data_by_freq[freq_type] = {'combined': combined_df}

        return data_by_freq

    def _step5_smart_missing_detection_and_align(
        self,
        data_by_freq: Dict[str, Dict]
    ) -> Tuple[Dict[str, pd.DataFrame], Dict]:
        """步骤5: 智能缺失值检测与频率对齐（统一处理所有变量）

        根据频率关系选择检测时机：
        - 目标频率 >= 原始频率（如日度→周度）：先检测再对齐
        - 目标频率 < 原始频率（如月度→周度）：先对齐再检测

        当enable_freq_alignment=False时：
        - 仅检测缺失值，不进行频率转换
        - 保留原始发布日期

        Args:
            data_by_freq: 按频率分类的数据

        Returns:
            Tuple[Dict, Dict]: (
                {
                    'daily': pd.DataFrame,
                    'weekly': pd.DataFrame,
                    'dekad': pd.DataFrame,
                    'monthly': pd.DataFrame
                },
                借调日志字典（聚合所有频率的借调记录）
            )
        """
        # 不对齐模式：仅检测缺失值，保留原始日期
        if not self.enable_freq_alignment:
            return self._step5_no_alignment(data_by_freq)

        aligned_data = {}
        all_borrowing_log = {}  # 聚合所有频率的借调日志
        target_level = self._get_freq_level(self.target_freq)

        # 处理日度数据
        if data_by_freq['daily']:
            logger.info("  处理日度数据...")
            aligned_df, borrowing_log = self._process_frequency_data(
                data_by_freq['daily'], 'D', target_level, 'daily'
            )
            aligned_data['daily'] = aligned_df
            if borrowing_log:
                all_borrowing_log.update(borrowing_log)

        # 处理周度数据
        if data_by_freq['weekly']:
            logger.info("  处理周度数据...")
            aligned_df, borrowing_log = self._process_frequency_data(
                data_by_freq['weekly'], 'W', target_level, 'weekly'
            )
            aligned_data['weekly'] = aligned_df
            if borrowing_log:
                all_borrowing_log.update(borrowing_log)

        # 处理旬度数据
        if data_by_freq['dekad']:
            logger.info("  处理旬度数据...")
            aligned_df, borrowing_log = self._process_frequency_data(
                data_by_freq['dekad'], 'M', target_level, 'dekad'
            )
            aligned_data['dekad'] = aligned_df
            if borrowing_log:
                all_borrowing_log.update(borrowing_log)

        # 处理月度数据
        if data_by_freq['monthly']:
            logger.info("  处理月度数据...")
            aligned_df, borrowing_log = self._process_frequency_data(
                data_by_freq['monthly'], 'M', target_level, 'monthly'
            )
            aligned_data['monthly'] = aligned_df
            if borrowing_log:
                all_borrowing_log.update(borrowing_log)

        # 处理季度数据
        if data_by_freq['quarterly']:
            logger.info("  处理季度数据...")
            aligned_df, borrowing_log = self._process_frequency_data(
                data_by_freq['quarterly'], 'Q', target_level, 'quarterly'
            )
            aligned_data['quarterly'] = aligned_df
            if borrowing_log:
                all_borrowing_log.update(borrowing_log)

        # 处理年度数据
        if data_by_freq['yearly']:
            logger.info("  处理年度数据...")
            aligned_df, borrowing_log = self._process_frequency_data(
                data_by_freq['yearly'], 'Y', target_level, 'yearly'
            )
            aligned_data['yearly'] = aligned_df
            if borrowing_log:
                all_borrowing_log.update(borrowing_log)

        return aligned_data, all_borrowing_log

    def _step5_no_alignment(
        self,
        data_by_freq: Dict[str, Dict]
    ) -> Tuple[Dict[str, pd.DataFrame], Dict]:
        """不对齐模式：仅检测缺失值，保留原始发布日期

        Args:
            data_by_freq: 按频率分类的数据

        Returns:
            Tuple[Dict, Dict]: (各频率数据字典, 空的借调日志)
        """
        logger.info("  [不对齐模式] 保留原始发布日期，仅检测缺失值...")
        result_data = {}

        for freq_name in ['daily', 'weekly', 'dekad', 'monthly', 'quarterly', 'yearly']:
            if not data_by_freq.get(freq_name):
                continue

            dfs = list(data_by_freq[freq_name].values())
            if not dfs:
                continue

            logger.info(f"    处理{freq_name}数据...")
            combined_df = pd.concat(dfs, axis=1)

            # 移除重复列
            combined_df = self.data_cleaner.remove_duplicate_columns(combined_df, f"[{freq_name}] ")
            self.removal_log.extend(self.data_cleaner.get_removed_variables_log())
            self.data_cleaner.clear_log()

            # 检测连续缺失值（在原始频率上检测）
            if self.consecutive_nan_threshold:
                cleaned_df = self.data_cleaner.handle_consecutive_nans(
                    combined_df,
                    self.consecutive_nan_threshold,
                    f"[{freq_name}原始频率] ",
                    self.data_start_date,
                    self.data_end_date
                )
                self.removal_log.extend(self.data_cleaner.get_removed_variables_log())
                self.data_cleaner.clear_log()
            else:
                cleaned_df = combined_df

            if not cleaned_df.empty:
                result_data[freq_name] = cleaned_df
                logger.info(f"    {freq_name}数据形状: {cleaned_df.shape}")

        return result_data, {}  # 不借调，返回空日志

    def _process_frequency_data(
        self,
        data_dict: Dict[str, pd.DataFrame],
        original_freq: str,
        target_level: int,
        freq_name: str
    ) -> Tuple[pd.DataFrame, Dict]:
        """处理特定频率的数据（智能缺失值检测）

        Args:
            data_dict: 该频率的所有DataFrame字典
            original_freq: 原始频率代码（'D', 'W', 'M'等）
            target_level: 目标频率等级
            freq_name: 频率名称（用于日志）

        Returns:
            Tuple[pd.DataFrame, Dict]: (对齐后的合并DataFrame, 借调日志字典)
        """
        original_level = self._get_freq_level(original_freq)
        borrowing_log = {}

        # 合并所有同频率的DataFrame
        dfs = list(data_dict.values())
        if not dfs:
            return pd.DataFrame(), borrowing_log

        combined_df = pd.concat(dfs, axis=1)

        # 移除重复列
        combined_df = self.data_cleaner.remove_duplicate_columns(combined_df, f"[{freq_name}] ")
        self.removal_log.extend(self.data_cleaner.get_removed_variables_log())
        self.data_cleaner.clear_log()

        logger.info(f"    合并后形状: {combined_df.shape}")

        # 根据频率关系选择检测时机
        if original_level <= target_level:
            # 原始频率 >= 目标频率（需要降频）：先对齐再检测
            logger.info(f"    先对齐到目标频率...")
            aligned_df, borrowing_log = self._align_by_type(combined_df, freq_name)

            logger.info(f"    再检测连续缺失值（对齐后）...")
            aligned_df = self.data_cleaner.handle_consecutive_nans(
                aligned_df,
                self.consecutive_nan_threshold,
                f"[{freq_name}对齐后] ",
                self.data_start_date,
                self.data_end_date
            )
            self.removal_log.extend(self.data_cleaner.get_removed_variables_log())
            self.data_cleaner.clear_log()

        else:
            # 原始频率 < 目标频率（需要升频）：先检测再对齐
            logger.info(f"    先检测连续缺失值（原始{freq_name}频率）...")
            cleaned_df = self.data_cleaner.handle_consecutive_nans(
                combined_df,
                self.consecutive_nan_threshold,
                f"[{freq_name}原始频率] ",
                self.data_start_date,
                self.data_end_date
            )
            self.removal_log.extend(self.data_cleaner.get_removed_variables_log())
            self.data_cleaner.clear_log()
            logger.info(f"    检测后形状: {cleaned_df.shape}")

            # 对齐到目标频率
            logger.info(f"    对齐到目标频率...")
            aligned_df, borrowing_log = self._align_by_type(cleaned_df, freq_name)

        logger.info(f"    最终形状: {aligned_df.shape}")
        return aligned_df, borrowing_log

    def _align_by_type(self, df: pd.DataFrame, freq_type: str) -> Tuple[pd.DataFrame, Dict]:
        """根据频率类型对齐数据

        Args:
            df: 待对齐的DataFrame
            freq_type: 频率类型（'daily', 'weekly', 'dekad', 'monthly', 'quarterly', 'yearly'）

        Returns:
            Tuple[pd.DataFrame, Dict]: (对齐后的DataFrame, 借调日志字典)
        """
        borrowing_log = {}

        if freq_type == 'daily':
            return self.data_aligner.convert_daily_to_weekly([df]), borrowing_log
        elif freq_type == 'weekly':
            # align_weekly_data 现在返回 Tuple[pd.DataFrame, Dict]，传递用户日期范围
            aligned_df, borrowing_log = self.data_aligner.align_weekly_data(
                [df], self.data_start_date, self.data_end_date
            )
            return aligned_df, borrowing_log
        elif freq_type == 'dekad':
            # convert_dekad_to_weekly 现在返回 Tuple[pd.DataFrame, Dict]，传递用户日期范围
            aligned_df, borrowing_log = self.data_aligner.convert_dekad_to_weekly(
                [df], self.data_start_date, self.data_end_date
            )
            return aligned_df, borrowing_log
        elif freq_type == 'monthly':
            return self.data_aligner.align_monthly_to_last_friday(df), borrowing_log
        elif freq_type == 'quarterly':
            return self.data_aligner.align_quarterly_to_friday(df), borrowing_log
        elif freq_type == 'yearly':
            return self.data_aligner.align_yearly_to_friday(df), borrowing_log
        else:
            raise ValueError(f"不支持的频率类型: {freq_type}")

    def _step6_merge_all_data(
        self,
        aligned_data: Dict[str, pd.DataFrame],
        start_date: str,
        end_date: str
    ) -> pd.DataFrame:
        """步骤6: 合并所有数据形成最终表

        Args:
            aligned_data: 对齐后的数据字典
            start_date: 起始日期
            end_date: 结束日期

        Returns:
            pd.DataFrame: 最终合并的DataFrame
        """
        # 收集所有数据部分（统一处理，无特殊顺序）
        all_parts = []

        for freq_name in ['daily', 'weekly', 'dekad', 'monthly', 'quarterly', 'yearly']:
            if freq_name in aligned_data and not aligned_data[freq_name].empty:
                all_parts.append(aligned_data[freq_name])
                logger.info(f"  添加{freq_name}数据: {aligned_data[freq_name].shape[1]}个变量")

        if not all_parts:
            raise ValueError("没有可合并的数据部分")

        # 合并所有部分
        logger.info("  合并所有数据...")
        combined_df = pd.concat(all_parts, axis=1)
        logger.info(f"  合并后形状: {combined_df.shape}")

        # 移除重复列
        combined_df = self.data_cleaner.remove_duplicate_columns(combined_df, "[最终合并] ")
        self.removal_log.extend(self.data_cleaner.get_removed_variables_log())
        self.data_cleaner.clear_log()

        # 确保目标变量在第一列（如果指定了目标变量）
        if self.target_variable_name and self.target_variable_name in combined_df.columns:
            cols = [self.target_variable_name] + [c for c in combined_df.columns if c != self.target_variable_name]
            combined_df = combined_df[cols]
            logger.info(f"  目标变量 '{self.target_variable_name}' 已移至第一列")
        elif self.target_variable_name:
            logger.warning(f"  目标变量 '{self.target_variable_name}' 未在数据中找到")

        # 检查并处理重复索引
        if combined_df.index.duplicated().any():
            logger.warning(f"  检测到重复索引，正在清理...")
            combined_df = combined_df[~combined_df.index.duplicated(keep='last')]
            logger.info(f"  清理后形状: {combined_df.shape}")

        # 不对齐模式：直接按日期范围过滤，保留原始日期
        if not self.enable_freq_alignment:
            logger.info("  [不对齐模式] 保留原始发布日期...")

            # 按日期范围过滤
            if start_date:
                start_dt = pd.to_datetime(start_date)
                combined_df = combined_df[combined_df.index >= start_dt]
            if end_date:
                end_dt = pd.to_datetime(end_date)
                combined_df = combined_df[combined_df.index <= end_dt]

            # 按日期排序
            combined_df = combined_df.sort_index()
            logger.info(f"  过滤后日期范围: {combined_df.index.min()} 至 {combined_df.index.max()}")

        else:
            # 对齐模式：创建完整日期范围并对齐
            logger.info("  创建完整日期范围...")
            all_indices = [part.index for part in all_parts if hasattr(part, 'index')]
            full_date_range = self.data_aligner.create_full_date_range(
                all_indices, start_date, end_date
            )
            logger.info(f"  日期范围: {full_date_range[0]} 至 {full_date_range[-1]}, 共{len(full_date_range)}个周五")

            # 重新索引到完整日期范围
            combined_df = combined_df.reindex(full_date_range)

            # 恢复频率信息
            if hasattr(full_date_range, 'freq') and full_date_range.freq is not None:
                combined_df.index.freq = full_date_range.freq

        # 最终清理：移除全NaN列（不替换0值，0值处理由用户在变量处理中设置）
        logger.info("  最终清理...")
        final_df, clean_log = clean_dataframe(
            combined_df,
            remove_zeros=False,
            remove_all_nan_cols=True,
            remove_all_nan_rows=False,
            log_prefix="[最终清理] "
        )
        self.removal_log.extend(clean_log)

        if final_df.empty:
            raise ValueError("最终数据为空")

        logger.info(f"  最终形状: {final_df.shape}")
        return final_df

    def _step7_generate_output(
        self,
        final_df: pd.DataFrame,
        borrowing_log: Optional[Dict] = None
    ) -> Tuple[pd.DataFrame, Dict[str, str], Dict, List[Dict]]:
        """步骤7: 生成输出

        Args:
            final_df: 最终处理后的DataFrame
            borrowing_log: 借调日志字典（可选）

        Returns:
            Tuple[pd.DataFrame, Dict, Dict, List]:
                - 处理后的DataFrame
                - 变量-行业映射
                - 转换日志（包含借调日志）
                - 移除变量日志
        """
        # 构建最终变量映射（只使用指标体系映射）
        final_var_mapping = {}
        for col in final_df.columns:
            col_norm = normalize_text(col)
            if col_norm in self.var_industry_map:
                final_var_mapping[col_norm] = self.var_industry_map[col_norm]
            else:
                final_var_mapping[col_norm] = "Unknown"
                logger.warning(f"  变量 '{col}' 未在指标体系中定义，标记为Unknown")

        # 生成转换日志（平稳性检验已移至变量处理功能区）
        transform_log = {
            'target_freq': self.target_freq,
            'data_start_date': self.data_start_date,
            'data_end_date': self.data_end_date,
            'consecutive_nan_threshold': self.consecutive_nan_threshold,
            'final_shape': final_df.shape,
            'processing_time': datetime.now().isoformat(),
            'borrowing_log': borrowing_log or {}
        }

        # 合并去趋势日志（如果有）
        if 'detrend' in self.transform_log:
            transform_log['detrend'] = self.transform_log['detrend']
        else:
            transform_log['detrend'] = {'enabled': False}

        # 记录借调统计
        if borrowing_log:
            total_borrowing_vars = len(borrowing_log)
            total_borrowing_count = sum(len(logs) for logs in borrowing_log.values())
            logger.info(f"  借调日志: {total_borrowing_vars}个变量，共{total_borrowing_count}次借调")

        logger.info(f"  变量映射: {len(final_var_mapping)}个变量")
        logger.info(f"  未定义变量: {sum(1 for v in final_var_mapping.values() if v == 'Unknown')}个")
        logger.info(f"  移除日志: {len(self.removal_log)}条记录")

        return final_df, final_var_mapping, transform_log, self.removal_log


    def _get_freq_level(self, freq: str) -> int:
        """获取频率等级

        Args:
            freq: 频率字符串（如'W-FRI', 'D', 'M'）

        Returns:
            int: 频率等级（1-5）
        """
        # 提取频率代码
        if '-' in freq:
            freq_code = freq.split('-')[0]
        else:
            freq_code = freq[0] if freq else 'W'

        return self.FREQ_ORDER.get(freq_code, 2)  # 默认为周度

    def _run_stationarity_check(self, df: pd.DataFrame, alpha: float = 0.05) -> Dict[str, str]:
        """执行平稳性检验（ADF检验）

        在最终数据上一次性执行，避免UI层重复调用

        Args:
            df: 待检验的DataFrame
            alpha: 显著性水平，默认0.05

        Returns:
            Dict[str, str]: {变量名: "是"/"否"/"数据不足"}
        """
        from statsmodels.tsa.stattools import adfuller

        results = {}
        for col in df.columns:
            series = df[col].dropna()

            # 检查样本量（ADF检验需要足够的样本）
            if len(series) < 12:
                results[col] = "数据不足"
                continue

            try:
                adf_result = adfuller(series)
                p_value = adf_result[1]
                results[col] = "是" if p_value < alpha else "否"
            except Exception as e:
                logger.warning(f"  变量 '{col}' ADF检验失败: {e}")
                results[col] = "检验失败"

        # 统计
        stationary_count = sum(1 for v in results.values() if v == "是")
        logger.info(f"  平稳性检验完成: {stationary_count}/{len(results)} 个变量平稳")

        return results
