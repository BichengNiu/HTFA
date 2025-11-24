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
from dashboard.models.DFM.prep.utils.text_utils import normalize_text

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
        data_end_date: Optional[str] = None
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
        """
        if not target_freq.upper().endswith('-FRI'):
            raise ValueError(f"当前仅支持周五对齐 (W-FRI)，提供的频率 '{target_freq}' 无效")

        self.excel_path = excel_path
        self.target_variable_name = target_variable_name
        self.var_industry_map = var_industry_map
        self.var_frequency_map = var_frequency_map
        self.target_freq = target_freq
        self.consecutive_nan_threshold = consecutive_nan_threshold
        self.data_start_date = standardize_date(data_start_date)
        self.data_end_date = standardize_date(data_end_date)

        # 初始化组件
        self.data_loader = DataLoader(
            reference_industry_map=var_industry_map,
            reference_frequency_map=var_frequency_map
        )
        self.data_aligner = DataAligner(target_freq)
        self.data_cleaner = DataCleaner()

        # 日志记录
        self.removal_log = []
        self.transform_log = {}

        logger.info(f"[Processor] 初始化完成: 目标变量={target_variable_name}")

    def execute(self) -> Tuple[pd.DataFrame, Dict[str, str], Dict, List[Dict]]:
        """执行完整的7步数据准备流程

        Returns:
            Tuple[pd.DataFrame, Dict, Dict, List]:
                - 处理后的DataFrame
                - 变量-行业映射
                - 转换日志
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
            aligned_data = self._step5_smart_missing_detection_and_align(data_by_freq)

            # 步骤6: 合并所有数据形成最终表
            logger.info("\n步骤6/7: 合并数据...")
            final_df = self._step6_merge_all_data(aligned_data, effective_start, effective_end)

            # 步骤7: 生成输出
            logger.info("\n步骤7/7: 生成输出...")
            result = self._step7_generate_output(final_df)

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
            'monthly': {}
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
                    else:
                        logger.warning(f"    变量 '{var_name}' 频率 '{freq}' 无法识别，跳过")

            except Exception as e:
                logger.warning(f"    加载失败: {e}")
                continue

        # 转换为DataFrame格式
        for freq_type in ['daily', 'weekly', 'dekad', 'monthly']:
            if data_by_freq[freq_type]:
                series_list = []
                for var_name, series in data_by_freq[freq_type].items():
                    if series.index.duplicated().any():
                        series = series[~series.index.duplicated(keep='last')]
                    series_list.append(series)
                if series_list:
                    combined_df = pd.concat(series_list, axis=1)
                    data_by_freq[freq_type] = {'combined': combined_df}

        return data_by_freq

    def _step5_smart_missing_detection_and_align(
        self,
        data_by_freq: Dict[str, Dict]
    ) -> Dict[str, pd.DataFrame]:
        """步骤5: 智能缺失值检测与频率对齐（统一处理所有变量）

        根据频率关系选择检测时机：
        - 目标频率 >= 原始频率（如日度→周度）：先检测再对齐
        - 目标频率 < 原始频率（如月度→周度）：先对齐再检测

        Args:
            data_by_freq: 按频率分类的数据

        Returns:
            Dict: {
                'daily': pd.DataFrame,
                'weekly': pd.DataFrame,
                'dekad': pd.DataFrame,
                'monthly': pd.DataFrame
            }
        """
        aligned_data = {}
        target_level = self._get_freq_level(self.target_freq)

        # 处理日度数据
        if data_by_freq['daily']:
            logger.info("  处理日度数据...")
            aligned_data['daily'] = self._process_frequency_data(
                data_by_freq['daily'], 'D', target_level, 'daily'
            )

        # 处理周度数据
        if data_by_freq['weekly']:
            logger.info("  处理周度数据...")
            aligned_data['weekly'] = self._process_frequency_data(
                data_by_freq['weekly'], 'W', target_level, 'weekly'
            )

        # 处理旬度数据
        if data_by_freq['dekad']:
            logger.info("  处理旬度数据...")
            aligned_data['dekad'] = self._process_frequency_data(
                data_by_freq['dekad'], 'M', target_level, 'dekad'
            )

        # 处理月度数据
        if data_by_freq['monthly']:
            logger.info("  处理月度数据...")
            aligned_data['monthly'] = self._process_frequency_data(
                data_by_freq['monthly'], 'M', target_level, 'monthly'
            )

        return aligned_data

    def _process_frequency_data(
        self,
        data_dict: Dict[str, pd.DataFrame],
        original_freq: str,
        target_level: int,
        freq_name: str
    ) -> pd.DataFrame:
        """处理特定频率的数据（智能缺失值检测）

        Args:
            data_dict: 该频率的所有DataFrame字典
            original_freq: 原始频率代码（'D', 'W', 'M'等）
            target_level: 目标频率等级
            freq_name: 频率名称（用于日志）

        Returns:
            pd.DataFrame: 对齐后的合并DataFrame
        """
        original_level = self._get_freq_level(original_freq)

        # 合并所有同频率的DataFrame
        dfs = list(data_dict.values())
        if not dfs:
            return pd.DataFrame()

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
            aligned_df = self._align_by_type(combined_df, freq_name)

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
            aligned_df = self._align_by_type(cleaned_df, freq_name)

        logger.info(f"    最终形状: {aligned_df.shape}")
        return aligned_df

    def _align_by_type(self, df: pd.DataFrame, freq_type: str) -> pd.DataFrame:
        """根据频率类型对齐数据

        Args:
            df: 待对齐的DataFrame
            freq_type: 频率类型（'daily', 'weekly', 'dekad', 'monthly'）

        Returns:
            pd.DataFrame: 对齐后的DataFrame
        """
        if freq_type == 'daily':
            return self.data_aligner.convert_daily_to_weekly([df])
        elif freq_type == 'weekly':
            return self.data_aligner.align_weekly_data([df])
        elif freq_type == 'dekad':
            return self.data_aligner.convert_dekad_to_weekly([df])
        elif freq_type == 'monthly':
            return self.data_aligner.align_monthly_to_last_friday(df)
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

        for freq_name in ['daily', 'weekly', 'dekad', 'monthly']:
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

        # 创建完整日期范围并对齐
        logger.info("  创建完整日期范围...")
        all_indices = [part.index for part in all_parts if hasattr(part, 'index')]
        full_date_range = self.data_aligner.create_full_date_range(
            all_indices, start_date, end_date
        )
        logger.info(f"  日期范围: {full_date_range[0]} 至 {full_date_range[-1]}, 共{len(full_date_range)}个周五")

        # 检查并处理重复索引
        if combined_df.index.duplicated().any():
            logger.warning(f"  检测到重复索引，正在清理...")
            combined_df = combined_df[~combined_df.index.duplicated(keep='last')]
            logger.info(f"  清理后形状: {combined_df.shape}")

        # 重新索引到完整日期范围
        combined_df = combined_df.reindex(full_date_range)

        # 恢复频率信息
        if hasattr(full_date_range, 'freq') and full_date_range.freq is not None:
            combined_df.index.freq = full_date_range.freq

        # 最终清理：移除全NaN列
        logger.info("  最终清理...")
        final_df, clean_log = clean_dataframe(
            combined_df,
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
        final_df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, Dict[str, str], Dict, List[Dict]]:
        """步骤7: 生成输出

        Args:
            final_df: 最终处理后的DataFrame

        Returns:
            Tuple[pd.DataFrame, Dict, Dict, List]:
                - 处理后的DataFrame
                - 变量-行业映射
                - 转换日志
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

        # 生成转换日志
        transform_log = {
            'target_freq': self.target_freq,
            'data_start_date': self.data_start_date,
            'data_end_date': self.data_end_date,
            'consecutive_nan_threshold': self.consecutive_nan_threshold,
            'final_shape': final_df.shape,
            'processing_time': datetime.now().isoformat()
        }

        # 合并去趋势日志（如果有）
        if 'detrend' in self.transform_log:
            transform_log['detrend'] = self.transform_log['detrend']
        else:
            transform_log['detrend'] = {'enabled': False}

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
