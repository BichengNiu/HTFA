# -*- coding: utf-8 -*-
"""
文件上传组件
处理预处理数据和行业映射文件的上传与加载
"""

import pandas as pd
import unicodedata
from typing import Tuple, Optional, Dict, Any


class FileUploaderComponent:
    """文件上传和加载组件"""

    def __init__(self, state_manager):
        """
        初始化文件上传组件

        Args:
            state_manager: 状态管理器，需要有get和set方法
        """
        self.state = state_manager

    def render(self, st_instance) -> Tuple[Optional[pd.DataFrame], Dict[str, str], Dict[str, str]]:
        """
        渲染文件上传UI并返回加载的数据

        Args:
            st_instance: Streamlit实例

        Returns:
            Tuple包含:
            - input_df: 预处理数据DataFrame
            - var_industry_map: 变量到行业的映射字典
            - dfm_default_map: DFM默认变量映射字典
        """
        st_instance.markdown("#### 数据文件上传")

        # 检查是否已有上传的文件
        existing_excel = self.state.get('train_uploaded_excel_file', None)

        # Excel文件上传（推荐方式）
        uploaded_excel_file = st_instance.file_uploader(
            "上传数据准备模块导出的Excel文件",
            type=['xlsx', 'xls'],
            key="train_excel_upload",
            help="包含'数据'和'映射'两个sheet的Excel文件"
        )

        if uploaded_excel_file:
            self.state.set("train_uploaded_excel_file", uploaded_excel_file)

        # 加载Excel文件
        excel_file = self.state.get('train_uploaded_excel_file', None)

        if excel_file is not None:
            # 从Excel加载数据和映射
            input_df, var_industry_map, dfm_default_map = self._load_excel_file(excel_file, st_instance)

            # 显示加载成功的信息
            if input_df is not None and var_industry_map:
                st_instance.success(f"成功加载Excel文件：数据{input_df.shape[0]}行×{input_df.shape[1]}列，映射{len(var_industry_map)}个变量")
        else:
            st_instance.warning("请上传数据准备模块导出的Excel文件")
            return None, {}, {}

        st_instance.markdown("---")

        # 数据验证
        if not self._validate_files(input_df, var_industry_map, excel_file, None, st_instance):
            return None, {}, {}

        return input_df, var_industry_map, dfm_default_map

    def _get_file_id(self, file_obj) -> Optional[str]:
        """生成文件标识用于缓存检测"""
        if file_obj is None:
            return None
        file_obj.seek(0, 2)
        size = file_obj.tell()
        file_obj.seek(0)
        name = getattr(file_obj, 'name', 'unknown')
        return f"{name}_{size}"

    def _load_excel_file(self, excel_file, st_instance) -> Tuple[Optional[pd.DataFrame], Dict[str, str], Dict[str, str]]:
        """
        从Excel文件加载数据和映射（带缓存）

        Excel文件应包含两个sheet：
        - '数据': 预处理数据（第一列为日期索引）
        - '映射': 行业映射表（包含Indicator和Industry列）

        Args:
            excel_file: Excel文件对象
            st_instance: Streamlit实例

        Returns:
            Tuple包含:
            - input_df: 预处理数据DataFrame
            - var_industry_map: 变量到行业的映射字典
            - dfm_default_map: DFM默认变量映射字典
        """
        if excel_file is None:
            return None, {}, {}

        current_file_id = self._get_file_id(excel_file)
        cached_file_id = self.state.get('cached_excel_file_id', None)
        cached_df = self.state.get('dfm_prepared_data_df', None)
        cached_industry_map = self.state.get('dfm_industry_map_obj', None)
        cached_single_stage_map = self.state.get('dfm_default_single_stage_map', None)

        # 检查缓存
        if (cached_df is not None and cached_industry_map is not None
            and current_file_id == cached_file_id):
            print(f"[模型训练] 使用缓存的Excel数据: 数据={cached_df.shape}, 映射={len(cached_industry_map)}个变量")
            return cached_df, cached_industry_map, cached_single_stage_map or {}

        # 重新加载Excel
        try:
            excel_file.seek(0)
            print(f"[模型训练] 开始加载Excel文件: {excel_file.name}")

            # 读取'数据'sheet
            print("[模型训练] 读取'数据'sheet...")
            input_df = pd.read_excel(excel_file, sheet_name='数据', index_col=0, parse_dates=True)
            print(f"[模型训练] 数据shape: {input_df.shape}, 列数: {len(input_df.columns)}")
            print(f"[模型训练] 时间范围: {input_df.index.min()} 至 {input_df.index.max()}")

            # 读取'映射'sheet
            excel_file.seek(0)
            print("[模型训练] 读取'映射'sheet...")
            industry_map_df = pd.read_excel(excel_file, sheet_name='映射')
            print(f"[模型训练] 映射表shape: {industry_map_df.shape}, 列名: {list(industry_map_df.columns)}")

            # 验证必需列
            if 'Indicator' not in industry_map_df.columns or 'Industry' not in industry_map_df.columns:
                st_instance.error("映射表格式错误：必须包含'Indicator'和'Industry'列")
                return None, {}, {}

            # 解析行业映射
            var_industry_map = {}
            filtered_count = 0

            for idx, (k, v) in enumerate(zip(industry_map_df['Indicator'], industry_map_df['Industry'])):
                if pd.notna(k) and pd.notna(v) and str(k).strip() and str(v).strip():
                    normalized_key = unicodedata.normalize('NFKC', str(k)).strip().lower()
                    var_industry_map[normalized_key] = str(v).strip()
                else:
                    filtered_count += 1

            if filtered_count > 0:
                print(f"[模型训练] 过滤掉{filtered_count}行空数据")

            print(f"[模型训练] 行业映射加载完成: {len(var_industry_map)}个变量")

            if len(var_industry_map) == 0:
                st_instance.error("行业映射为空，请检查Excel文件的'映射'sheet")
                return None, {}, {}

            # 解析默认变量配置（支持四列）
            dfm_default_single_stage = {}
            dfm_first_stage_pred = {}
            dfm_first_stage_target = {}
            dfm_second_stage_target = {}

            if '一次估计' in industry_map_df.columns:
                dfm_default_single_stage = {
                    unicodedata.normalize('NFKC', str(k)).strip().lower(): str(v).strip()
                    for k, v in zip(industry_map_df['Indicator'], industry_map_df['一次估计'])
                    if pd.notna(k) and pd.notna(v) and str(v).strip() == '是'
                }
                print(f"[模型训练] 一次估计默认变量: {len(dfm_default_single_stage)}个")

            if '一阶段预测' in industry_map_df.columns:
                dfm_first_stage_pred = {
                    unicodedata.normalize('NFKC', str(k)).strip().lower(): str(v).strip()
                    for k, v in zip(industry_map_df['Indicator'], industry_map_df['一阶段预测'])
                    if pd.notna(k) and pd.notna(v) and str(v).strip() == '是'
                }
                print(f"[模型训练] 一阶段预测默认变量: {len(dfm_first_stage_pred)}个")

            if '一阶段目标' in industry_map_df.columns:
                dfm_first_stage_target = {
                    unicodedata.normalize('NFKC', str(k)).strip().lower(): str(v).strip()
                    for k, v in zip(industry_map_df['Indicator'], industry_map_df['一阶段目标'])
                    if pd.notna(k) and pd.notna(v) and str(v).strip() == '是'
                }
                print(f"[模型训练] 一阶段目标默认变量: {len(dfm_first_stage_target)}个")

            if '二阶段目标' in industry_map_df.columns:
                dfm_second_stage_target = {
                    unicodedata.normalize('NFKC', str(k)).strip().lower(): str(v).strip()
                    for k, v in zip(industry_map_df['Indicator'], industry_map_df['二阶段目标'])
                    if pd.notna(k) and pd.notna(v) and str(v).strip() == '是'
                }
                print(f"[模型训练] 二阶段目标默认变量: {len(dfm_second_stage_target)}个")

            # 缓存数据
            self.state.set('dfm_prepared_data_df', input_df)
            self.state.set('dfm_industry_map_obj', var_industry_map)
            self.state.set('dfm_default_single_stage_map', dfm_default_single_stage)
            self.state.set('dfm_first_stage_pred_map', dfm_first_stage_pred)
            self.state.set('dfm_first_stage_target_map', dfm_first_stage_target)
            self.state.set('dfm_second_stage_target_map', dfm_second_stage_target)
            self.state.set('cached_excel_file_id', current_file_id)

            print("[模型训练] Excel文件加载成功并已缓存")

            return input_df, var_industry_map, dfm_default_single_stage

        except ValueError as e:
            if "Worksheet named" in str(e):
                st_instance.error(f"Excel文件格式错误：缺少必需的sheet。需要包含'数据'和'映射'两个sheet。")
            else:
                st_instance.error(f"加载Excel文件失败: {e}")
            import traceback
            st_instance.code(traceback.format_exc(), language="python")
            return None, {}, {}
        except Exception as e:
            st_instance.error(f"加载Excel文件失败: {e}")
            import traceback
            st_instance.code(traceback.format_exc(), language="python")
            return None, {}, {}

    def _validate_files(self, input_df, var_industry_map, excel_file, _, st_instance) -> bool:
        """验证Excel文件是否已上传并成功解析"""
        if input_df is None or not var_industry_map:
            missing_details = []

            if input_df is None:
                if excel_file is None:
                    missing_details.append("Excel文件：未上传")
                else:
                    missing_details.append("Excel文件：上传成功但解析失败，请检查文件格式")

            if not var_industry_map:
                missing_details.append("行业映射：解析失败，请检查Excel文件的'映射'sheet是否包含'Indicator'和'Industry'列")

            st_instance.error(f"数据验证失败：\n\n" + "\n\n".join([f"{i+1}. {detail}" for i, detail in enumerate(missing_details)]))

            # 添加排查建议
            with st_instance.expander("查看排查建议"):
                st_instance.markdown("""
                **Excel文件要求：**
                - 包含两个sheet：'数据'和'映射'
                - '数据'sheet：第一列为日期索引，其余列为变量
                - '映射'sheet：包含'Indicator'和'Industry'两列

                **常见问题：**
                - Sheet名称错误：确保sheet名称为'数据'和'映射'
                - 列名拼写错误：检查列名是否完全匹配
                - 数据为空：确保文件中有有效数据行
                """)

            return False

        return True
