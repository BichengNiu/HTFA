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
        st_instance.markdown("### 数据文件上传")

        # 检查是否已有上传的文件
        existing_data = self.state.get('train_uploaded_data_file', None)
        existing_map = self.state.get('train_uploaded_industry_map_file', None)

        col_upload1, col_upload2 = st_instance.columns(2)

        # 预处理数据文件上传
        with col_upload1:
            st_instance.markdown("**预处理数据文件 (.csv)**")
            uploaded_data_file = st_instance.file_uploader(
                "选择预处理数据文件",
                type=['csv'],
                key="train_data_upload",
                help="上传数据准备模块导出的预处理数据CSV文件（包含日期索引和所有变量列）"
            )

            if uploaded_data_file:
                self.state.set("train_uploaded_data_file", uploaded_data_file)
                st_instance.success(f"已上传: {uploaded_data_file.name}")
            else:
                if existing_data is not None and hasattr(existing_data, 'name'):
                    st_instance.success(f"当前文件: {existing_data.name}")

        # 行业映射文件上传
        with col_upload2:
            st_instance.markdown("**行业映射文件 (.csv)**")
            uploaded_industry_map_file = st_instance.file_uploader(
                "选择行业映射文件",
                type=['csv'],
                key="train_industry_map_upload",
                help="上传数据准备模块导出的行业映射CSV文件（包含Indicator和Industry两列）"
            )

            if uploaded_industry_map_file:
                self.state.set("train_uploaded_industry_map_file", uploaded_industry_map_file)
                st_instance.success(f"已上传: {uploaded_industry_map_file.name}")
            else:
                if existing_map is not None and hasattr(existing_map, 'name'):
                    st_instance.success(f"当前文件: {existing_map.name}")

        st_instance.markdown("---")

        # 加载文件
        data_file = self.state.get('train_uploaded_data_file', None)
        industry_map_file = self.state.get('train_uploaded_industry_map_file', None)

        # 加载预处理数据
        input_df = self._load_data_file(data_file, st_instance)

        # 加载行业映射
        var_industry_map, dfm_default_map = self._load_mapping_file(industry_map_file, st_instance)

        # 数据验证
        if not self._validate_files(input_df, var_industry_map, data_file, industry_map_file, st_instance):
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

    def _load_data_file(self, data_file, st_instance) -> Optional[pd.DataFrame]:
        """加载预处理数据文件（带缓存）"""
        if data_file is None:
            return None

        current_file_id = self._get_file_id(data_file)
        cached_file_id = self.state.get('cached_data_file_id', None)
        cached_df = self.state.get('dfm_prepared_data_df', None)

        # 检查缓存
        if cached_df is not None and current_file_id == cached_file_id:
            print(f"[模型训练] 使用缓存的预处理数据: {cached_df.shape}")
            return cached_df

        # 重新加载
        try:
            data_file.seek(0)
            input_df = pd.read_csv(data_file, index_col=0, parse_dates=True)
            self.state.set('dfm_prepared_data_df', input_df)
            self.state.set('cached_data_file_id', current_file_id)
            print(f"[模型训练] 重新加载预处理数据: {input_df.shape}")
            return input_df
        except Exception as e:
            st_instance.error(f"加载预处理数据失败: {e}")
            return None

    def _load_mapping_file(self, industry_map_file, st_instance) -> Tuple[Dict[str, str], Dict[str, str]]:
        """加载行业映射文件（带缓存）"""
        if industry_map_file is None:
            return {}, {}

        current_file_id = self._get_file_id(industry_map_file)
        cached_file_id = self.state.get('cached_map_file_id', None)
        cached_industry_map = self.state.get('dfm_industry_map_obj', None)
        cached_single_stage_map = self.state.get('dfm_default_single_stage_map', None)
        cached_first_stage_pred_map = self.state.get('dfm_first_stage_pred_map', None)
        cached_first_stage_target_map = self.state.get('dfm_first_stage_target_map', None)
        cached_second_stage_target_map = self.state.get('dfm_second_stage_target_map', None)

        # 检查缓存
        if cached_industry_map is not None and current_file_id == cached_file_id:
            print(f"[模型训练] 使用缓存的行业映射: {len(cached_industry_map)} 个变量")
            if cached_single_stage_map is not None:
                print(f"[模型训练] 使用缓存的一次估计默认变量: {len(cached_single_stage_map)} 个")
            if cached_first_stage_pred_map is not None:
                print(f"[模型训练] 使用缓存的一阶段预测默认变量: {len(cached_first_stage_pred_map)} 个")
            if cached_first_stage_target_map is not None:
                print(f"[模型训练] 使用缓存的一阶段目标默认变量: {len(cached_first_stage_target_map)} 个")
            if cached_second_stage_target_map is not None:
                print(f"[模型训练] 使用缓存的二阶段目标默认变量: {len(cached_second_stage_target_map)} 个")
            return cached_industry_map, cached_single_stage_map or {}

        # 重新加载
        try:
            industry_map_file.seek(0)
            industry_map_df = pd.read_csv(industry_map_file)

            # 打印CSV文件基本信息
            print(f"[模型训练] 行业映射CSV文件形状: {industry_map_df.shape}")
            print(f"[模型训练] 行业映射CSV列名: {list(industry_map_df.columns)}")

            if 'Indicator' not in industry_map_df.columns or 'Industry' not in industry_map_df.columns:
                st_instance.error("映射文件格式错误：必须包含 'Indicator' 和 'Industry' 列")
                print(f"[ERROR] 缺少必需列。当前列名: {list(industry_map_df.columns)}")
                return {}, {}

            # 统计原始数据行数
            total_rows = len(industry_map_df)
            print(f"[模型训练] 行业映射CSV总行数: {total_rows}")

            # 加载行业映射（记录过滤信息）
            filtered_count = 0
            var_industry_map = {}

            for idx, (k, v) in enumerate(zip(industry_map_df['Indicator'], industry_map_df['Industry'])):
                if pd.notna(k) and pd.notna(v) and str(k).strip() and str(v).strip():
                    normalized_key = unicodedata.normalize('NFKC', str(k)).strip().lower()
                    var_industry_map[normalized_key] = str(v).strip()
                else:
                    filtered_count += 1
                    if filtered_count <= 3:  # 只打印前3个被过滤的行
                        print(f"[模型训练] 过滤第{idx+1}行: Indicator={k}, Industry={v}")

            if filtered_count > 0:
                print(f"[模型训练] 共过滤掉 {filtered_count} 行（Indicator或Industry为空）")

            self.state.set('dfm_industry_map_obj', var_industry_map)
            self.state.set('cached_map_file_id', current_file_id)
            print(f"[模型训练] 重新加载行业映射: {len(var_industry_map)} 个变量")

            if len(var_industry_map) == 0:
                st_instance.error("行业映射文件解析后为空：所有行的Indicator或Industry列都为空")
                print(f"[ERROR] 行业映射为空字典，请检查CSV文件数据")
                return {}, {}

            # 加载DFM默认变量（支持四列：一次估计、一阶段预测、一阶段目标、二阶段目标）
            dfm_default_single_stage = {}
            dfm_first_stage_pred = {}
            dfm_first_stage_target = {}
            dfm_second_stage_target = {}

            # 读取一次估计列
            if '一次估计' in industry_map_df.columns:
                dfm_default_single_stage = {
                    unicodedata.normalize('NFKC', str(k)).strip().lower(): str(v).strip()
                    for k, v in zip(industry_map_df['Indicator'], industry_map_df['一次估计'])
                    if pd.notna(k) and pd.notna(v) and str(v).strip() == '是'
                }
                print(f"[模型训练] 从CSV文件加载了一次估计默认变量: {len(dfm_default_single_stage)} 个")

            # 读取一阶段预测列
            if '一阶段预测' in industry_map_df.columns:
                dfm_first_stage_pred = {
                    unicodedata.normalize('NFKC', str(k)).strip().lower(): str(v).strip()
                    for k, v in zip(industry_map_df['Indicator'], industry_map_df['一阶段预测'])
                    if pd.notna(k) and pd.notna(v) and str(v).strip() == '是'
                }
                print(f"[模型训练] 从CSV文件加载了一阶段预测默认变量: {len(dfm_first_stage_pred)} 个")

            # 读取一阶段目标列
            if '一阶段目标' in industry_map_df.columns:
                dfm_first_stage_target = {
                    unicodedata.normalize('NFKC', str(k)).strip().lower(): str(v).strip()
                    for k, v in zip(industry_map_df['Indicator'], industry_map_df['一阶段目标'])
                    if pd.notna(k) and pd.notna(v) and str(v).strip() == '是'
                }
                print(f"[模型训练] 从CSV文件加载了一阶段目标默认变量: {len(dfm_first_stage_target)} 个")

            # 读取二阶段目标列
            if '二阶段目标' in industry_map_df.columns:
                dfm_second_stage_target = {
                    unicodedata.normalize('NFKC', str(k)).strip().lower(): str(v).strip()
                    for k, v in zip(industry_map_df['Indicator'], industry_map_df['二阶段目标'])
                    if pd.notna(k) and pd.notna(v) and str(v).strip() == '是'
                }
                print(f"[模型训练] 从CSV文件加载了二阶段目标默认变量: {len(dfm_second_stage_target)} 个")

            # 存储四个映射
            self.state.set('dfm_default_single_stage_map', dfm_default_single_stage)
            self.state.set('dfm_first_stage_pred_map', dfm_first_stage_pred)
            self.state.set('dfm_first_stage_target_map', dfm_first_stage_target)
            self.state.set('dfm_second_stage_target_map', dfm_second_stage_target)

            # 返回single_stage作为默认（向后兼容）
            return var_industry_map, dfm_default_single_stage

        except Exception as e:
            st_instance.error(f"加载映射文件失败: {e}")
            import traceback
            st_instance.code(traceback.format_exc(), language="python")
            return {}, {}

    def _validate_files(self, input_df, var_industry_map, data_file, industry_map_file, st_instance) -> bool:
        """验证文件是否都已上传并成功解析"""
        if input_df is None or not var_industry_map:
            missing_details = []

            if input_df is None:
                if data_file is None:
                    missing_details.append("预处理数据文件：未上传")
                else:
                    missing_details.append("预处理数据文件：上传成功但解析失败，请检查CSV格式是否正确")

            if not var_industry_map:
                if industry_map_file is None:
                    missing_details.append("行业映射文件：未上传")
                else:
                    missing_details.append("行业映射文件：上传成功但解析失败，可能原因：\n  - 缺少必需的列（需要包含'Indicator'和'Industry'列）\n  - 数据行全部为空或格式不正确")

            st_instance.error(f"数据验证失败：\n\n" + "\n\n".join([f"{i+1}. {detail}" for i, detail in enumerate(missing_details)]))

            # 添加排查建议
            with st_instance.expander("查看排查建议"):
                st_instance.markdown("""
                **预处理数据文件要求：**
                - 必须是CSV格式
                - 第一列为日期索引
                - 包含所有变量列

                **行业映射文件要求：**
                - 必须是CSV格式
                - 必须包含'Indicator'和'Industry'两列
                - 每行数据不能为空
                - 可选包含'DFM_Default'列（标记默认变量）

                **常见问题：**
                - 文件编码问题：确保使用UTF-8编码
                - 列名拼写错误：检查列名是否完全匹配
                - 数据为空：确保文件中有有效数据行
                """)

            return False

        return True
