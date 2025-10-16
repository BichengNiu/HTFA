#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
工业增加值行业归并脚本
将2012年、2018年、2020年的分行业工业增加值数据按照统一的行业分类进行归并
生成包含归并数据和映射关系的Excel文件

使用方法：
1. 确保数据文件 '分行业工业增加值.xlsx' 在同一目录下
2. 运行脚本: python 工业增加值行业归并脚本.py
3. 生成结果文件: '工业增加值行业归并结果.xlsx'

作者：Augment Agent
日期：2025-07-05
"""

import pandas as pd
from typing import Dict, List, Tuple, Optional
import os


def load_industry_data() -> Tuple[Dict[str, pd.DataFrame], List[str]]:
    """
    读取所有年份的行业数据和标准分类
    
    Returns:
        Tuple[Dict[str, pd.DataFrame], List[str]]: (各年份数据字典, 标准行业列表)
    """
    print("正在读取工业增加值数据...")
    
    # 读取Excel文件
    excel_path = '分行业工业增加值.xlsx'
    if not os.path.exists(excel_path):
        raise FileNotFoundError(f"数据文件不存在: {excel_path}")
    
    xl = pd.ExcelFile(excel_path)
    
    # 读取标准行业分类（包括第一行）
    df_standard_raw = pd.read_excel(xl, sheet_name='行业', header=None)
    standard_industries = df_standard_raw.iloc[:, 0].dropna().tolist()
    print(f"读取到 {len(standard_industries)} 个标准行业分类")
    
    # 读取各年份数据
    years_to_process = ['2012', '2018', '2020']
    year_data = {}
    
    for year in years_to_process:
        try:
            df_year = pd.read_excel(xl, sheet_name=year)
            # 确保工业增加值列为数值类型
            df_year['工业增加值'] = pd.to_numeric(df_year['工业增加值'], errors='coerce')
            year_data[year] = df_year
            print(f"读取 {year} 年数据: {len(df_year)} 个行业")
        except Exception as e:
            print(f"读取 {year} 年数据失败: {e}")
            continue
    
    return year_data, standard_industries


def create_industry_mapping() -> Dict[str, List[str]]:
    """
    创建行业映射规则
    
    Returns:
        Dict[str, List[str]]: 标准行业名到关键词列表的映射
    """
    print("创建行业映射规则...")
    
    keyword_mapping = {
        '农副食品加工业': ['谷物', '饲料', '植物油', '糖', '屠宰', '肉类', '水产', '蔬菜', '水果', '坚果', '乳制品'],
        '食品制造业': ['方便食品', '调味品', '发酵制品', '其他食品'],
        '酒、饮料和精制茶制造业': ['酒精和酒', '饮料', '精制茶'],
        '烟草制品业': ['烟草'],
        '纺织业': ['棉', '化纤纺织', '毛纺织', '麻', '丝绢纺织', '针织', '钩针编织', '纺织制成品'],
        '纺织服装、服饰业': ['纺织服装服饰'],
        '皮革、毛皮、羽毛及其制品和制鞋业': ['皮革', '毛皮', '羽毛', '鞋'],
        '木材加工和木、竹、藤、棕、草制品业': ['木材加工', '木、竹、藤、棕、草'],
        '家具制造业': ['家具'],
        '造纸和纸制品业': ['造纸', '纸制品'],
        '印刷和记录媒介复制业': ['印刷', '记录媒介复制'],
        '文教、工美、体育和娱乐用品制造业': ['工艺美术', '文教', '娱乐用品'],
        '石油、煤炭及其他燃料加工业': ['精炼石油', '核燃料', '炼焦', '煤炭加工'],
        '化学原料和化学制品制造业': ['基础化学原料', '肥料', '农药', '涂料', '油墨', '颜料', '合成材料', '专用化学', '炸药', '火工', '焰火', '日用化学'],
        '医药制造业': ['医药制品'],
        '化学纤维制造业': ['化学纤维制品'],
        '橡胶和塑料制品业': ['橡胶制品', '塑料制品'],
        '非金属矿物制品业': ['水泥', '石灰', '石膏', '砖瓦', '石材', '建筑材料', '玻璃', '陶瓷', '耐火材料', '石墨', '非金属矿物制品'],
        '黑色金属冶炼和压延加工业': ['钢', '铁及铁合金', '钢压延', '钢、铁及其铸件'],
        '有色金属矿采选业': ['有色金属矿采选'],
        '有色金属冶炼和压延加工业': ['有色金属及其合金', '有色金属压延'],
        '金属制品业': ['金属制品'],
        '通用设备制造业': ['锅炉', '原动设备', '金属加工机械', '物料搬运设备', '泵', '阀门', '压缩机', '烘炉', '风机', '包装', '文化、办公用机械', '其他通用设备'],
        '专用设备制造业': ['采矿', '冶金', '建筑专用设备', '化工', '木材', '非金属加工专用设备', '农、林、牧、渔专用机械', '医疗仪器设备', '其他专用设备'],
        '汽车制造业': ['汽车整车', '汽车零部件'],
        '铁路、船舶、航空航天和其他运输设备制造业': ['铁路运输和城市轨道交通设备', '船舶', '其他交通运输设备'],
        '电气机械和器材制造业': ['电机', '输配电', '控制设备', '电线', '电缆', '光缆', '电工器材', '电池', '家用器具', '其他电气机械'],
        '计算机、通信和其他电子设备制造业': ['计算机', '通信设备', '广播电视设备', '雷达', '视听设备', '电子元器件', '其他电子设备'],
        '仪器仪表制造业': ['仪器仪表'],
        '其他制造业': ['其他制造产品'],
        '废弃资源综合利用业': ['废弃资源', '废旧材料回收'],
        '金属制品、机械和设备修理业': ['金属制品、机械和设备修理'],
        '电力、热力生产和供应业': ['电力', '热力生产和供应'],
        '燃气生产和供应业': ['燃气生产和供应'],
        '水的生产和供应业': ['水的生产和供应'],
        '煤炭开采和洗选业': ['煤炭采选', '煤炭开采'],
        '石油和天然气开采业': ['石油和天然气开采'],
        '黑色金属矿采选业': ['黑色金属矿采选'],
        '有色金属矿采选业': ['有色金属矿采选'],
        '非金属矿采选业': ['非金属矿采选'],
        '开采辅助活动': ['开采辅助'],
        '其他采矿业': ['其他采矿', '其他矿采选']
    }
    
    print(f"创建了 {len(keyword_mapping)} 个标准行业的映射规则")
    return keyword_mapping


def map_industry_to_standard(industry_name: str, mapping_rules: Dict[str, List[str]]) -> Optional[str]:
    """
    将单个行业映射到标准分类
    
    Args:
        industry_name: 原始行业名称
        mapping_rules: 映射规则字典
    
    Returns:
        Optional[str]: 标准行业名称，如果未找到匹配则返回None
    """
    # 优先匹配更具体的关键词，避免被通用关键词抢先匹配
    
    # 特殊处理：优先匹配完整的特定行业名称
    if '开采辅助' in industry_name and '采矿' in industry_name:
        return '开采辅助活动'
    
    if '金属制品、机械和设备修理' in industry_name:
        return '金属制品、机械和设备修理业'
    
    # 铁合金产品应该归到黑色金属冶炼和压延加工业
    if '铁合金产品' in industry_name:
        return '黑色金属冶炼和压延加工业'
    
    # 农、林、牧、渔专用机械应该归到专用设备制造业，不是农副食品加工业
    if '农、林、牧、渔专用机械' in industry_name:
        return '专用设备制造业'
    
    # 农、林、牧、渔服务是服务业，不应归到农副食品加工业
    if '农、林、牧、渔服务' in industry_name:
        return None  # 标记为未映射
    
    # 铁路运输相关的特殊处理
    if '铁路运输和城市轨道交通设备' in industry_name:
        return '铁路、船舶、航空航天和其他运输设备制造业'
    
    if '铁路运输' in industry_name and '设备' not in industry_name:
        return None  # 铁路运输服务标记为未映射
    
    # 其他采矿业的特殊处理
    if industry_name in ['其他采矿产品', '其他矿采选产品']:
        return '其他采矿业'
    
    # 常规匹配
    for std_industry, keywords in mapping_rules.items():
        for keyword in keywords:
            if keyword in industry_name:
                return std_industry
    return None


def process_year_data(year_data: Dict[str, pd.DataFrame], mapping_rules: Dict[str, List[str]]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    处理所有年份数据，生成映射关系和聚合数据

    Args:
        year_data: 各年份数据字典
        mapping_rules: 映射规则字典

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: (映射关系数据, 聚合数据)
    """
    print("正在处理年份数据...")

    # 存储映射关系的列表
    mapping_records = []
    # 存储聚合数据的列表
    aggregation_records = []

    for year, df in year_data.items():
        print(f"处理 {year} 年数据...")

        # 为每个原始行业创建映射记录
        for _, row in df.iterrows():
            industry_name = row['行业']
            industry_value = row['工业增加值']

            # 尝试映射到标准行业
            mapped_industry = map_industry_to_standard(industry_name, mapping_rules)

            # 添加到映射关系记录
            mapping_records.append({
                '原始行业名': industry_name,
                '年份': int(year),
                '归并行业名': mapped_industry if mapped_industry else '未映射'
            })

            # 如果映射成功，添加到聚合记录
            if mapped_industry and not pd.isna(industry_value):
                aggregation_records.append({
                    '归并行业名': mapped_industry,
                    '年份': int(year),
                    '原始行业名': industry_name,
                    '工业增加值': float(industry_value)
                })

    # 创建映射关系DataFrame
    mapping_df = pd.DataFrame(mapping_records)

    # 创建聚合数据DataFrame并按标准行业分组求和
    if aggregation_records:
        temp_df = pd.DataFrame(aggregation_records)
        # 按归并行业名和年份分组，对工业增加值求和
        consolidated_df = temp_df.groupby(['归并行业名', '年份'])['工业增加值'].sum().reset_index()

        # 计算各行业在每年的占比
        print("正在计算各行业占比...")
        # 计算每年的总和
        yearly_totals = consolidated_df.groupby('年份')['工业增加值'].sum()

        # 计算占比（小数形式，不乘以100）
        consolidated_df['权重_new'] = consolidated_df.apply(
            lambda row: (row['工业增加值'] / yearly_totals[row['年份']]), axis=1
        )

        # 修改行业名称格式：添加前缀和后缀
        consolidated_df['归并行业名'] = consolidated_df['归并行业名'].apply(
            lambda x: f"规模以上工业增加值:{x}:当月同比"
        )

        # 将数据从长格式转换为宽格式
        print("正在转换为宽格式数据...")
        consolidated_df = consolidated_df.pivot(index='归并行业名', columns='年份', values='权重_new')

        # 重命名列名，添加权重_前缀
        consolidated_df.columns = [f'权重_{int(year)}' for year in consolidated_df.columns]

        # 重置索引，使归并行业名成为普通列
        consolidated_df = consolidated_df.reset_index()

        # 填充可能的NaN值为0（如果某些年份缺少某些行业数据）
        consolidated_df = consolidated_df.fillna(0)
    else:
        # 如果没有数据，创建空的宽格式DataFrame
        years = list(year_data.keys())
        columns = ['归并行业名'] + [f'权重_{year}' for year in sorted(years)]
        consolidated_df = pd.DataFrame(columns=columns)

    print(f"生成映射关系记录: {len(mapping_df)} 条")
    print(f"生成聚合数据记录: {len(consolidated_df)} 条")

    return mapping_df, consolidated_df


def create_output_excel(consolidated_data: pd.DataFrame, mapping_data: pd.DataFrame, output_path: str) -> None:
    """
    创建输出Excel文件

    Args:
        consolidated_data: 归并后的数据
        mapping_data: 映射关系数据
        output_path: 输出文件路径
    """
    print(f"正在创建输出Excel文件: {output_path}")

    try:
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            # 写入第一个sheet：归并数据
            consolidated_data.to_excel(writer, sheet_name='归并数据', index=False)
            print(f"写入归并数据: {len(consolidated_data)} 行")

            # 写入第二个sheet：映射关系
            mapping_data.to_excel(writer, sheet_name='映射关系', index=False)
            print(f"写入映射关系: {len(mapping_data)} 行")

        print(f"Excel文件创建成功: {output_path}")

    except Exception as e:
        print(f"创建Excel文件失败: {e}")
        raise




def validate_results(consolidated_data: pd.DataFrame, mapping_data: pd.DataFrame, year_data: Dict[str, pd.DataFrame]) -> None:
    """
    验证结果数据的完整性和准确性

    Args:
        consolidated_data: 归并后的数据
        mapping_data: 映射关系数据
        year_data: 原始年份数据
    """
    print("\n=== 数据验证 ===")

    # 验证映射关系数据完整性
    total_original_records = sum(len(df) for df in year_data.values())
    print(f"原始数据总记录数: {total_original_records}")
    print(f"映射关系记录数: {len(mapping_data)}")

    if len(mapping_data) != total_original_records:
        print("警告: 映射关系记录数与原始数据不匹配")
    else:
        print("映射关系数据完整")

    # 统计映射成功率
    mapped_count = len(mapping_data[mapping_data['归并行业名'] != '未映射'])
    unmapped_count = len(mapping_data[mapping_data['归并行业名'] == '未映射'])
    mapping_rate = mapped_count / len(mapping_data) * 100

    print(f"映射成功: {mapped_count} 条 ({mapping_rate:.1f}%)")
    print(f"未映射: {unmapped_count} 条 ({100-mapping_rate:.1f}%)")

    # 验证聚合数据（宽格式）
    unique_industries = len(consolidated_data)
    # 获取年份列（以权重_开头的列）
    year_columns = [col for col in consolidated_data.columns if col.startswith('权重_')]
    unique_years = len(year_columns)
    print(f"归并后标准行业数: {unique_industries}")
    print(f"涵盖年份数: {unique_years}")

    # 显示各年份的归并行业数量
    for col in sorted(year_columns):
        year = col.replace('权重_', '')
        non_zero_industries = (consolidated_data[col] > 0).sum()
        print(f"  {year}年: {non_zero_industries} 个标准行业")


def main():
    """
    主执行函数
    """
    print("=== 工业增加值行业归并脚本 ===")

    try:
        # 步骤1: 读取数据
        year_data, standard_industries = load_industry_data()

        # 步骤2: 创建映射规则
        mapping_rules = create_industry_mapping()

        # 步骤3: 处理数据
        mapping_df, consolidated_df = process_year_data(year_data, mapping_rules)

        # 步骤4: 验证结果
        validate_results(consolidated_df, mapping_df, year_data)

        # 步骤5: 创建输出文件
        output_path = '工业增加值行业归并结果_final.xlsx'
        create_output_excel(consolidated_df, mapping_df, output_path)

        print("\n=== 脚本执行完成 ===")
        print(f"输出文件: {output_path}")
        print("\n文件说明:")
        print("- 第一个工作表'归并数据': 包含按标准行业归并后的工业增加值占比数据(权重_new)")
        print("- 第二个工作表'映射关系': 包含原始行业到标准行业的完整映射关系")

    except Exception as e:
        print(f"脚本执行失败: {e}")
        raise


if __name__ == "__main__":
    main()
