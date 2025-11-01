import pandas as pd
import sys
sys.stdout.reconfigure(encoding='utf-8')

file_path = 'data/经济数据库1027.xlsx'
target_variable = '聚酯瓶片:企业开工负荷:当周值'

# 读取所有sheets
all_sheets = pd.read_excel(file_path, sheet_name=None)

print(f"查找变量: {target_variable}\n")
print("=" * 80)

# 查找包含该变量的sheet
found = False
for sheet_name, df in all_sheets.items():
    if sheet_name == '指标体系':
        continue

    # 检查列名中是否包含目标变量
    if target_variable in df.columns or (len(df.columns) > 1 and target_variable in df.iloc[:, 1:].columns):
        found = True
        print(f"找到变量所在Sheet: {sheet_name}\n")

        # 如果第一列是时间列
        if len(df.columns) >= 2:
            # 转换第一列为datetime
            date_col = df.columns[0]
            df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
            df = df.dropna(subset=[date_col])
            df = df.set_index(date_col)
            df.sort_index(inplace=True)

            # 获取该变量的数据
            if target_variable in df.columns:
                series = df[target_variable]

                # 去除NaN值
                valid_series = series.dropna()

                print(f"原始数据统计:")
                print(f"  - Sheet名称: {sheet_name}")
                print(f"  - 第一个有效值时间: {valid_series.index.min().strftime('%Y-%m-%d')}")
                print(f"  - 最后一个有效值时间: {valid_series.index.max().strftime('%Y-%m-%d')}")
                print(f"  - 有效值个数: {len(valid_series)}")
                print(f"  - 总行数（包括NaN）: {len(series)}")

                print(f"\n前5个有效值:")
                print(valid_series.head())

                print(f"\n后5个有效值:")
                print(valid_series.tail())

        break

if not found:
    print(f"未找到变量: {target_variable}")
    print("\n可能的周度变量（包含'聚酯'或'瓶片'）:")
    for sheet_name, df in all_sheets.items():
        if '周度' in sheet_name:
            if len(df.columns) > 1:
                for col in df.columns[1:]:
                    if '聚酯' in str(col) or '瓶片' in str(col):
                        print(f"  - {sheet_name}: {col}")
