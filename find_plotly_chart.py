import os
import re

def find_plotly_chart_calls(directory):
    """搜索所有 plotly_chart 调用"""
    results = []

    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.py'):
                filepath = os.path.join(root, file)
                try:
                    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                        lines = content.split('\n')

                        for i, line in enumerate(lines, 1):
                            if 'plotly_chart' in line:
                                # 查找多行调用
                                start = max(0, i-2)
                                end = min(len(lines), i+5)
                                context = '\n'.join([f"  {j}: {lines[j-1]}" for j in range(start+1, end+1)])

                                results.append({
                                    'file': filepath,
                                    'line': i,
                                    'context': context
                                })
                except Exception as e:
                    print(f"Error reading {filepath}: {e}")

    return results

# 搜索
print("搜索 dashboard/analysis/industrial 目录...")
results = find_plotly_chart_calls('dashboard/analysis/industrial')

print(f"\n找到 {len(results)} 个 plotly_chart 调用:\n")

for r in results:
    print(f"\n{'='*80}")
    print(f"文件: {r['file']}")
    print(f"行号: {r['line']}")
    print(f"上下文:")
    print(r['context'])
