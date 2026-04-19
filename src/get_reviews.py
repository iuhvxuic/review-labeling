"""
文件1: extract_data.py
功能: 从ChnSentiCorp_htl_all.csv中只提取评论，保存为新的CSV文件
"""

import pandas as pd

# 配置文件路径
INPUT_CSV = "../data/ChnSentiCorp_htl_all.csv"
OUTPUT_CSV = "../data/reviews_only.csv"

# 读取原始文件
print(f"正在读取: {INPUT_CSV}")
df = pd.read_csv(INPUT_CSV)
print(f"原始数据: {len(df)} 条")

# 找到评论列
review_col = None
for col in df.columns:
    if 'review' in col.lower() or 'text' in col.lower() or 'comment' in col.lower():
        review_col = col
        break

if review_col is None:
    print(f"错误: 未找到评论列，可用列: {df.columns.tolist()}")
    exit(1)

print(f"找到评论列: {review_col}")

# 提取评论列
reviews = df[review_col].copy()

# 数据清洗
print("正在清洗数据...")

# 1. 删除空值
before_drop = len(reviews)
reviews = reviews.dropna()
print(f"删除空值: {before_drop - len(reviews)} 条")

# 2. 转换为字符串类型
reviews = reviews.astype(str)

# 3. 删除空字符串和无效内容
reviews = reviews[reviews.str.strip() != '']
reviews = reviews[reviews != 'nan']
reviews = reviews[reviews != 'None']

print(f"删除无效内容: {before_drop - len(reviews)} 条")

# 4. 清洗文本（可选：去除多余空格、特殊字符等）
reviews = reviews.str.strip()
reviews = reviews.str.replace(r'\s+', ' ', regex=True)

# 5. 只保留评论列
result_df = pd.DataFrame({'review': reviews})

# 保存
result_df.to_csv(OUTPUT_CSV, index=False, encoding='utf-8-sig')

print(f"\n完成!")
print(f"原始: {before_drop} 条 → 有效: {len(result_df)} 条")
print(f"已保存到: {OUTPUT_CSV}")

# 显示前3条预览
print("\n预览前3条:")
for i in range(min(3, len(result_df))):
    print(f"  {i+1}. {result_df.iloc[i]['review'][:100]}...")