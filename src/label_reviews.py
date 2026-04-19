"""
文件2: label_reviews.py
功能: 读取评论，调用大模型打标，保存结果（格式：label, review）
"""

import pandas as pd
import time
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

# ==================== 配置 ====================
DASHSCOPE_API_KEY = "API_KEY"
BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
MODEL_NAME = "qwen-turbo"

INPUT_CSV = "../data/reviews_only.csv"
OUTPUT_CSV = "../data/labeled_results.csv"

# ==================== 初始化 ====================
llm = ChatOpenAI(
    model=MODEL_NAME,
    api_key=DASHSCOPE_API_KEY,
    base_url=BASE_URL,
    temperature=0.1,
    max_tokens=50,
)

SYSTEM_PROMPT = """你是一位情感分析专家。对中文酒店评论进行情感分类。

分类标准：
- 正面情感：输出 1
- 负面情感：输出 0

只输出一个数字，不要输出任何其他内容。"""

# ==================== 打标函数 ====================
def label_text(text):
    try:
        text = text.strip()[:800]
        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=f"评论：{text}")
        ]
        result = llm.invoke(messages)
        # 提取数字
        output = result.content.strip()
        if '1' in output:
            return 1
        else:
            return 0
    except Exception as e:
        print(f"错误: {e}")
        return None

# ==================== 主程序 ====================
print(f"读取文件: {INPUT_CSV}")
df = pd.read_csv(INPUT_CSV)
print(f"共 {len(df)} 条评论")

results = []

for idx, row in df.iterrows():
    review_text = row.iloc[0]
    print(f"处理 {idx+1}/{len(df)}: {review_text[:50]}...")

    label = label_text(review_text)

    if label is not None:
        results.append({
            "label": label,
            "review": review_text
        })
    else:
        results.append({
            "label": -1,
            "review": review_text
        })

    time.sleep(0.2)

# 保存结果
result_df = pd.DataFrame(results)
result_df.to_csv(OUTPUT_CSV, index=False, encoding='utf-8-sig')
print(f"\n完成! 结果保存到 {OUTPUT_CSV}")
print(f"正面(1): {len(result_df[result_df['label']==1])} 条")
print(f"负面(0): {len(result_df[result_df['label']==0])} 条")
print(f"失败(-1): {len(result_df[result_df['label']==-1])} 条")