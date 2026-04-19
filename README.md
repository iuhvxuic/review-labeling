# 中文酒店评论情感分析

基于通义千问大模型的中文酒店评论情感自动打标系统。

## 📋 项目简介

本项目使用通义千问（qwen-turbo）大模型，对 ChnSentiCorp 中文酒店评论数据集进行自动情感分类打标，并对比原数据集标签生成详细的分析报告和可视化图表。

### 任务描述
- **数据集**：ChnSentiCorp（中文酒店评论）
- **输入特征**：ChnSentiCorp中的中文评论文本
- **打标目标**：情感二分类（正面/负面）


## 🚀 快速开始

### 1. 环境准备

**Python版本要求**：Python 3.8+  
**添加API-KEY**：在"src/label_reviews.py"文件中DASHSCOPE_API_KEY处添加API-KEY

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 运行程序

```bash
# 步骤1：提取评论（只提取评论列并清洗）
python src/extract_data.py

# 步骤2：大模型打标
python src/label_reviews.py

# 步骤3：结果分析
python src/analyze_results.py
```
