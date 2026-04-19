"""
文件3: analyze_results.py
功能: 读取源文件（含标签）和打标结果，进行对比分析，每个图表单独保存
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

SOURCE_CSV = "../data/ChnSentiCorp_htl_all.csv"
LABELED_CSV = "../data/labeled_results.csv"

print("=" * 60)
print("情感分析结果对比")
print("=" * 60)

# 读取文件
source_df = pd.read_csv(SOURCE_CSV)
labeled_df = pd.read_csv(LABELED_CSV)

print(f"\n源文件: {len(source_df)} 条")
print(f"打标结果: {len(labeled_df)} 条")

# 合并数据
merged = labeled_df.merge(
    source_df[['review', 'label']],
    on='review',
    how='inner'
)

print(f"匹配成功: {len(merged)} 条")

# 重命名列
merged = merged.rename(columns={
    'label_x': 'pred_label',
    'label_y': 'true_label'
})

# 转换标签格式
def convert_true_label(label):
    label_str = str(label).lower()
    if label_str in ['1', 'positive', 'pos']:
        return 1
    elif label_str in ['0', 'negative', 'neg']:
        return 0
    return None

def convert_pred_label(label):
    try:
        label_val = int(label)
        if label_val == 1:
            return 1
        elif label_val == 0:
            return 0
        else:
            return None
    except:
        return None

merged['true_label'] = merged['true_label'].apply(convert_true_label)
merged['pred_label'] = merged['pred_label'].apply(convert_pred_label)

# 过滤有效数据
valid = merged.dropna(subset=['true_label', 'pred_label'])
valid = valid[valid['pred_label'].isin([0, 1])]

print(f"有效数据: {len(valid)} 条")

if len(valid) == 0:
    print("错误: 没有有效数据可分析")
    exit(1)

# 计算准确率
valid['is_correct'] = valid['true_label'] == valid['pred_label']
accuracy = valid['is_correct'].mean()

print(f"\n整体准确率: {accuracy:.2%}")

# 各类别准确率
print("\n各类别准确率:")
for label, name in [(1, '正面'), (0, '负面')]:
    label_df = valid[valid['true_label'] == label]
    if len(label_df) > 0:
        acc = label_df['is_correct'].mean()
        print(f"  {name}: {acc:.2%} ({len(label_df)}条)")

# 计算指标
tn = len(valid[(valid['true_label']==0) & (valid['pred_label']==0)])
fp = len(valid[(valid['true_label']==0) & (valid['pred_label']==1)])
fn = len(valid[(valid['true_label']==1) & (valid['pred_label']==0)])
tp = len(valid[(valid['true_label']==1) & (valid['pred_label']==1)])

precision = tp / (tp + fp) if (tp + fp) > 0 else 0
recall = tp / (tp + fn) if (tp + fn) > 0 else 0
f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

print(f"\n详细指标:")
print(f"  Precision: {precision:.4f}")
print(f"  Recall: {recall:.4f}")
print(f"  F1-Score: {f1:.4f}")

# 错误案例
print("\n错误案例 (前5个):")
wrong_cases = valid[valid['is_correct'] == False].head(5)
if len(wrong_cases) > 0:
    for i, row in wrong_cases.iterrows():
        true_text = "正面" if row['true_label'] == 1 else "负面"
        pred_text = "正面" if row['pred_label'] == 1 else "负面"
        print(f"\n  原文: {str(row['review'])[:100]}...")
        print(f"  真实: {true_text} → 预测: {pred_text}")
else:
    print("  无错误案例！")

# ==================== 可视化展示（分开保存）====================
print("\n" + "=" * 60)
print("生成可视化图表...")
print("=" * 60)

pos_acc = valid[valid['true_label']==1]['is_correct'].mean() if len(valid[valid['true_label']==1])>0 else 0
neg_acc = valid[valid['true_label']==0]['is_correct'].mean() if len(valid[valid['true_label']==0])>0 else 0

# 图表1: 准确率柱状图
fig1, ax1 = plt.subplots(figsize=(8, 6))
accuracies = [accuracy, pos_acc, neg_acc]
bars = ax1.bar(['整体', '正面', '负面'], accuracies, color=['#3498db', '#2ecc71', '#e74c3c'])
ax1.set_ylim(0, 1)
ax1.set_ylabel('准确率')
ax1.set_title('各类别准确率对比')
ax1.axhline(y=0.9, color='gray', linestyle='--', alpha=0.5)
for bar, acc in zip(bars, accuracies):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
             f'{acc:.2%}', ha='center', va='bottom', fontsize=12)
plt.tight_layout()
plt.savefig('../img/chart1_accuracy.png', dpi=150, bbox_inches='tight')
plt.close()
print("已保存: chart1_accuracy.png")

# 图表2: 混淆矩阵热力图
fig2, ax2 = plt.subplots(figsize=(8, 6))
confusion_matrix = np.array([[tn, fp], [fn, tp]])
im = ax2.imshow(confusion_matrix, cmap='Blues')
ax2.set_xticks([0, 1])
ax2.set_yticks([0, 1])
ax2.set_xticklabels(['预测负面', '预测正面'])
ax2.set_yticklabels(['真实负面', '真实正面'])
ax2.set_title('混淆矩阵')
for i in range(2):
    for j in range(2):
        ax2.text(j, i, str(confusion_matrix[i, j]), ha='center', va='center', fontsize=16)
plt.colorbar(im, ax=ax2)
plt.tight_layout()
plt.savefig('../img/chart2_confusion_matrix.png', dpi=150, bbox_inches='tight')
plt.close()
print("已保存: chart2_confusion_matrix.png")

# 图表3: 预测结果分布饼图
fig3, ax3 = plt.subplots(figsize=(8, 6))
pred_counts = valid['pred_label'].value_counts()
labels = ['正面', '负面']
colors = ['#2ecc71', '#e74c3c']
ax3.pie(pred_counts, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
ax3.set_title('模型预测结果分布')
plt.tight_layout()
plt.savefig('../img/chart3_pred_distribution.png', dpi=150, bbox_inches='tight')
plt.close()
print("已保存: chart3_pred_distribution.png")

# 图表4: 真实标签分布饼图
fig4, ax4 = plt.subplots(figsize=(8, 6))
true_counts = valid['true_label'].value_counts()
ax4.pie(true_counts, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
ax4.set_title('真实标签分布')
plt.tight_layout()
plt.savefig('../img/chart4_true_distribution.png', dpi=150, bbox_inches='tight')
plt.close()
print("已保存: chart4_true_distribution.png")

# 图表5: 指标雷达图
fig5, ax5 = plt.subplots(figsize=(8, 6), subplot_kw={'projection': 'polar'})
metric_names = ['准确率', '精确率', '召回率', 'F1分数']
metric_values = [accuracy, precision, recall, f1]
angles = np.linspace(0, 2 * np.pi, len(metric_names), endpoint=False).tolist()
metric_values += metric_values[:1]
angles += angles[:1]
ax5.plot(angles, metric_values, 'o-', linewidth=2, color='#3498db')
ax5.fill(angles, metric_values, alpha=0.25, color='#3498db')
ax5.set_xticks(angles[:-1])
ax5.set_xticklabels(metric_names)
ax5.set_ylim(0, 1)
ax5.set_title('性能指标雷达图')
plt.tight_layout()
plt.savefig('../img/chart5_radar.png', dpi=150, bbox_inches='tight')
plt.close()
print("已保存: chart5_radar.png")

# 图表6: 错误类型分析
fig6, ax6 = plt.subplots(figsize=(8, 6))
fp_count = len(valid[(valid['true_label']==0) & (valid['pred_label']==1)])
fn_count = len(valid[(valid['true_label']==1) & (valid['pred_label']==0)])
error_types = ['假阳性\n(负面判正面)', '假阴性\n(正面判负面)']
error_counts = [fp_count, fn_count]
colors_error = ['#f39c12', '#e74c3c']
bars = ax6.bar(error_types, error_counts, color=colors_error)
ax6.set_ylabel('数量')
ax6.set_title('错误类型分布')
for bar, count in zip(bars, error_counts):
    ax6.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
             str(count), ha='center', va='bottom', fontsize=14)
plt.tight_layout()
plt.savefig('../img/chart6_error_types.png', dpi=150, bbox_inches='tight')
plt.close()
print("已保存: chart6_error_types.png")

# 输出总结报告
print("\n" + "=" * 60)
print("分析报告总结")
print("=" * 60)
print(f"""
┌────────────────────────────────────────────────────────────┐
│                      性能指标汇总                           │
├────────────────────────────────────────────────────────────┤
│  整体准确率:     {accuracy:.2%}                                     
│  正面准确率:     {pos_acc:.2%}                     
│  负面准确率:     {neg_acc:.2%}                     
├────────────────────────────────────────────────────────────┤
│  Precision:      {precision:.4f}                                         
│  Recall:         {recall:.4f}                                         
│  F1-Score:       {f1:.4f}                                         
├────────────────────────────────────────────────────────────┤
│  正确预测:       {valid['is_correct'].sum()} 条                                    
│  错误预测:       {(~valid['is_correct']).sum()} 条                                    
│  假阳性(FP):     {fp_count} 条                                          
│  假阴性(FN):     {fn_count} 条                                          
└────────────────────────────────────────────────────────────┘
""")

print("\n✅ 分析完成！")
print("生成的图表文件:")
print("  1. chart1_accuracy.png - 各类别准确率对比")
print("  2. chart2_confusion_matrix.png - 混淆矩阵")
print("  3. chart3_pred_distribution.png - 模型预测结果分布")
print("  4. chart4_true_distribution.png - 真实标签分布")
print("  5. chart5_radar.png - 性能指标雷达图")
print("  6. chart6_error_types.png - 错误类型分布")