#!/usr/bin/env python
"""
消融实验对比可视化脚本。
绘制三模态(全) vs RGB+Thermal(无Event)的mAP/mAR对比图。
"""
import json
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# 设置中文字体
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# === 数据 ===
full_data = {
    "mAP": 0.8557, "mAP_50": 0.9830, "mAP_75": 0.9593,
    "mAP_small": 0.6665, "mAP_medium": 0.8692, "mAP_large": 0.7430,
    "mAR_1": 0.2859, "mAR_10": 0.8805, "mAR_100": 0.8881,
    "mAR_small": 0.7573, "mAR_medium": 0.9004, "mAR_large": 0.7625,
}
no_event_data = {
    "mAP": 0.8347, "mAP_50": 0.9814, "mAP_75": 0.9512,
    "mAP_small": 0.6298, "mAP_medium": 0.8474, "mAP_large": 0.7511,
    "mAR_1": 0.2801, "mAR_10": 0.8622, "mAR_100": 0.8699,
    "mAR_small": 0.7401, "mAR_medium": 0.8821, "mAR_large": 0.7750,
}

# Training progress (no_event)
epochs = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
train_loss = [0.3814, 0.2485, 0.1923, 0.1655, 0.1461, 0.1359, 0.1261, 0.1197, 0.1103, 0.1044, 0.1015, 0.0977, 0.0975, 0.0911, 0.0881]
eval_mAP = [None]*4 + [0.7913] + [None]*4 + [0.8199] + [None]*4 + [0.8347]
eval_mAP50 = [None]*4 + [0.9777] + [None]*4 + [0.9811] + [None]*4 + [0.9814]
eval_mAP75 = [None]*4 + [0.9351] + [None]*4 + [0.9509] + [None]*4 + [0.9512]

# === 图1: 消融对比柱状图 ===
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# mAP comparison
metrics = ['mAP', 'mAP@50', 'mAP@75', 'mAP_s', 'mAP_m', 'mAP_l']
full_vals = [full_data['mAP'], full_data['mAP_50'], full_data['mAP_75'],
             full_data['mAP_small'], full_data['mAP_medium'], full_data['mAP_large']]
noev_vals = [no_event_data['mAP'], no_event_data['mAP_50'], no_event_data['mAP_75'],
             no_event_data['mAP_small'], no_event_data['mAP_medium'], no_event_data['mAP_large']]

x = np.arange(len(metrics))
w = 0.35
ax = axes[0]
bars1 = ax.bar(x - w/2, full_vals, w, label='RGB+Thermal+Event (Full)', color='#2196F3', edgecolor='white')
bars2 = ax.bar(x + w/2, noev_vals, w, label='RGB+Thermal (No Event)', color='#FF9800', edgecolor='white')
ax.set_ylabel('Score', fontsize=12)
ax.set_title('Detection Accuracy (mAP)', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(metrics, fontsize=10)
ax.legend(fontsize=9)
ax.set_ylim(0.5, 1.0)
ax.grid(axis='y', alpha=0.3)
for bar, val in zip(bars1, full_vals):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005, f'{val:.3f}', ha='center', fontsize=7)
for bar, val in zip(bars2, noev_vals):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005, f'{val:.3f}', ha='center', fontsize=7)

# mAR comparison
metrics_ar = ['mAR@1', 'mAR@10', 'mAR@100', 'mAR_s', 'mAR_m', 'mAR_l']
full_ar = [full_data['mAR_1'], full_data['mAR_10'], full_data['mAR_100'],
           full_data['mAR_small'], full_data['mAR_medium'], full_data['mAR_large']]
noev_ar = [no_event_data['mAR_1'], no_event_data['mAR_10'], no_event_data['mAR_100'],
           no_event_data['mAR_small'], no_event_data['mAR_medium'], no_event_data['mAR_large']]

ax = axes[1]
bars1 = ax.bar(x - w/2, full_ar, w, label='RGB+Thermal+Event (Full)', color='#2196F3', edgecolor='white')
bars2 = ax.bar(x + w/2, noev_ar, w, label='RGB+Thermal (No Event)', color='#FF9800', edgecolor='white')
ax.set_ylabel('Score', fontsize=12)
ax.set_title('Detection Recall (mAR)', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(metrics_ar, fontsize=10)
ax.legend(fontsize=9)
ax.set_ylim(0.2, 1.0)
ax.grid(axis='y', alpha=0.3)
for bar, val in zip(bars1, full_ar):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005, f'{val:.3f}', ha='center', fontsize=7)
for bar, val in zip(bars2, noev_ar):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005, f'{val:.3f}', ha='center', fontsize=7)

# Event contribution (delta)
ax = axes[2]
delta_mAP = [full_data[k] - no_event_data[k] for k in ['mAP', 'mAP_50', 'mAP_75', 'mAP_small', 'mAP_medium', 'mAP_large']]
delta_mAR = [full_data[k] - no_event_data[k] for k in ['mAR_1', 'mAR_10', 'mAR_100', 'mAR_small', 'mAR_medium', 'mAR_large']]
all_deltas = delta_mAP + delta_mAR
all_labels = metrics + metrics_ar
colors = ['#4CAF50' if d > 0 else '#F44336' for d in all_deltas]

bars = ax.bar(range(len(all_deltas)), all_deltas, color=colors, edgecolor='white')
ax.set_ylabel('Delta Score', fontsize=12)
ax.set_title('Event Camera Contribution (Full - No Event)', fontsize=14, fontweight='bold')
ax.set_xticks(range(len(all_deltas)))
ax.set_xticklabels(all_labels, fontsize=8, rotation=45)
ax.axhline(y=0, color='black', linewidth=0.5)
ax.grid(axis='y', alpha=0.3)
for bar, val in zip(bars, all_deltas):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001, f'{val:+.4f}', ha='center', fontsize=7)

plt.tight_layout()
plt.savefig('results/ablation_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: results/ablation_comparison.png")

# === 图2: 训练曲线 (No Event) ===
fig, ax1 = plt.subplots(figsize=(12, 5))

color = '#2196F3'
ax1.set_xlabel('Epoch', fontsize=12)
ax1.set_ylabel('Training Loss', color=color, fontsize=12)
ax1.plot(epochs, train_loss, 'o-', color=color, linewidth=2, markersize=6, label='Train Loss')
ax1.tick_params(axis='y', labelcolor=color)
ax1.set_ylim(0, 0.5)
ax1.grid(alpha=0.3)

ax2 = ax1.twinx()
color2 = '#FF5722'
eval_epochs = [e for e, v in zip(epochs, eval_mAP) if v is not None]
eval_vals = [v for v in eval_mAP if v is not None]
ax2.set_ylabel('Test mAP', color=color2, fontsize=12)
ax2.plot(eval_epochs, eval_vals, 's-', color=color2, linewidth=2, markersize=10, label='Test mAP')
ax2.tick_params(axis='y', labelcolor=color2)
ax2.set_ylim(0.7, 1.0)

for ep, val in zip(eval_epochs, eval_vals):
    ax2.annotate(f'{val:.4f}', (ep, val), textcoords="offset points", xytext=(0, 12), ha='center', fontsize=9, color=color2)

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=10)

plt.title('Training Progress: RGB+Thermal (No Event) — mit_b1, batch_size=3', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('results/training_curve_no_event.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: results/training_curve_no_event.png")

# === 图3: Event贡献雷达图 ===
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, polar=True)

categories = ['mAP', 'mAP@50', 'mAP@75', 'mAP_small', 'mAP_medium', 'mAP_large']
N = len(categories)
angles = [n / float(N) * 2 * np.pi for n in range(N)]
angles += angles[:1]

full_radar = [full_data[c.replace('@', '_')] if '@' in c else full_data.get(c, 0) for c in categories]
noev_radar = [no_event_data[c.replace('@', '_')] if '@' in c else no_event_data.get(c, 0) for c in categories]

# Remap full_data keys
full_radar = [full_data['mAP'], full_data['mAP_50'], full_data['mAP_75'],
              full_data['mAP_small'], full_data['mAP_medium'], full_data['mAP_large']]
noev_radar = [no_event_data['mAP'], no_event_data['mAP_50'], no_event_data['mAP_75'],
              no_event_data['mAP_small'], no_event_data['mAP_medium'], no_event_data['mAP_large']]

full_radar += full_radar[:1]
noev_radar += noev_radar[:1]

ax.plot(angles, full_radar, 'o-', linewidth=2, color='#2196F3', label='Full (RGB+Thermal+Event)')
ax.fill(angles, full_radar, alpha=0.1, color='#2196F3')
ax.plot(angles, noev_radar, 's-', linewidth=2, color='#FF9800', label='No Event (RGB+Thermal)')
ax.fill(angles, noev_radar, alpha=0.1, color='#FF9800')

ax.set_xticks(angles[:-1])
ax.set_xticklabels(categories, fontsize=11)
ax.set_ylim(0.55, 1.0)
ax.set_title('Event Camera: Detection Performance Radar', fontsize=14, fontweight='bold', pad=20)
ax.legend(loc='upper right', bbox_to_anchor=(1.2, 1.1), fontsize=10)

plt.tight_layout()
plt.savefig('results/ablation_radar.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: results/ablation_radar.png")

# === 文本报告 ===
print("\n" + "=" * 60)
print("EVENT CAMERA CONTRIBUTION ANALYSIS")
print("=" * 60)
print(f"{'Metric':<15} {'Full':>8} {'No Event':>8} {'Delta':>8} {'Gain %':>8}")
print("-" * 55)
for key in ['mAP', 'mAP_50', 'mAP_75', 'mAP_small', 'mAP_medium', 'mAP_large',
            'mAR_1', 'mAR_10', 'mAR_100', 'mAR_small', 'mAR_medium', 'mAR_large']:
    f = full_data[key]
    n = no_event_data[key]
    d = f - n
    pct = (d / n) * 100
    print(f"{key:<15} {f:8.4f} {n:8.4f} {d:+8.4f} {pct:+7.1f}%")
print("=" * 60)
