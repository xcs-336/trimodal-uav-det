# TriModalDet — 三模态无人机目标检测

## 项目概述

TriModalDet 是一个基于 RGB、热红外（Thermal）和事件相机（Event）的三模态目标检测框架。通过 MAGE（Modality-Aware Gated Exchange）和 BiTE（Bidirectional Token Exchange）融合机制，在不同环境条件下实现鲁棒的目标检测。

## 架构设计

### 总体架构

```
输入 (B, 5, H, W)
├── RGB 分支 (3 ch)  → Patch Embed → MiT Transformer Stages 1-4
├── X 分支 (2 ch: Thermal+Event) → Patch Embed → MiT Transformer Stages 1-4
│   ├── MAGE（通道+空间门控） ← 自适应模态加权
│   └── BiTE（跨模态注意力） ← 双向特征交换
└── FPN (4 stages) → Faster R-CNN 检测头
```

### 核心组件

**1. 跨模态骨干网络（Inter-Modal Backbone）**
- 基于 SegFormer MiT（Mix Transformer）的双分支架构
- RGB 分支：处理 3 通道 RGB 图像
- 辅助模态分支（X）：处理 2 通道热红外+事件相机数据
- 支持 5 种变体（mit_b0 ~ mit_b4），平衡速度/精度

**2. 三模态融合模块**
- **MAGE（Modality-Aware Gated Exchange）**：通道门控 + 空间门控，自适应加权各模态贡献
- **BiTE（Bidirectional Token Exchange）**：跨模态注意力，实现双向信息流动
- 阶段级融合：在每个 Transformer Stage 后进行融合

**3. 检测头**
- **FPN（Feature Pyramid Network）**：多尺度特征聚合
- **Faster R-CNN**：分类 + 定位（RPN 提案 + RoI Align + 分类/回归）

## 数据格式

### 输入图像

图像为 `.npy` 格式的 NumPy 数组，shape 为 `(H, W, 5)`：
- **通道 0-2**：RGB
- **通道 3**：热红外（Thermal）
- **通道 4**：事件相机（Event）

```python
import numpy as np
image = np.zeros((480, 640, 5), dtype=np.uint8)
image[:, :, 0:3] = rgb_data
image[:, :, 3] = thermal_data
image[:, :, 4] = event_data
np.save('data/images/frame_001.npy', image)
```

### 标注格式

使用 YOLO 格式（每行一个目标）：
```
class_id x_center y_center width height
```
所有坐标归一化到 `[0, 1]`。

### 目录结构

```
data/
├── images/    (放置 .npy 图像文件)
└── labels/    (放置 .txt 标注文件)
```

## 环境配置

### 环境信息

- **Python**：3.12
- **PyTorch**：2.9.1 + CUDA 12.8
- **GPU**：RTX 5080（Blackwell 架构，16GB 显存）
- **Conda 环境**：`triair`

### 安装步骤

```bash
conda create -n triair python=3.12 -y
conda activate triair

# 安装 PyTorch CUDA 12.8 版本
pip install torch==2.9.1+cu128 torchvision==0.24.1+cu128 torchaudio==2.9.1+cu128 --index-url https://download.pytorch.org/whl/cu128

# 安装其余依赖
pip install timm matplotlib scikit-learn torchmetrics pycocotools

# 或直接使用 requirements2.txt
pip install -r requirements2.txt --index-url https://download.pytorch.org/whl/cu128
```

### 验证环境

```python
import torch
print(torch.__version__)          # 2.9.1+cu128
print(torch.cuda.is_available())  # True
print(torch.version.cuda)         # 12.8
```

---

## 使用指南

### 一、主模型训练

主模型使用完整的三模态输入（RGB + Thermal + Event），适合最终部署。

```bash
# RTX 5080 16GB 推荐配置（mit_b1，batch_size=1 或 2）
python scripts/train.py --data E:\dataset\CV\triair\data --epochs 15 --batch-size 1

# 快速原型验证（mit_b0，最小模型，batch_size 可稍大）
python scripts/train.py --data E:\dataset\CV\triair\data --epochs 15 --backbone mit_b0 --batch-size 4

# 高精度配置（mit_b4，需减小 batch_size）
python scripts/train.py --data E:\dataset\CV\triair\data --epochs 15 --backbone mit_b4 --batch-size 1
```

#### 训练监控参数

训练脚本内置资源监控，支持以下参数：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--monitor-interval` | `5.0` | 资源监控采样间隔（秒） |
| `--monitor-output` | `results/monitor_log.json` | 监控日志 JSON 输出路径 |
| `--max-gpu-mem-pct` | `85.0` | GPU 显存使用率警告阈值（百分比） |
| `--enable-oom-protection` | `True` | 是否启用 OOM 保护 |
| `--check-interval-batches` | `50` | 每隔多少 batch 检查一次显存 |

```bash
# 自定义监控参数
python scripts/train.py \
    --data E:\dataset\CV\triair\data \
    --epochs 15 --batch-size 1 \
    --monitor-interval 5 --max-gpu-mem-pct 85
```

#### 显存监控与 OOM 保护

训练过程中会自动：

1. **后台监控**：每 N 秒采集 GPU 显存/利用率、CPU 占用率、系统内存
2. **显存检查**：每 50 个 batch 检查一次 GPU 显存使用率，超过 85% 时打印警告
3. **碎片清理**：每 100 个 batch 自动调用 `torch.cuda.empty_cache()`
4. **OOM 保护**：当发生 CUDA OOM 时，自动捕获错误并保存 emergency checkpoint
5. **中断保护**：按 `Ctrl+C` 时自动保存当前进度
6. **优化建议**：训练结束时打印资源摘要和调参建议

```bash
# 监控日志保存在 results/monitor_log.json，可用以下命令查看
python -c "import json; data=json.load(open('results/monitor_log.json')); print(f'{len(data)} records')"
```

#### RTX 5080 显存参考

| 配置 | 显存占用 | 状态 |
|------|---------|------|
| batch_size=1, mit_b1 | ~7.2 GB (44%) | ✅ 安全 |
| batch_size=2, mit_b1 | ~9.3 GB (57%) | ✅ 安全 |
| batch_size=4, mit_b1 | ~19 GB (116%) | ❌ OOM |
| batch_size=4, mit_b0 | 待测试 | - |

> **建议**：如需更大的等效 batch_size，可使用梯度累积（见下方高级用法）。

### 二、主模型测试

训练完成后，使用保存的权重文件进行测试：

```bash
# 使用默认权重路径
python scripts/test.py --data E:\dataset\CV\triair\data --model trimodaldet.pth
```

测试脚本会：
1. 加载模型权重（支持 `state_dict`、`model_state_dict` 等多种格式）
2. 在测试集上计算 mAP、mAP@50、mAP@75 等指标
3. 将每个样本的预测结果可视化保存到 `test_results/` 目录

---

## 消融实验

消融实验用于分析不同输入模态（RGB、Thermal、Event）对检测性能的贡献。支持任意模态组合，可独立训练和测试。

### 消融实验概览

| 实验 | 模态组合 | 命令示例 | 目的 |
|------|---------|---------|------|
| 全模态基线 | RGB + Thermal + Event | `--modalities rgb,thermal,event` | 最佳性能参考 |
| 无事件相机 | RGB + Thermal | `--modalities rgb,thermal` | 评估事件相机贡献 |
| 无热红外 | RGB + Event | `--modalities rgb,event` | 评估热红外贡献 |
| 仅 RGB | RGB | `--modalities rgb` | 单模态基线 |

### 消融实验训练

```bash
# 全模态基线（RGB + Thermal + Event）
python trimodaldet/ablations/scripts/train_modality_ablation.py \
    --data E:\dataset\CV\triair\data\images \
    --labels E:\dataset\CV\triair\data\labels \
    --epochs 15 --batch-size 1 --backbone mit_b1 \
    --modalities rgb,thermal,event \
    --output-dir results/ablation_full
```

```bash
# 消融：无事件相机（RGB + Thermal）
python trimodaldet/ablations/scripts/train_modality_ablation.py \
    --data E:\dataset\CV\triair\data\images \
    --labels E:\dataset\CV\triair\data\labels \
    --epochs 15 --batch-size 1 --backbone mit_b1 \
    --modalities rgb,thermal \
    --output-dir results/ablation_no_event
```

```bash
# 消融：无热红外（RGB + Event）
python trimodaldet/ablations/scripts/train_modality_ablation.py \
    --data E:\dataset\CV\triair\data\images \
    --labels E:\dataset\CV\triair\data\labels \
    --epochs 15 --batch-size 1 --backbone mit_b1 \
    --modalities rgb,event \
    --output-dir results/ablation_no_thermal
```

```bash
# 消融：仅 RGB（单模态基线）
python trimodaldet/ablations/scripts/train_modality_ablation.py \
    --data E:\dataset\CV\triair\data\images \
    --labels E:\dataset\CV\triair\data\labels \
    --epochs 15 --batch-size 1 --backbone mit_b1 \
    --modalities rgb \
    --output-dir results/ablation_rgb_only
```

### 消融实验参数说明

| 参数 | 必填 | 说明 |
|------|------|------|
| `--data` | 是 | 图像目录路径（images 文件夹） |
| `--labels` | 是 | 标签目录路径（labels 文件夹） |
| `--output-dir` | 是 | 输出目录（日志、checkpoint、评估结果） |
| `--modalities` | 否 | 模态列表，逗号分隔。可选：`rgb`, `thermal`, `event`。默认 `rgb,thermal,event` |
| `--backbone` | 否 | 骨干网络，默认 `mit_b1` |
| `--epochs` | 否 | 训练轮数，默认 `15` |
| `--batch-size` | 否 | 批次大小，默认 `16`（RTX 5080 建议 `1`） |
| `--lr` | 否 | 学习率，默认 `0.02` |

### 消融实验输出

训练完成后，`--output-dir` 目录中会生成以下文件：

```
results/ablation_no_event/
├── training.log              # 训练日志（含每个 batch 的 loss 详情）
├── config.json               # 实验配置快照
├── model_info.json           # 模型参数量信息
├── metrics_per_epoch.csv     # 每 epoch 的 train_loss / mAP / mAP@50 / mAP@75
├── metrics_per_batch.csv     # 每 batch 的各损失分量
├── evaluation_history.json   # 评估历史记录
├── final_results.json        # 最终结果汇总
├── checkpoint_best.pth       # 最佳 checkpoint（完整训练状态，含 optimizer）
├── model_best_weights.pth    # 最佳模型权重（仅 state_dict，可跨脚本加载）
├── checkpoint_latest.pth     # 最新 checkpoint
└── checkpoint_epoch_N.pth    # 每 5 个 epoch 的 checkpoint
```

### 消融实验测试（仅评估已训练模型）

消融实验训练过程中会在每 5 个 epoch 自动评估一次（Training 日志中可见）。如果只想对已保存的 checkpoint 进行独立测试，使用以下脚本：

```bash
# 消融模型专用测试脚本
python trimodaldet/ablations/scripts/test_modality_ablation.py \
    --data E:\dataset\CV\triair\data\images \
    --labels E:\dataset\CV\triair\data\labels \
    --backbone mit_b1 \
    --modalities rgb,thermal \
    --checkpoint results/ablation_no_event/model_best_weights.pth \
    --output-dir results/ablation_no_event/test_results
```

> 测试脚本 `test_modality_ablation.py` 会：
> 1. 重建与训练时完全相同的模型架构（含 `ModalityConfigurableBackbone`）
> 2. 加载保存的最佳权重
> 3. 在测试集上计算 mAP/mAP@50/mAP@75/mAR 等所有指标
> 4. 将结果保存到 `--output-dir`

### 消融实验快速对比

如果已完成多个消融实验的训练，可以查看各自的 `final_results.json` 快速对比：

```bash
# 查看全模态结果
python -c "import json; r=json.load(open('results/ablation_full/final_results.json')); print('Full:', r['final_test_results']['mAP'])"

# 查看无事件相机结果
python -c "import json; r=json.load(open('results/ablation_no_event/final_results.json')); print('No Event:', r['final_test_results']['mAP'])"

# 查看无热红外结果
python -c "import json; r=json.load(open('results/ablation_no_thermal/final_results.json')); print('No Thermal:', r['final_test_results']['mAP'])"
```

### 消融实验完整工作流

```bash
# Step 1: 训练全模态基线
python trimodaldet/ablations/scripts/train_modality_ablation.py \
    --data E:\dataset\CV\triair\data\images \
    --labels E:\dataset\CV\triair\data\labels \
    --epochs 15 --batch-size 1 --backbone mit_b1 \
    --modalities rgb,thermal,event \
    --output-dir results/ablation_full

# Step 2: 训练无事件相机消融
python trimodaldet/ablations/scripts/train_modality_ablation.py \
    --data E:\dataset\CV\triair\data\images \
    --labels E:\dataset\CV\triair\data\labels \
    --epochs 15 --batch-size 1 --backbone mit_b1 \
    --modalities rgb,thermal \
    --output-dir results/ablation_no_event

# Step 3: 独立测试已训练模型（如只需评估，跳过训练）
python trimodaldet/ablations/scripts/test_modality_ablation.py \
    --data E:\dataset\CV\triair\data\images \
    --labels E:\dataset\CV\triair\data\labels \
    --backbone mit_b1 \
    --modalities rgb,thermal,event \
    --checkpoint results/ablation_full/model_best_weights.pth \
    --output-dir results/ablation_full/test_results

# Step 4: 对比结果
python -c "
import json
for exp in ['ablation_full', 'ablation_no_event']:
    r = json.load(open(f'results/{exp}/final_results.json'))
    cfg = r['config']['modality_config']
    mAP = r['final_test_results']['mAP']
    mAP50 = r['final_test_results']['mAP_50']
    print(f'{cfg}: mAP={mAP:.4f}, mAP@50={mAP50:.4f}')
"
```

---

## 可视化

```bash
# 可视化数据集样本（显示第 0 个样本的各通道和标注框）
python scripts/visualize.py --vis 0 --data E:\dataset\CV\triair\data

# 可视化指定索引的样本
python scripts/visualize.py --vis 100 --data E:\dataset\CV\triair\data
```

---

## 配置参数

### 训练参数

| 参数 | CLI 参数 | 默认值 | 说明 |
|------|---------|--------|------|
| `backbone_type` | `--backbone` | `mit_b1` | 骨干网络变体 |
| `num_epochs` | `--epochs` | `15` | 训练轮数 |
| `batch_size` | `--batch-size` | `16` | 批次大小（RTX 5080 建议 1-2） |
| `learning_rate` | `--lr` | `0.02` | 学习率 |
| `momentum` | - | `0.9` | SGD 动量 |
| `weight_decay` | - | `0.0001` | 权重衰减 |
| `test_size` | - | `0.2` | 测试集比例 |
| `score_threshold` | - | `0.5` | 置信度阈值 |

### 监控参数

| 参数 | CLI 参数 | 默认值 | 说明 |
|------|---------|--------|------|
| `monitor_interval` | `--monitor-interval` | `5.0` | 资源采样间隔（秒） |
| `monitor_output` | `--monitor-output` | `results/monitor_log.json` | 监控日志路径 |
| `max_gpu_mem_pct` | `--max-gpu-mem-pct` | `85.0` | 显存警告阈值（%） |
| `enable_oom_protection` | `--enable-oom-protection` | `True` | OOM 保护开关 |
| `check_interval_batches` | `--check-interval-batches` | `50` | 显存检查间隔（batch） |

### 数据参数

| 参数 | CLI 参数 | 说明 |
|------|---------|------|
| `data_root` | `--data` | 数据根目录（含 images/ 和 labels/ 子目录） |
| `model_path` | `--model` | 模型/checkpoint 路径 |

### 骨干网络对比

| 变体 | 参数量 | 深度 | 嵌入维度 | 适用场景 |
|------|--------|------|----------|----------|
| mit_b0 | ~3.7M | [2,2,2,2] | [32,64,160,256] | 快速原型、边缘设备、显存受限 |
| mit_b1 | ~13.5M | [2,2,2,2] | [64,128,320,512] | 默认均衡配置 |
| mit_b2 | ~24.7M | [3,4,6,3] | [64,128,320,512] | 更高精度 |
| mit_b3 | ~44M | [3,4,18,3] | [64,128,320,512] | 中大型模型 |
| mit_b4 | ~61.4M | [3,8,27,3] | [64,128,320,512] | 最大精度 |

---

## 高级用法

### 梯度累积（模拟大 batch_size）

当显存不足以支持大 batch_size 时，使用梯度累积保持等效 batch_size：

```python
# 修改 trainer.py 或训练脚本，在 optimizer.step() 前增加累积逻辑
accumulation_steps = 4  # 等效 batch_size = 1 × 4 = 4
if (i + 1) % accumulation_steps == 0:
    optimizer.step()
    optimizer.zero_grad()
```

### 内存映射加载

项目已默认启用 `np.load(mmap_mode='r')` 内存映射模式加载 .npy 文件，降低系统内存压力。

### 紧急恢复

训练异常中断后，可从 emergency checkpoint 恢复：

```bash
python scripts/train.py --data E:\dataset\CV\triair\data --model results/checkpoint_emergency.pth --epochs 15
```

---

## 项目结构

```
trimodal-uav-det/
├── scripts/                          # 入口脚本
│   ├── train.py                      # 主训练脚本（含监控 + OOM 保护）
│   ├── test.py                       # 测试评估脚本
│   ├── visualize.py                  # 可视化脚本
│   ├── monitor.py                    # 独立资源监控器
│   └── quick_test.py                 # 快速训练验证（5 batch）
├── trimodaldet/                      # 主包
│   ├── models/                       # 模型架构
│   │   ├── encoder.py                # InterModalBackbone (MiT 双分支)
│   │   ├── fusion.py                 # MAGE + BiTE 融合模块
│   │   ├── backbone.py               # FPN 包装器
│   │   └── transformer.py            # Transformer 基础模块
│   ├── training/                     # 训练与评估
│   │   ├── trainer.py                # 训练循环（含 OOM 保护、显存检查）
│   │   ├── evaluator.py              # 评估器
│   │   └── monitor_utils.py          # 监控工具（资源分析、优化建议）
│   ├── data/                         # 数据加载
│   │   ├── dataset.py                # NpyYoloDataset (五通道 .npy，内存映射)
│   │   └── transforms.py             # YOLO/COCO 格式转换
│   ├── utils/                        # 工具函数
│   │   ├── metrics.py                # mAP 评估指标
│   │   ├── timm_compat.py            # Timm 兼容性模块
│   │   └── visualization.py          # 可视化
│   ├── ablations/                    # 消融实验
│   │   ├── backbone_modality.py      # 模态可配置 Backbone
│   │   └── scripts/                  # 消融脚本
│   │       ├── train_modality_ablation.py  # 模态消融训练
│   │       └── test_modality_ablation.py   # 模态消融测试
│   └── config.py                     # 配置管理
├── archive/                          # 原始代码备份
├── requirements2.txt                 # 依赖清单
├── README.md                         # 英文 README
├── README_CN.md                      # 中文 README
├── CLAUDE.md                         # 项目速览
└── data/                             # 数据目录（用户创建）
    ├── images/
    └── labels/
```

---

## 训练详情

- **优化器**：SGD + Momentum（momentum=0.9，weight_decay=0.0001）
- **损失函数**：Faster R-CNN 多任务损失（分类 + RPN + 边界框回归）
- **数据划分**：自动 80/20 训练/测试划分
- **学习率调度**：Linear Warmup（500 steps）+ Cosine Annealing
- **显存优化**：内存映射加载、定期缓存清理、OOM 自动保护
- **设备**：自动检测 GPU（CUDA），否则回退到 CPU

## 许可证

MIT License
