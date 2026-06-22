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
- 支持 5 种变体（mit_b0 ~ mit_b4）

**2. 三模态融合模块**
- **MAGE（Modality-Aware Gated Exchange）**：通道门控 + 空间门控，自适应加权各模态贡献
- **BiTE（Bidirectional Token Exchange）**：跨模态注意力，实现双向信息流动
- 阶段级融合：在每个 Transformer Stage 后进行融合

**3. 检测头**
- **FPN（Feature Pyramid Network）**：多尺度特征聚合
- **Faster R-CNN**：分类 + 定位（RPN 提案 + RoI Align + 分类/回归）

---

## 数据格式

### 输入图像

图像为 `.npy` 格式的 NumPy 数组，shape 为 `(H, W, 5)`：
- **通道 0-2**：RGB
- **通道 3**：热红外（Thermal）
- **通道 4**：事件相机（Event）

### 标注格式

使用 YOLO 格式（每行一个目标）：`class_id x_center y_center width height`，所有坐标归一化到 `[0, 1]`。

### 目录结构

```
data/
├── images/    (.npy 文件)
└── labels/    (.txt 文件)
```

---

## 环境配置

- **Python**：3.12
- **PyTorch**：2.9.1 + CUDA 12.8
- **GPU**：RTX 5080（Blackwell 架构，16GB 显存）
- **Conda 环境**：`triair`

```bash
conda create -n triair python=3.12 -y && conda activate triair

# PyTorch CUDA 12.8
pip install torch==2.9.1+cu128 torchvision==0.24.1+cu128 torchaudio==2.9.1+cu128 \
    --index-url https://download.pytorch.org/whl/cu128

# 其余依赖（或直接 pip install -r requirements2.txt）
pip install timm matplotlib scikit-learn torchmetrics pycocotools
```

---

## 使用指南

### 一、主模型（全模态：RGB + Thermal + Event）

#### 训练

```bash
# RTX 5080 16GB 推荐（batch_size 取决于 backbone，mit_b1 建议 1-2）
python scripts/train.py --data E:\dataset\CV\triair\data --epochs 15 --batch-size 1

# 显存充足时可使用梯度累积 + 混合精度
python scripts/train.py --data E:\dataset\CV\triair\data \
    --epochs 15 --batch-size 1 --backbone mit_b1 \
    --grad-accumulation-steps 4 --use-amp

# 快速原型（mit_b0 更轻量）
python scripts/train.py --data E:\dataset\CV\triair\data \
    --epochs 15 --backbone mit_b0 --batch-size 4
```

训练过程中自动监控 GPU 显存/利用率、CPU、内存，OOM 时自动保存 emergency checkpoint。

#### 测试

```bash
python scripts/test.py --data E:\dataset\CV\triair\data --model trimodaldet.pth
```

---

### 二、消融实验

用于分析不同模态组合对检测性能的贡献。训练、测试脚本独立于主模型。

#### 训练

```bash
# 全模态基线
python trimodaldet/ablations/scripts/train_modality_ablation.py \
    --data E:\dataset\CV\triair\data\images \
    --labels E:\dataset\CV\triair\data\labels \
    --epochs 15 --batch-size 1 --backbone mit_b1 \
    --modalities rgb,thermal,event \
    --output-dir results/ablation_full

# 无事件相机
python trimodaldet/ablations/scripts/train_modality_ablation.py \
    --data E:\dataset\CV\triair\data\images \
    --labels E:\dataset\CV\triair\data\labels \
    --epochs 15 --batch-size 1 --backbone mit_b1 \
    --modalities rgb,thermal \
    --output-dir results/ablation_no_event

# 无热红外
python trimodaldet/ablations/scripts/train_modality_ablation.py \
    --data E:\dataset\CV\triair\data\images \
    --labels E:\dataset\CV\triair\data\labels \
    --epochs 15 --batch-size 1 --backbone mit_b1 \
    --modalities rgb,event \
    --output-dir results/ablation_no_thermal

# 仅 RGB（单模态基线）
python trimodaldet/ablations/scripts/train_modality_ablation.py \
    --data E:\dataset\CV\triair\data\images \
    --labels E:\dataset\CV\triair\data\labels \
    --epochs 15 --batch-size 1 --backbone mit_b1 \
    --modalities rgb \
    --output-dir results/ablation_rgb_only
```

#### 参数

| 参数 | 必填 | 默认 | 说明 |
|------|------|------|------|
| `--data` | 是 | - | 图像目录 |
| `--labels` | 是 | - | 标签目录 |
| `--output-dir` | 是 | - | 输出目录 |
| `--modalities` | 否 | `rgb,thermal,event` | 模态组合（逗号分隔） |
| `--backbone` | 否 | `mit_b1` | mit_b0 ~ mit_b4 |
| `--epochs` | 否 | `15` | 训练轮数 |
| `--batch-size` | 否 | `16` | RTX 5080 建议 1 |
| `--lr` | 否 | `0.02` | 学习率 |

#### 输出文件

```
results/ablation_no_event/
├── training.log / config.json / model_info.json
├── metrics_per_epoch.csv / metrics_per_batch.csv
├── final_results.json / evaluation_history.json
├── checkpoint_best.pth / model_best_weights.pth
├── checkpoint_latest.pth / checkpoint_epoch_N.pth
```

#### 独立测试（不重新训练）

训练过程中每 5 个 epoch 自动评估一次。如需单独测试已保存的 checkpoint：

```bash
python trimodaldet/ablations/scripts/test_modality_ablation.py \
    --data E:\dataset\CV\triair\data\images \
    --labels E:\dataset\CV\triair\data\labels \
    --backbone mit_b1 --modalities rgb,thermal \
    --checkpoint results/ablation_no_event/model_best_weights.pth
```

这会重建与训练时相同的模型架构、加载权重、计算全部 12 项 mAP/mAR 指标。

#### 对比多个消融结果

```bash
python -c "
import json
for exp in ['ablation_full', 'ablation_no_event', 'ablation_no_thermal', 'ablation_rgb_only']:
    r = json.load(open(f'results/{exp}/final_results.json'))
    print(f\"{r['config']['modality_config']:20s} mAP={r['final_test_results']['mAP']:.4f}  mAP@50={r['final_test_results']['mAP_50']:.4f}\")
"
```

---

### 三、可视化

```bash
python scripts/visualize.py --vis 0 --data E:\dataset\CV\triair\data
```

---

### 四、快速验证

```bash
python scripts/quick_test.py --batches 5 --batch-size 1
```

只跑 5 个 batch 验证环境和数据是否正常，附带资源监控。

---

## 配置参数

### 训练

| CLI 参数 | 默认值 | 说明 |
|---------|--------|------|
| `--data` | `data` | 数据根目录（含 images/ 和 labels/） |
| `--model` | `trimodaldet.pth` | 模型保存/加载路径 |
| `--epochs` | `15` | 训练轮数 |
| `--batch-size` | `16` | 批次大小（RTX 5080 mit_b1 建议 1-2） |
| `--lr` | `0.02` | 学习率 |
| `--backbone` | `mit_b1` | mit_b0 / mit_b1 / mit_b2 / mit_b3 / mit_b4 |
| `--grad-accumulation-steps` | `1` | 梯度累积步数（1=禁用，4=等效 batch_size×4） |
| `--use-amp` | `False` | 启用混合精度训练（可节省 40-50% 显存） |
| `--results-dir` | `test_results` | 结果输出目录 |

### 监控

| CLI 参数 | 默认值 | 说明 |
|---------|--------|------|
| `--monitor-interval` | `5.0` | 资源采样间隔（秒） |
| `--monitor-output` | `results/monitor_log.json` | 监控日志路径 |
| `--max-gpu-mem-pct` | `85.0` | 显存警告阈值（百分比） |
| `--enable-oom-protection` | `True` | OOM 自动保护 |
| `--check-interval-batches` | `50` | 显存检查间隔（batch 数） |

### 骨干网络

| 变体 | 参数量 | 适用场景 |
|------|--------|----------|
| mit_b0 | ~3.7M | 快速原型、显存受限 |
| mit_b1 | ~13.5M | 默认均衡 |
| mit_b2 | ~24.7M | 更高精度 |
| mit_b3 | ~44M | 中大模型 |
| mit_b4 | ~61.4M | 最大精度 |

---

## 项目结构

```
trimodal-uav-det/
├── scripts/
│   ├── train.py / test.py / visualize.py
│   ├── monitor.py / quick_test.py
├── trimodaldet/
│   ├── models/          # encoder.py, fusion.py, backbone.py, transformer.py
│   ├── training/        # trainer.py, evaluator.py, monitor_utils.py
│   ├── data/            # dataset.py, transforms.py
│   ├── utils/           # metrics.py, timm_compat.py, visualization.py
│   ├── ablations/       # backbone_modality.py
│   │   └── scripts/     # train_modality_ablation.py, test_modality_ablation.py
│   └── config.py
├── archive/             # 原始代码备份
├── requirements2.txt
├── README_CN.md / README.md / CLAUDE.md
└── data/
    ├── images/ (.npy)
    └── labels/ (.txt)
```

---

## 训练详情

- **优化器**：SGD + Momentum（0.9），weight_decay=0.0001
- **损失函数**：Faster R-CNN 多任务损失（分类 + RPN + 边界框回归）
- **学习率调度**：Linear Warmup（500 steps）+ Cosine Annealing
- **数据划分**：80/20 训练/测试
- **显存优化**：内存映射加载（`np.load(mmap_mode='r')`）、定期缓存清理、OOM 保护
- **设备**：自动检测 CUDA GPU，否则回退 CPU

## 许可证

MIT License
