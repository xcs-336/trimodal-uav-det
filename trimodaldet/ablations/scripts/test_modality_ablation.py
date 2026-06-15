#!/usr/bin/env python -u
"""
Standalone testing script for modality ablation experiments.

Loads a saved ablation checkpoint and runs evaluation on the test set.
Use this to test ablation models without re-running training.

Usage:
    python trimodaldet/ablations/scripts/test_modality_ablation.py \
        --data ../data/images \
        --labels ../data/labels \
        --backbone mit_b1 \
        --modalities rgb,thermal \
        --checkpoint results/ablation_no_event/model_best_weights.pth
"""

import sys
import os
import argparse
import json
from datetime import datetime

# Force unbuffered output
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
sys.path.insert(0, project_root)

import torch
from torch.utils.data import DataLoader
from torchvision.ops import MultiScaleRoIAlign
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection import FasterRCNN

from trimodaldet.config import Config, get_num_classes
from trimodaldet.data.dataset import NpyYoloDataset
from trimodaldet.ablations.backbone_modality import ModalityConfigurableBackbone
from trimodaldet.models.encoder import get_encoder


def build_model(config, active_modalities, device):
    """Reconstruct the same model architecture used in training."""
    print(f"Building model: backbone={config.backbone_type}, modalities={active_modalities}")

    encoder_base = get_encoder(
        backbone_name=config.backbone_type,
        in_chans_rgb=config.in_chans_rgb,
        in_chans_x=config.in_chans_x
    )

    backbone = ModalityConfigurableBackbone(
        encoder_base,
        fpn_out_channels=config.fpn_out_channels,
        active_modalities=active_modalities
    )
    print(f"  FPN output channels: {backbone.out_channels}")
    print(f"  Modality config: {backbone.get_modality_config()}")

    anchor_generator = AnchorGenerator(
        sizes=config.anchor_sizes,
        aspect_ratios=config.anchor_aspect_ratios
    )

    roi_pooler = MultiScaleRoIAlign(
        featmap_names=config.roi_featmap_names,
        output_size=config.roi_output_size,
        sampling_ratio=config.roi_sampling_ratio
    )

    model = FasterRCNN(
        backbone,
        num_classes=config.num_classes,
        rpn_anchor_generator=anchor_generator,
        box_roi_pool=roi_pooler,
        image_mean=config.image_mean,
        image_std=config.image_std
    )

    model.to(device)
    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"  Total parameters: {total_params:.2f}M")
    return model


def load_checkpoint(model, checkpoint_path, device):
    """Load model weights from checkpoint (supports multiple formats)."""
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    if isinstance(checkpoint, dict):
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            print("  Loaded from 'model_state_dict' key")
            # Print extra info if available
            if 'epoch' in checkpoint:
                print(f"  Trained epoch: {checkpoint['epoch']}")
            if 'loss' in checkpoint:
                print(f"  Training loss: {checkpoint['loss']:.4f}")
        elif 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
            print("  Loaded from 'state_dict' key")
        else:
            model.load_state_dict(checkpoint)
            print("  Loaded directly as state_dict")
    else:
        model.load_state_dict(checkpoint)
        print("  Loaded directly as state_dict")

    model.eval()
    print("  Model loaded successfully.")
    return model


def evaluate(model, test_loader, device):
    """Run evaluation on test set and return metrics."""
    from torchmetrics.detection.mean_ap import MeanAveragePrecision

    print("\nRunning evaluation on test set...")
    metric = MeanAveragePrecision(iou_type="bbox")
    model.eval()

    num_images = 0
    with torch.no_grad():
        for i, (images, targets) in enumerate(test_loader):
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            predictions = model(images)

            preds = []
            for pred in predictions:
                preds.append({
                    'boxes': pred['boxes'].cpu(),
                    'scores': pred['scores'].cpu(),
                    'labels': pred['labels'].cpu()
                })

            tgts = []
            for tgt in targets:
                tgts.append({
                    'boxes': tgt['boxes'].cpu(),
                    'labels': tgt['labels'].cpu()
                })

            metric.update(preds, tgts)
            num_images += len(images)

            if (i + 1) % 50 == 0:
                print(f"  Evaluated {num_images} images...")

    results = metric.compute()
    return {
        'mAP': results['map'].item(),
        'mAP_50': results['map_50'].item(),
        'mAP_75': results['map_75'].item(),
        'mAP_small': results['map_small'].item(),
        'mAP_medium': results['map_medium'].item(),
        'mAP_large': results['map_large'].item(),
        'mAR_1': results['mar_1'].item(),
        'mAR_10': results['mar_10'].item(),
        'mAR_100': results['mar_100'].item(),
        'mAR_small': results['mar_small'].item(),
        'mAR_medium': results['mar_medium'].item(),
        'mAR_large': results['mar_large'].item(),
        'num_test_images': num_images,
    }


def main():
    parser = argparse.ArgumentParser(description='Test Modality Ablation Model')

    # Dataset
    parser.add_argument('--data', type=str, required=True, help='Path to image data directory')
    parser.add_argument('--labels', type=str, required=True, help='Path to labels directory')

    # Model
    parser.add_argument('--backbone', type=str, default='mit_b1',
                        choices=['mit_b0', 'mit_b1', 'mit_b2', 'mit_b3', 'mit_b4'])
    parser.add_argument('--modalities', type=str, required=True,
                        help='Comma-separated modalities, e.g. "rgb,thermal"')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to checkpoint (.pth file)')

    # Output
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Directory to save results (default: derived from checkpoint path)')
    parser.add_argument('--batch-size', type=int, default=1, help='Batch size for testing')

    args = parser.parse_args()

    # Parse modalities
    active_modalities = [m.strip().lower() for m in args.modalities.split(',')]

    # Validate
    valid_modalities = {'rgb', 'thermal', 'event'}
    for m in active_modalities:
        if m not in valid_modalities:
            raise ValueError(f"Invalid modality '{m}'. Must be one of {valid_modalities}")

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"Modalities: {active_modalities}")
    print(f"Backbone: {args.backbone}")

    # Build config
    config = Config()
    config.backbone_type = args.backbone
    config.batch_size = args.batch_size

    # Detect num_classes from labels
    num_classes = get_num_classes(args.labels)
    config.num_classes = num_classes + 1  # +1 for background
    print(f"Detected {num_classes} classes ({config.num_classes} including background)")

    # Build model
    model = build_model(config, active_modalities, device)

    # Load checkpoint
    model = load_checkpoint(model, args.checkpoint, device)

    # Create test dataset
    test_dataset = NpyYoloDataset(
        image_dir=args.data,
        label_dir=args.labels,
        mode='test'
    )
    print(f"Test dataset: {len(test_dataset)} images")

    def collate_fn(batch):
        return tuple(zip(*batch))

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn
    )

    # Run evaluation
    eval_results = evaluate(model, test_loader, device)

    # Print results
    print("\n" + "=" * 50)
    print("EVALUATION RESULTS")
    print("=" * 50)
    print(f"  Modalities: {'+'.join(active_modalities)}")
    print(f"  Backbone: {args.backbone}")
    print(f"  Checkpoint: {args.checkpoint}")
    print(f"  Test images: {eval_results['num_test_images']}")
    print("-" * 50)
    print(f"  mAP:        {eval_results['mAP']:.4f}")
    print(f"  mAP@50:     {eval_results['mAP_50']:.4f}")
    print(f"  mAP@75:     {eval_results['mAP_75']:.4f}")
    print(f"  mAP_small:  {eval_results['mAP_small']:.4f}")
    print(f"  mAP_medium: {eval_results['mAP_medium']:.4f}")
    print(f"  mAP_large:  {eval_results['mAP_large']:.4f}")
    print("-" * 50)
    print(f"  mAR@1:      {eval_results['mAR_1']:.4f}")
    print(f"  mAR@10:     {eval_results['mAR_10']:.4f}")
    print(f"  mAR@100:    {eval_results['mAR_100']:.4f}")
    print(f"  mAR_small:  {eval_results['mAR_small']:.4f}")
    print(f"  mAR_medium: {eval_results['mAR_medium']:.4f}")
    print(f"  mAR_large:  {eval_results['mAR_large']:.4f}")
    print("=" * 50)

    # Save results
    output_dir = args.output_dir
    if output_dir is None:
        output_dir = os.path.dirname(args.checkpoint)
    os.makedirs(output_dir, exist_ok=True)

    results = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'config': {
            'backbone': args.backbone,
            'modalities': active_modalities,
            'modality_config': '+'.join(sorted(active_modalities)),
            'checkpoint': args.checkpoint,
        },
        'evaluation': eval_results,
    }

    results_path = os.path.join(output_dir, 'test_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {results_path}")


if __name__ == '__main__':
    main()
