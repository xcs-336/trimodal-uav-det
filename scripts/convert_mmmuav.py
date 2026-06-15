#!/usr/bin/env python
"""
Convert MMMUAV MOT-format dataset to TriModalDet-compatible format.

Input:
    MMMUAV/train/XXXX/{rgb_frame,ir_frame,event_frame,gt_rgb,gt_ir}/

Output:
    triair_mmmuav/images/XXXX_frameYYYYY_rgb.npy    (3-channel RGB)
    triair_mmmuav/images/XXXX_frameYYYYY_ir.npy     (1-channel IR, cropped to 360p)
    triair_mmmuav/images/XXXX_frameYYYYY_event.npy  (1-channel event, resized to 360×640)
    triair_mmmuav/images/XXXX_frameYYYYY_fused.npy  (5-channel: RGB+IR+Event, aligned)
    triair_mmmuav/labels_rgb/XXXX_frameYYYYY.txt    (YOLO format, per RGB annotation)
    triair_mmmuav/labels_ir/XXXX_frameYYYYY.txt     (YOLO format, per IR annotation)

Strategy:
    - RGB: keep as-is (360×640)
    - IR: center-crop from 512p to 360p (keeping same horizontal FOV)
    - Event: resize from native resolution to 360×640 (upsample)
    - Fused: concatenate aligned modalities as 5-channel .npy
    - Labels: only export annotated frames (sparse → ~50 frames/sequence)

Usage:
    python scripts/convert_mmmuav.py --input E:/dataset/CV/MMMUAV/MMMUAV/train --output E:/dataset/CV/triair_mmmuav
"""

import os
import sys
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm


def crop_ir_to_rgb(ir_img, rgb_h=360):
    """
    Crop IR image (512×640) to match RGB vertical FOV (360×640).

    Strategy: center-crop - removes top and bottom of IR image.
    IR has larger vertical FOV, so we take the middle 360 rows.
    """
    ir_h, ir_w = ir_img.shape[:2]
    top = (ir_h - rgb_h) // 2
    bottom = top + rgb_h
    if len(ir_img.shape) == 3:
        return ir_img[top:bottom, :, :]
    else:
        return ir_img[top:bottom, :]


def resize_event(event_img, target_h=360, target_w=640):
    """Resize event frame to match RGB resolution."""
    pil = Image.fromarray(event_img)
    pil = pil.resize((target_w, target_h), Image.BILINEAR)
    return np.array(pil)


def load_image(path):
    """Load image as numpy array. Returns None if not found."""
    if not os.path.exists(path):
        return None
    return np.array(Image.open(path))


def mot_to_yolo(mot_line, img_w, img_h):
    """
    Convert MOT format to YOLO format.

    MOT: frame, id, bb_left, bb_top, bb_width, bb_height, conf, class, vis
    YOLO: class_id x_center y_center width height (normalized 0-1)
    """
    parts = mot_line.strip().split(',')
    if len(parts) < 6:
        return None

    class_id = int(parts[7]) if len(parts) > 7 else 0
    if class_id != 0:  # Only class 0 = drone
        return None

    bb_left = float(parts[2])
    bb_top = float(parts[3])
    bb_width = float(parts[4])
    bb_height = float(parts[5])

    # Convert to YOLO normalized format
    x_center = (bb_left + bb_width / 2) / img_w
    y_center = (bb_top + bb_height / 2) / img_h
    w = bb_width / img_w
    h = bb_height / img_h

    return f"0 {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}\n"


def process_sequence(seq_dir, output_dir, modalities=('rgb', 'ir', 'event', 'fused')):
    """
    Process a single MMMUAV sequence.

    Args:
        seq_dir: Path to sequence directory (e.g., MMMUAV/train/0050)
        output_dir: Root output directory
        modalities: Tuple of output modalities to generate
    """
    seq_name = os.path.basename(seq_dir)

    # Read annotations
    rgb_gt_path = os.path.join(seq_dir, 'gt_rgb', 'gt.txt')
    ir_gt_path = os.path.join(seq_dir, 'gt_ir', 'gt.txt')

    rgb_annotations = {}
    if os.path.exists(rgb_gt_path):
        with open(rgb_gt_path) as f:
            for line in f:
                parts = line.strip().split(',')
                frame = int(parts[0])
                if frame not in rgb_annotations:
                    rgb_annotations[frame] = []
                rgb_annotations[frame].append(line.strip())

    ir_annotations = {}
    if os.path.exists(ir_gt_path):
        with open(ir_gt_path) as f:
            for line in f:
                parts = line.strip().split(',')
                frame = int(parts[0])
                if frame not in ir_annotations:
                    ir_annotations[frame] = []
                ir_annotations[frame].append(line.strip())

    # Get annotated frames (union of RGB and IR annotations)
    annotated_frames = sorted(set(list(rgb_annotations.keys()) + list(ir_annotations.keys())))

    rgb_count = 0
    ir_count = 0
    event_count = 0
    fused_count = 0

    for frame_num in annotated_frames:
        # Load images
        rgb = load_image(os.path.join(seq_dir, 'rgb_frame', f'{frame_num:04d}.jpg'))
        ir = load_image(os.path.join(seq_dir, 'ir_frame', f'{frame_num:04d}.jpg'))
        event = load_image(os.path.join(seq_dir, 'event_frame', f'{frame_num:04d}.jpg'))

        if rgb is None:
            continue

        img_id = f'{seq_name}_frame{frame_num:05d}'

        # Export per-modality images
        if 'rgb' in modalities:
            rgb_path = os.path.join(output_dir, 'images', f'{img_id}_rgb.npy')
            np.save(rgb_path, rgb)
            rgb_count += 1

        if 'ir' in modalities:
            if ir is not None:
                ir_cropped = crop_ir_to_rgb(ir, rgb_h=rgb.shape[0])
                # Take only 1 channel (grayscale → single channel)
                if len(ir_cropped.shape) == 3:
                    ir_cropped = ir_cropped[:, :, 0]
                ir_path = os.path.join(output_dir, 'images', f'{img_id}_ir.npy')
                np.save(ir_path, ir_cropped)
                ir_count += 1

        if 'event' in modalities:
            if event is not None:
                event_resized = resize_event(event, target_h=rgb.shape[0], target_w=rgb.shape[1])
                if len(event_resized.shape) == 3:
                    event_resized = event_resized[:, :, 0]
                event_path = os.path.join(output_dir, 'images', f'{img_id}_event.npy')
                np.save(event_path, event_resized)
                event_count += 1

        # Export fused 5-channel image
        if 'fused' in modalities:
            if ir is not None and event is not None:
                # RGB (3ch) + IR (1ch, cropped) + Event (1ch, resized)
                ir_cropped = crop_ir_to_rgb(ir, rgb_h=rgb.shape[0])
                if len(ir_cropped.shape) == 3:
                    ir_cropped = ir_cropped[:, :, 0:1]
                elif len(ir_cropped.shape) == 2:
                    ir_cropped = ir_cropped[:, :, np.newaxis]
                else:
                    ir_cropped = ir_cropped[:, :, 0:1]

                event_resized = resize_event(event, target_h=rgb.shape[0], target_w=rgb.shape[1])
                if len(event_resized.shape) == 3:
                    event_resized = event_resized[:, :, 0:1]
                elif len(event_resized.shape) == 2:
                    event_resized = event_resized[:, :, np.newaxis]
                else:
                    event_resized = event_resized[:, :, 0:1]

                fused = np.concatenate([rgb, ir_cropped, event_resized], axis=2)  # (H, W, 5)
                fused_path = os.path.join(output_dir, 'images', f'{img_id}_fused.npy')
                np.save(fused_path, fused)
                fused_count += 1

        # Export RGB labels (YOLO format)
        if frame_num in rgb_annotations:
            yolo_lines = []
            for mot_line in rgb_annotations[frame_num]:
                yolo_line = mot_to_yolo(mot_line, 640, 360)
                if yolo_line:
                    yolo_lines.append(yolo_line)
            if yolo_lines:
                label_path = os.path.join(output_dir, 'labels_rgb', f'{img_id}.txt')
                with open(label_path, 'w') as f:
                    f.writelines(yolo_lines)

        # Export IR labels (YOLO format - boxes are in IR 512p coordinates,
        # but IR image is center-cropped to 360p)
        # IR: 640×512 → crop center 360 rows → 640×360
        # top = (512 - 360) // 2 = 76, so y_offset = 76
        ir_crop_top = (512 - rgb.shape[0]) // 2 if ir is not None else 0
        if frame_num in ir_annotations:
            yolo_lines = []
            for mot_line in ir_annotations[frame_num]:
                parts = mot_line.strip().split(',')
                if len(parts) < 6:
                    continue
                # Adjust bb_top for center crop (subtract the top offset)
                bb_left = float(parts[2])
                bb_top = float(parts[3]) - ir_crop_top  # Adjust for crop
                bb_width = float(parts[4])
                bb_height = float(parts[5])
                class_id = int(parts[7]) if len(parts) > 7 else 0
                if class_id != 0:
                    continue
                # Convert to YOLO normalized format (using cropped dimensions 640×360)
                x_center = (bb_left + bb_width / 2) / 640
                y_center = (bb_top + bb_height / 2) / 360
                w = bb_width / 640
                h = bb_height / 360
                yolo_lines.append(f"0 {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}\n")
            if yolo_lines:
                label_path = os.path.join(output_dir, 'labels_ir', f'{img_id}.txt')
                with open(label_path, 'w') as f:
                    f.writelines(yolo_lines)

    return rgb_count, ir_count, event_count, fused_count


def main():
    parser = argparse.ArgumentParser(description='Convert MMMUAV to TriModalDet format')
    parser.add_argument('--input', type=str, required=True, help='Path to MMMUAV train directory')
    parser.add_argument('--output', type=str, required=True, help='Output root directory')
    parser.add_argument('--modalities', type=str, default='rgb,ir,event,fused',
                       help='Comma-separated modalities to export')
    args = parser.parse_args()

    modalities = tuple(m.strip() for m in args.modalities.split(','))

    # Create output directories
    os.makedirs(os.path.join(args.output, 'images'), exist_ok=True)
    os.makedirs(os.path.join(args.output, 'labels_rgb'), exist_ok=True)
    os.makedirs(os.path.join(args.output, 'labels_ir'), exist_ok=True)

    # Find all sequences
    seqs = sorted([d for d in os.listdir(args.input)
                   if os.path.isdir(os.path.join(args.input, d))])

    print(f'Found {len(seqs)} sequences')
    print(f'Output modalities: {modalities}')

    total_rgb = total_ir = total_event = total_fused = 0

    for seq in tqdm(seqs, desc='Converting sequences'):
        seq_dir = os.path.join(args.input, seq)
        try:
            rgb_n, ir_n, ev_n, fused_n = process_sequence(seq_dir, args.output, modalities)
            total_rgb += rgb_n
            total_ir += ir_n
            total_event += ev_n
            total_fused += fused_n
        except Exception as e:
            print(f'\nError processing {seq}: {e}')
            continue

    print(f'\nConversion complete:')
    if 'rgb' in modalities:
        print(f'  RGB images: {total_rgb}')
    if 'ir' in modalities:
        print(f'  IR images: {total_ir}')
    if 'event' in modalities:
        print(f'  Event images: {total_event}')
    if 'fused' in modalities:
        print(f'  Fused images (5-channel): {total_fused}')
    print(f'  Output directory: {args.output}')


if __name__ == '__main__':
    main()
