#!/usr/bin/env python -u
"""
Convert MM-UAV raw frames to TriModalDet .npy format.

Reads MM-UAV directory structure:
    MMMUAV/train/XXXX/
        rgb_frame/    *.jpg  (360x640)
        ir_frame/     *.jpg  (640x512)
        event_frame/  *.jpg  (native resolution)
        gt_rgb/gt.txt        (MOT format)
        gt_ir/gt.txt         (MOT format)

and MMMUAV/annotations/train-rgb.json (COCO format)

Produces:
    output/images/   XXXX_frameYYYYY.npy  (360x640x5 fused)
    output/labels/   XXXX_frameYYYYY.txt  (YOLO format)

Alignment is MINIMAL by design — only pixel-level matching:
    RGB:  kept as-is (360x640x3)
    IR:   center-crop 512->360, then optional resize 640x640->640x360
    Event: resize to 640x360
The STN module in the detection pipeline handles feature-level alignment.
"""

import argparse, json, os, sys
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

sys.stdout.reconfigure(line_buffering=True)


def resize_array(arr, target_h, target_w):
    """Resize single-channel array to target size."""
    pil = Image.fromarray(arr)
    return np.array(pil.resize((target_w, target_h), Image.BILINEAR), dtype=np.uint8)


def crop_ir_center(ir_arr, target_h=360):
    """Center-crop IR from 512x640 to target_h x 640."""
    h, w = ir_arr.shape[:2]
    top = (h - target_h) // 2
    return ir_arr[top:top + target_h, :]


def warp_with_homography(img, H, target_h, target_w):
    """Warp image using 3x3 homography to target resolution."""
    return cv2.warpPerspective(img, H, (target_w, target_h))


def mot_to_yolo(gt_path, img_w, img_h, crop_offset_y=0):
    """
    Convert MOT-format gt.txt to YOLO normalized labels.

    MOT: frame, id, bb_left, bb_top, bb_width, bb_height, conf, class, vis
    YOLO: class_id x_center y_center width height (normalized 0-1)

    crop_offset_y: vertical offset applied to IR crop (76px for 512->360)
    """
    if not os.path.exists(gt_path):
        return {}

    labels_by_frame = {}
    with open(gt_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line: continue
            parts = list(map(float, line.split(',')))
            if len(parts) < 8: continue
            frame_id = int(parts[0])
            # track_id = int(parts[1])
            bb_left, bb_top = parts[2], parts[3]
            bb_width, bb_height = parts[4], parts[5]
            class_id = int(parts[7]) if len(parts) > 7 else 0

            # Only class 0 (drone) — MM-UAV uses class_id=1
            if class_id not in (0, 1): continue

            # Adjust for IR crop offset
            bb_top -= crop_offset_y

            # Convert to YOLO normalized
            x_center = (bb_left + bb_width / 2) / img_w
            y_center = (bb_top + bb_height / 2) / img_h
            w_norm = bb_width / img_w
            h_norm = bb_height / img_h

            # Clamp
            x_center = max(0, min(1, x_center))
            y_center = max(0, min(1, y_center))
            w_norm = max(0, min(1, w_norm))
            h_norm = max(0, min(1, h_norm))

            if frame_id not in labels_by_frame:
                labels_by_frame[frame_id] = []
            labels_by_frame[frame_id].append(f"0 {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}")

    return labels_by_frame


def process_sequence(seq_dir, seq_name, out_img_dir, out_label_dir,
                     modalities=('rgb', 'ir', 'event'), rgb_h=360, rgb_w=640,
                     align='center_crop', H=None):
    """
    Process one MM-UAV sequence directory.
    Returns count of exported fused frames.

    align: 'center_crop' (default) | 'homography'
    H: 3x3 homography matrix (required when align='homography')
    """
    rgb_dir = os.path.join(seq_dir, 'rgb_frame')
    ir_dir = os.path.join(seq_dir, 'ir_frame')
    event_dir = os.path.join(seq_dir, 'event_frame')
    gt_rgb_file = os.path.join(seq_dir, 'gt_rgb', 'gt.txt')
    gt_ir_file = os.path.join(seq_dir, 'gt_ir', 'gt.txt')

    if not os.path.isdir(rgb_dir):
        print(f"Warning: no rgb_frame in {seq_dir}")
        return 0

    # Load annotations
    ir_crop_offset = 76 if align == 'center_crop' else 0
    rgb_labels = mot_to_yolo(gt_rgb_file, rgb_w, rgb_h, crop_offset_y=0)
    ir_labels = mot_to_yolo(gt_ir_file, rgb_w, rgb_h, crop_offset_y=ir_crop_offset)

    # For homography mode, pre-compute H_inv (XoFTR outputs RGB→IR, we need IR→RGB)
    H_inv = None
    if align == 'homography' and H is not None:
        H_inv = np.linalg.inv(H)

    # Get common frame IDs
    rgb_frames = sorted([f for f in os.listdir(rgb_dir)
                         if f.endswith(('.jpg', '.jpeg', '.png'))])
    exported = 0

    for fname in tqdm(rgb_frames, desc=seq_name, leave=False):
        frame_id_str = os.path.splitext(fname)[0]
        try:
            frame_id = int(frame_id_str)
        except ValueError:
            frame_id = frame_id_str

        out_name = f"{seq_name}_frame{frame_id_str}"

        # Load RGB
        rgb_path = os.path.join(rgb_dir, fname)
        rgb_img = np.array(Image.open(rgb_path).convert('RGB'))  # (H, W, 3)

        # Load IR
        ir_path = os.path.join(ir_dir, fname)
        if os.path.exists(ir_path):
            ir_img = np.array(Image.open(ir_path).convert('L'))  # (H, W)
            if align == 'homography' and H_inv is not None:
                ir_img = warp_with_homography(ir_img, H_inv, rgb_h, rgb_w)
            else:
                ir_img = crop_ir_center(ir_img, target_h=rgb_h)   # (rgb_h, W)
        else:
            ir_img = np.zeros((rgb_h, rgb_w), dtype=np.uint8)

        # Load Event
        ev_path = os.path.join(event_dir, fname)
        if os.path.exists(ev_path):
            ev_img = np.array(Image.open(ev_path).convert('L'))
            if align == 'homography' and H_inv is not None:
                ev_img = warp_with_homography(ev_img, H_inv, rgb_h, rgb_w)
            else:
                ev_img = resize_array(ev_img, rgb_h, rgb_w)        # (rgb_h, W)
        else:
            ev_img = np.zeros((rgb_h, rgb_w), dtype=np.uint8)

        # Build fused .npy
        fused = np.concatenate([
            rgb_img,
            ir_img[:, :, np.newaxis],
            ev_img[:, :, np.newaxis],
        ], axis=2)  # (H, W, 5)

        np.save(os.path.join(out_img_dir, out_name + '.npy'), fused)

        # Save YOLO labels
        labels = rgb_labels.get(frame_id, [])
        if not labels:
            labels = ir_labels.get(frame_id, [])

        if labels:
            with open(os.path.join(out_label_dir, out_name + '.txt'), 'w') as lf:
                lf.write('\n'.join(labels))
            exported += 1
        # else: frame without GT — still save .npy but no label file

    return exported


def main():
    parser = argparse.ArgumentParser(description='Convert MM-UAV to TRI .npy format')
    parser.add_argument('--mmuav-root', required=True,
                        help='Path to MMMUAV dataset root (contains train/, test/, annotations/)')
    parser.add_argument('--output-root', required=True,
                        help='Output root directory')
    parser.add_argument('--splits', nargs='+', default=['train', 'test'],
                        help='Which splits to convert')
    parser.add_argument('--max-seqs', type=int, default=0,
                        help='Limit number of sequences per split (0=all)')
    parser.add_argument('--align', choices=['center_crop', 'homography'],
                        default='center_crop',
                        help='IR/Event alignment method (default: center_crop)')
    parser.add_argument('--homography', type=str, default=None,
                        help='Path to 3x3 homography .npy file (required for --align homography)')
    args = parser.parse_args()

    # Load homography matrix if requested
    H = None
    if args.align == 'homography':
        if args.homography is None:
            parser.error('--align homography requires --homography PATH')
        H = np.load(args.homography)
        if H.shape != (3, 3):
            parser.error(f'Homography must be 3x3, got {H.shape}')
        print(f"Loaded homography from {args.homography}")
        print(f"H (RGB→IR):\n{H}")

    for split in args.splits:
        split_dir = os.path.join(args.mmuav_root, split)
        if not os.path.isdir(split_dir):
            print(f"Skip: {split_dir} not found")
            continue

        out_img_dir = os.path.join(args.output_root, split, 'images')
        out_label_dir = os.path.join(args.output_root, split, 'labels')
        os.makedirs(out_img_dir, exist_ok=True)
        os.makedirs(out_label_dir, exist_ok=True)

        seqs = sorted([d for d in os.listdir(split_dir)
                       if os.path.isdir(os.path.join(split_dir, d))])
        if args.max_seqs > 0:
            seqs = seqs[:args.max_seqs]

        total = 0
        for seq_name in tqdm(seqs, desc=f"Split {split}"):
            seq_path = os.path.join(split_dir, seq_name)
            n = process_sequence(seq_path, seq_name, out_img_dir, out_label_dir,
                                 align=args.align, H=H)
            total += n

        print(f"Split {split}: {len(seqs)} sequences, {total} annotated frames exported")
        print(f"  Images: {out_img_dir}")
        print(f"  Labels: {out_label_dir}")


if __name__ == '__main__':
    main()