# QwT Detection - Quick Start Guide

## Setup (5 minutes)

```bash
# Install dependencies
pip install -U openmim
mim install mmcv-full
pip install -v -e .
pip install tqdm scipy
```

## Download Checkpoint

```bash
mkdir -p ../pretrained_weights
cd ../pretrained_weights
wget https://github.com/SwinTransformer/storage/releases/download/v1.0.2/cascade_mask_rcnn_swin_tiny_patch4_window7.pth
cd ../QwT-det-RepQ-ViT
```

## Download COCO Dataset

Place COCO 2017 at your preferred location, then update:
`configs/_base_/datasets/coco_instance.py`

Change line 2:
```python
data_root = '/path/to/your/coco/'
```

## Run Experiment

```bash
# W4/A4 quantization
CUDA_VISIBLE_DEVICES=0 python tools/test.py \
    configs/swin/cascade_mask_rcnn_swin_tiny_patch4_window7_mstrain_480-800_giou_4conv1f_adamw_3x_coco.py \
    ../pretrained_weights/cascade_mask_rcnn_swin_tiny_patch4_window7.pth \
    --eval bbox segm \
    --w_bit 4 \
    --a_bits 4
```

## Expected Output

You'll see TWO sets of results:

**RepQ-ViT Baseline:**
```
Average Precision (AP) @[ IoU=0.50:0.95 ] = 0.470 (bbox)
Average Precision (AP) @[ IoU=0.50:0.95 ] = 0.414 (segm)
```

**RepQ-ViT + QwT:**
```
Average Precision (AP) @[ IoU=0.50:0.95 ] = 0.476 (bbox)  ← +0.6% improvement
Average Precision (AP) @[ IoU=0.50:0.95 ] = 0.418 (segm)  ← +0.4% improvement
```

## Runtime

- Total: ~35-40 minutes on single GPU
- Breakdown:
  - Loading & calibration: ~5 min
  - Baseline evaluation: ~15 min
  - QwT compensation: ~5 min
  - Final evaluation: ~15 min

## Try Different Precisions

```bash
# W6/A6 (better accuracy)
--w_bit 6 --a_bits 6

# W8/A8 (near full precision)
--w_bit 8 --a_bits 8
```
