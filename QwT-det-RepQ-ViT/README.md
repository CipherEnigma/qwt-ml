# Quantization without Tears

This repository extends [mmdetection](https://github.com/open-mmlab/mmdetection) and [RepQ-ViT-Detection](https://github.com/zkkli/RepQ-ViT/tree/main/detection) to reproduce QwT’s object-detection results on COCO.

## Prerequisites

- **Python**: 3.10 or 3.11 (Python 3.9 and below are not supported)
- **PyTorch**: 1.13.0 or higher
- **CUDA**: 11.0 or higher (optional, for GPU support)

## Installation

### Step 1: Create a Python Environment

We recommend using conda to create a clean environment:

```bash
conda create -n qwt python=3.10
conda activate qwt
```

Or with Python 3.11:

```bash
conda create -n qwt python=3.11
conda activate qwt
```

### Step 2: Install PyTorch

Install PyTorch 1.13.0 or higher. Visit [PyTorch's official website](https://pytorch.org/get-started/locally/) for installation instructions specific to your system.

Example for CUDA 11.8:
```bash
pip install torch>=1.13.0 torchvision>=0.14.0 --index-url https://download.pytorch.org/whl/cu118
```

For CPU-only:
```bash
pip install torch>=1.13.0 torchvision>=0.14.0
```

### Step 3: Install MMCV via MIM

```bash
pip install -U openmim
mim install "mmcv>=2.0.0,<2.2.0"
```

**Note**: This project requires mmcv 2.0.0 or higher. The older `mmcv-full` package is not compatible with Python 3.10+.

### Step 4: Install MMDetection

```bash
cd QwT/detection
pip install -v -e .
```

### Step 5: Setup Dataset and Checkpoints

Get the checkpoints from the [Swin-Transformer-Object-Detection](https://github.com/SwinTransformer/Swin-Transformer-Object-Detection) repo and place them in your *pretrained_weights* directory.

| Backbone | Method | Lr Schd | box mAP | mask mAP | #params | FLOPs | config | model |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Swin-T | Mask R-CNN         | 3x | 46.0 | 41.6 | 48M | 267G | [config](configs/swin/mask_rcnn_swin_tiny_patch4_window7_mstrain_480-800_adamw_3x_coco.py) | [github](https://github.com/SwinTransformer/storage/releases/download/v1.0.2/mask_rcnn_swin_tiny_patch4_window7.pth)/[baidu](https://pan.baidu.com/s/1YpauXYAFOohyMi3Vkb6DBg) |
| Swin-S | Mask R-CNN         | 3x | 48.5 | 43.3 | 69M | 359G | [config](configs/swin/mask_rcnn_swin_small_patch4_window7_mstrain_480-800_adamw_3x_coco.py) | [github](https://github.com/SwinTransformer/storage/releases/download/v1.0.2/mask_rcnn_swin_small_patch4_window7.pth)/[baidu](https://pan.baidu.com/s/1V4w4aaV7HSjXNFTOSA6v6w) |
| Swin-T | Cascade Mask R-CNN | 3x | 50.4 | 43.7 | 86M | 745G | [config](configs/swin/cascade_mask_rcnn_swin_tiny_patch4_window7_mstrain_480-800_giou_4conv1f_adamw_3x_coco.py) | [github](https://github.com/SwinTransformer/storage/releases/download/v1.0.2/cascade_mask_rcnn_swin_tiny_patch4_window7.pth)/[baidu](https://pan.baidu.com/s/1i-izBrODgQmMwTv6F6-x3A) |
| Swin-S | Cascade Mask R-CNN | 3x | 51.9 | 45.0 | 107M | 838G | [config](configs/swin/cascade_mask_rcnn_swin_small_patch4_window7_mstrain_480-800_giou_4conv1f_adamw_3x_coco.py) | [github](https://github.com/SwinTransformer/storage/releases/download/v1.0.2/cascade_mask_rcnn_swin_small_patch4_window7.pth)/[baidu](https://pan.baidu.com/s/1Sv9-gP1Qpl6SGOF6DBhUbw) |
| Swin-B | Cascade Mask R-CNN | 3x | 51.9 | 45.0 | 145M | 982G | [config](configs/swin/cascade_mask_rcnn_swin_base_patch4_window7_mstrain_480-800_giou_4conv1f_adamw_3x_coco.py) | [github](https://github.com/SwinTransformer/storage/releases/download/v1.0.2/cascade_mask_rcnn_swin_base_patch4_window7.pth)/[baidu](https://pan.baidu.com/s/1tHoC9PMVnldQUAfcF6FT3A) |

Access code for `baidu` is `swin`.

Update the **data_root** setting in `configs/_base_/datasets/coco_instance.py` to point to the root directory of your [COCO](https://cocodataset.org/#home) dataset.


## Evaluation

- You can quantize and evaluate model using the following command:

```bash
CUDA_VISIBLE_DEVICES=<YOUR GPU IDs> tools/dist_test.sh <CONFIG_FILE> <DET_CHECKPOINT_FILE> <\#GPU USED> --eval bbox segm [--w_bits] [--a_bits]

Required arguments:
 <CONFIG_FILE> : Path to config. You can find it at ./configs/swin/
 <DET_CHECKPOINT_FILE> : Path to checkpoint of pre-trained models.

optional arguments:
--w_bit: Bit-precision of weights, default=4.
--a_bit: Bit-precision of activation, default=4.
```

- Example: Quantize *Cascade Mask R-CNN with Swin-T* at W4/A4 precision:

```bash
# single GPUs
CUDA_VISIBLE_DEVICES=0 bash tools/dist_test.sh configs/swin/cascade_mask_rcnn_swin_tiny_patch4_window7_mstrain_480-800_giou_4conv1f_adamw_3x_coco.py pretrained_weights/cascade_mask_rcnn_swin_tiny_patch4_window7.pth 1  --eval bbox segm --w_bit 4 --a_bits 4

# 4 GPUs
CUDA_VISIBLE_DEVICES=0,1,2,3 bash tools/dist_test.sh configs/swin/cascade_mask_rcnn_swin_tiny_patch4_window7_mstrain_480-800_giou_4conv1f_adamw_3x_coco.py pretrained_weights/cascade_mask_rcnn_swin_tiny_patch4_window7.pth 4  --eval bbox segm --w_bit 4 --a_bits 4
```

### Results

Below are the experimental results obtained on the COCO dataset using QwT.

| Model                                     | Method           | Prec. | AP<sup>box</sup> / AP<sup>mask</sup> | Prec. | AP<sup>box</sup> / AP<sup>mask</sup> |
|:-----------------------------------------:|:----------------:|:-----:|:------------------------------------:|:-----:|:------------------------------------:|
| Mask RCNN + Swin_T (46.0 / 41.6)          |  RepQ-ViT        | W4/A4 | 36.1 / 36.0                          | W6/A6 | 45.5 / 41.3                          |  
|                                           |  RepQ-ViT + QwT  | W4/A4 | 36.3 / 36.0                          | W6/A6 | 45.4 / 41.3                          |
| Mask RCNN + Swin_S (48.5 / 43.3)          |  RepQ-ViT        | W4/A4 | 42.6 / 40.0                          | W6/A6 | 47.6 / 42.9                          |
|                                           |  RepQ-ViT + QwT  | W4/A4 | 43.1 / 40.4                          | W6/A6 | 48.0 / 43.1                          |
| Cascade Mask RCNN + Swin_T (50.4 / 43.7)  |  RepQ-ViT        | W4/A4 | 47.0 / 41.4                          | W6/A6 | 50.1 / 43.5                          |
|                                           |  RepQ-ViT + QwT  | W4/A4 | 47.6 / 41.8                          | W6/A6 | 50.1 / 43.6                          |
| Cascade Mask RCNN + Swin_S (51.9 / 45.0)  |  RepQ-ViT        | W4/A4 | 49.3 / 43.1                          | W6/A6 | 51.4 / 44.6                          |
|                                           |  RepQ-ViT + QwT  | W4/A4 | 49.9 / 43.4                          | W6/A6 | 51.7 / 44.8                          |
| Cascade Mask RCNN + Swin_B (51.9 / 45.0)  |  RepQ-ViT        | W4/A4 | 49.3 / 43.1                          | W6/A6 | 51.5 / 44.8                          |
|                                           |  RepQ-ViT + QwT  | W4/A4 | 50.0 / 43.7                          | W6/A6 | 51.8 / 45.0                          |

## Troubleshooting

### Common Installation Issues

#### Issue: "Python version not supported"
**Solution**: Ensure you're using Python 3.10 or 3.11. Check your version with:
```bash
python --version
```

#### Issue: "mmcv version mismatch" or "mmcv-full not found"
**Solution**: This project requires mmcv 2.0.0 or higher. Uninstall old versions and reinstall:
```bash
pip uninstall mmcv mmcv-full
pip install -U openmim
mim install "mmcv>=2.0.0,<2.2.0"
```

#### Issue: "torch not found" during installation
**Solution**: Install PyTorch before installing mmdet:
```bash
pip install torch>=1.13.0 torchvision>=0.14.0
```

#### Issue: CUDA extension compilation fails
**Solution**: If you don't need CUDA support, you can install without building extensions:
```bash
pip install -v -e . --no-build-isolation
```

For CUDA support, ensure you have:
- Compatible CUDA toolkit installed (11.0+)
- Matching PyTorch CUDA version
- C++ compiler (gcc/g++ on Linux, MSVC on Windows)

#### Issue: "ImportError: cannot import name 'Iterable' from 'collections'"
**Solution**: This error indicates you're using an incompatible version of a dependency. Ensure all dependencies are updated:
```bash
pip install --upgrade -r requirements.txt
```

#### Issue: Model evaluation produces different results than expected
**Solution**: Ensure you're using the correct checkpoint files and that your COCO dataset is properly configured. Check that the `data_root` in `configs/_base_/datasets/coco_instance.py` points to the correct location.

### Getting Help

If you encounter issues not covered here:
1. Check the [mmdetection documentation](https://mmdetection.readthedocs.io/)
2. Review the [mmcv documentation](https://mmcv.readthedocs.io/)
3. Open an issue on the GitHub repository with:
   - Your Python version (`python --version`)
   - Your PyTorch version (`python -c "import torch; print(torch.__version__)"`)
   - Your mmcv version (`python -c "import mmcv; print(mmcv.__version__)"`)
   - Full error message and stack trace

## Migration from Older Versions

If you're upgrading from a previous version that used Python 3.6-3.9 and mmcv 1.x, please see [MIGRATION.md](MIGRATION.md) for detailed upgrade instructions.

## Citation

We would greatly appreciate it if you could cite our paper if you find our implementation helpful in your work.

```bash
@InProceedings{Fu_2025_CVPR,
    author    = {Fu, Minghao and Yu, Hao and Shao, Jie and Zhou, Junjie and Zhu, Ke and Wu, Jianxin},
    title     = {Quantization without Tears},
    booktitle = {Proceedings of the Computer Vision and Pattern Recognition Conference (CVPR)},
    month     = {June},
    year      = {2025},
    pages     = {4462-4472}
}
```
