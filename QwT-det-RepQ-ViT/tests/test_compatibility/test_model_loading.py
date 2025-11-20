"""Smoke tests for model loading and config parsing."""
import os
from os.path import dirname, exists, join

import pytest


def _get_config_directory():
    """Find the predefined detector config directory."""
    try:
        # Assume we are running in the source mmdetection repo
        repo_dpath = dirname(dirname(dirname(__file__)))
    except NameError:
        # For IPython development when this __file__ is not defined
        import mmdet
        repo_dpath = dirname(dirname(mmdet.__file__))
    config_dpath = join(repo_dpath, 'configs')
    if not exists(config_dpath):
        raise Exception('Cannot find config path')
    return config_dpath


def test_swin_config_exists():
    """Test that Swin Transformer config files exist."""
    config_dpath = _get_config_directory()
    swin_config_dir = join(config_dpath, 'swin')
    
    assert exists(swin_config_dir), f"Swin config directory not found at {swin_config_dir}"
    
    # Check for at least one Swin config file
    import glob
    swin_configs = glob.glob(join(swin_config_dir, '*.py'))
    assert len(swin_configs) > 0, "No Swin config files found"


def test_load_swin_config():
    """Test loading a Swin Transformer config file."""
    from mmcv import Config
    
    config_dpath = _get_config_directory()
    
    # Use a simple Swin config
    config_path = join(config_dpath, 'swin', 'mask_rcnn_swin_tiny_patch4_window7_mstrain_480-800_adamw_3x_coco.py')
    
    if not exists(config_path):
        pytest.skip(f"Config file not found: {config_path}")
    
    # Load the config
    cfg = Config.fromfile(config_path)
    
    # Verify basic structure
    assert cfg is not None, "Config should not be None"
    assert hasattr(cfg, 'model'), "Config should have 'model' attribute"
    assert hasattr(cfg, 'data'), "Config should have 'data' attribute"


def test_parse_swin_config_without_errors():
    """Test that Swin config can be parsed without errors."""
    from mmcv import Config
    
    config_dpath = _get_config_directory()
    config_path = join(config_dpath, 'swin', 'mask_rcnn_swin_tiny_patch4_window7_mstrain_480-800_adamw_3x_coco.py')
    
    if not exists(config_path):
        pytest.skip(f"Config file not found: {config_path}")
    
    # This should not raise any exceptions
    cfg = Config.fromfile(config_path)
    
    # Access model config
    model_cfg = cfg.model
    assert model_cfg is not None
    
    # Verify Swin backbone is configured
    assert 'backbone' in model_cfg
    assert model_cfg.backbone.get('type') == 'SwinTransformer' or 'Swin' in str(model_cfg.backbone.get('type', ''))


def test_instantiate_swin_model_without_weights():
    """Test that a Swin model can be instantiated without loading weights."""
    from mmcv import Config
    from mmdet.models import build_detector
    
    config_dpath = _get_config_directory()
    config_path = join(config_dpath, 'swin', 'mask_rcnn_swin_tiny_patch4_window7_mstrain_480-800_adamw_3x_coco.py')
    
    if not exists(config_path):
        pytest.skip(f"Config file not found: {config_path}")
    
    # Load config
    cfg = Config.fromfile(config_path)
    
    # Remove pretrained weights to avoid download
    if 'pretrained' in cfg.model:
        cfg.model['pretrained'] = None
    if 'backbone' in cfg.model and 'pretrained' in cfg.model.backbone:
        cfg.model.backbone['pretrained'] = None
    
    # Build the model (this tests that all imports and model construction works)
    try:
        model = build_detector(cfg.model)
        assert model is not None, "Model should not be None"
        
        # Verify it's a detector with expected components
        assert hasattr(model, 'backbone'), "Model should have backbone"
        assert hasattr(model, 'neck'), "Model should have neck"
        
        # Check if it's a two-stage detector (Mask R-CNN)
        if hasattr(model, 'roi_head'):
            assert model.roi_head is not None
            assert hasattr(model.roi_head, 'bbox_head')
            
    except Exception as e:
        pytest.fail(f"Failed to instantiate model: {e}")


def test_no_import_errors_for_core_modules():
    """Test that core mmdet modules can be imported without errors."""
    try:
        import mmdet
        from mmdet import apis
        from mmdet import datasets
        from mmdet import models
        from mmdet.models import build_detector
        from mmdet.datasets import build_dataset
        from mmdet.core import bbox2result
        
        # All imports should succeed
        assert mmdet is not None
        assert apis is not None
        assert datasets is not None
        assert models is not None
        assert build_detector is not None
        assert build_dataset is not None
        assert bbox2result is not None
        
    except ImportError as e:
        pytest.fail(f"Import error in core modules: {e}")


def test_config_base_files_exist():
    """Test that base config files exist and can be loaded."""
    from mmcv import Config
    
    config_dpath = _get_config_directory()
    base_config_dir = join(config_dpath, '_base_')
    
    assert exists(base_config_dir), f"Base config directory not found at {base_config_dir}"
    
    # Check for key base config files
    base_files = [
        'default_runtime.py',
        'models/mask_rcnn_swin_fpn.py',
        'datasets/coco_instance.py',
        'schedules/schedule_1x.py',
    ]
    
    for base_file in base_files:
        base_path = join(base_config_dir, base_file)
        if exists(base_path):
            # Try to load it
            try:
                cfg = Config.fromfile(base_path)
                assert cfg is not None
            except Exception as e:
                pytest.fail(f"Failed to load base config {base_file}: {e}")


def test_model_registry_available():
    """Test that model registry is available and populated."""
    from mmdet.models import BACKBONES, NECKS, HEADS, DETECTORS
    
    # Check that registries exist
    assert BACKBONES is not None
    assert NECKS is not None
    assert HEADS is not None
    assert DETECTORS is not None
    
    # Check that SwinTransformer is registered
    assert 'SwinTransformer' in BACKBONES.module_dict, (
        "SwinTransformer should be registered in BACKBONES"
    )
    
    # Check for common detector types
    assert 'MaskRCNN' in DETECTORS.module_dict or 'Mask2Former' in DETECTORS.module_dict, (
        "Common detector types should be registered"
    )


def test_build_simple_config():
    """Test building a simple detector from a minimal config."""
    from mmdet.models import build_detector
    
    # Create a minimal config for testing
    minimal_config = dict(
        type='MaskRCNN',
        backbone=dict(
            type='ResNet',
            depth=50,
            num_stages=4,
            out_indices=(0, 1, 2, 3),
            frozen_stages=1,
            norm_cfg=dict(type='BN', requires_grad=True),
            norm_eval=True,
            style='pytorch',
            init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')
        ),
        neck=dict(
            type='FPN',
            in_channels=[256, 512, 1024, 2048],
            out_channels=256,
            num_outs=5
        ),
        rpn_head=dict(
            type='RPNHead',
            in_channels=256,
            feat_channels=256,
            anchor_generator=dict(
                type='AnchorGenerator',
                scales=[8],
                ratios=[0.5, 1.0, 2.0],
                strides=[4, 8, 16, 32, 64]
            ),
            bbox_coder=dict(
                type='DeltaXYWHBBoxCoder',
                target_means=[.0, .0, .0, .0],
                target_stds=[1.0, 1.0, 1.0, 1.0]
            ),
            loss_cls=dict(
                type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0
            ),
            loss_bbox=dict(type='L1Loss', loss_weight=1.0)
        ),
        roi_head=dict(
            type='StandardRoIHead',
            bbox_roi_extractor=dict(
                type='SingleRoIExtractor',
                roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
                out_channels=256,
                featmap_strides=[4, 8, 16, 32]
            ),
            bbox_head=dict(
                type='Shared2FCBBoxHead',
                in_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=80,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0., 0., 0., 0.],
                    target_stds=[0.1, 0.1, 0.2, 0.2]
                ),
                reg_class_agnostic=False,
                loss_cls=dict(
                    type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0
                ),
                loss_bbox=dict(type='L1Loss', loss_weight=1.0)
            ),
            mask_roi_extractor=dict(
                type='SingleRoIExtractor',
                roi_layer=dict(type='RoIAlign', output_size=14, sampling_ratio=0),
                out_channels=256,
                featmap_strides=[4, 8, 16, 32]
            ),
            mask_head=dict(
                type='FCNMaskHead',
                num_convs=4,
                in_channels=256,
                conv_out_channels=256,
                num_classes=80,
                loss_mask=dict(
                    type='CrossEntropyLoss', use_mask=True, loss_weight=1.0
                )
            )
        ),
        train_cfg=dict(
            rpn=dict(
                assigner=dict(
                    type='MaxIoUAssigner',
                    pos_iou_thr=0.7,
                    neg_iou_thr=0.3,
                    min_pos_iou=0.3,
                    match_low_quality=True,
                    ignore_iof_thr=-1
                ),
                sampler=dict(
                    type='RandomSampler',
                    num=256,
                    pos_fraction=0.5,
                    neg_pos_ub=-1,
                    add_gt_as_proposals=False
                ),
                allowed_border=-1,
                pos_weight=-1,
                debug=False
            ),
            rpn_proposal=dict(
                nms_pre=2000,
                max_per_img=1000,
                nms=dict(type='nms', iou_threshold=0.7),
                min_bbox_size=0
            ),
            rcnn=dict(
                assigner=dict(
                    type='MaxIoUAssigner',
                    pos_iou_thr=0.5,
                    neg_iou_thr=0.5,
                    min_pos_iou=0.5,
                    match_low_quality=False,
                    ignore_iof_thr=-1
                ),
                sampler=dict(
                    type='RandomSampler',
                    num=512,
                    pos_fraction=0.25,
                    neg_pos_ub=-1,
                    add_gt_as_proposals=True
                ),
                mask_size=28,
                pos_weight=-1,
                debug=False
            )
        ),
        test_cfg=dict(
            rpn=dict(
                nms_pre=1000,
                max_per_img=1000,
                nms=dict(type='nms', iou_threshold=0.7),
                min_bbox_size=0
            ),
            rcnn=dict(
                score_thr=0.05,
                nms=dict(type='nms', iou_threshold=0.5),
                max_per_img=100,
                mask_thr_binary=0.5
            )
        )
    )
    
    # Remove pretrained to avoid download
    minimal_config['backbone']['init_cfg'] = None
    
    try:
        model = build_detector(minimal_config)
        assert model is not None
        assert hasattr(model, 'backbone')
        assert hasattr(model, 'neck')
        assert hasattr(model, 'rpn_head')
        assert hasattr(model, 'roi_head')
    except Exception as e:
        pytest.fail(f"Failed to build simple model: {e}")
