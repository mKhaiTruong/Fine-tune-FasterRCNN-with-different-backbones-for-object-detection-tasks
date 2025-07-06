# ============================================ Import ========================================
# 1. Handle datasets
import io
import os
import gc
import sys
import cv2
import time
import copy
import math
import timm
import cvzone
import random
import pydicom
import dicomsdl
import numpy as np
import pandas as pd
from PIL import Image
from glob import glob
import tifffile as tiff
import imageio.v3 as iio
import SimpleITK as sitk
from pathlib import Path
from tqdm.auto import tqdm
import multiprocessing as mp
from collections import Counter
from joblib import Parallel, delayed
from pydicom.pixel_data_handlers.util import apply_voi_lut

# 2. Visualize datasets
import datetime as dtime
from datetime import datetime
import itertools
import matplotlib.pyplot as plt 
import seaborn as sns 
import plotly.express as px
import plotly.figure_factory as pff
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from matplotlib.patches import Rectangle
from IPython.display import display_html, clear_output, display, Image

# 3. Preprocess datasets
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.impute import SimpleImputer, KNNImputer
## import iterative impute
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
## fastai
# from fastai.data.all import *
# from fastai.vision.all import *

# 4. machine learning
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.model_selection import StratifiedKFold, GroupKFold
## for classification
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from sklearn.preprocessing import KBinsDiscretizer
from xgboost import XGBClassifier

# 5. Deep Learning
## Augmentation
import ultralytics
from ultralytics import YOLO
from roboflow import Roboflow
from pycocotools.coco import COCO
from ensemble_boxes import weighted_boxes_fusion
import albumentations as A
from albumentations.pytorch import ToTensorV2
from transformers import ViTModel, ViTFeatureExtractor, ViTForImageClassification

## Torch
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch import FloatTensor, LongTensor
from torchvision.utils import draw_bounding_boxes
from torch.utils.data import Dataset, DataLoader, Subset
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR, SequentialLR, LinearLR
from efficientnet_pytorch import EfficientNet
from torchvision.models import resnet34, resnet50, ResNet50_Weights, efficientnet_v2_m, EfficientNet_V2_M_Weights, detection
from torchvision import datasets, transforms
from torchvision.models.detection.backbone_utils import BackboneWithFPN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.transform import GeneralizedRCNNTransform
from torchvision.models.feature_extraction import create_feature_extractor
from torchvision.ops.feature_pyramid_network import FeaturePyramidNetwork
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torchvision.ops import box_iou

# 6. metrics
import optuna
from scipy.stats import mode
from timeit import default_timer as timer
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from sklearn.metrics import f1_score, r2_score
from sklearn.metrics import classification_report

# 7. ignore warnings and wandb 
import warnings
import wandb

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# =========================================================================================

# === Here lies some general functionalities ===
# === Seeds and Visualization ===
def set_seed(seed = 1234):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    
    # On CuDNN we need 2 further options
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    os.environ['PYTHONHASHSEED'] = str(seed)

def show_values_on_bars(axs, h_v = 'v', space = 0.4):
    def _show_on_single_plot(ax):
        if h_v == 'v':
            for p in ax.patches:
                _x = p.get_x() + p.get_width() / 2
                _y = p.get_y() + p.get_height()
                
                value = int(p.get_height())
                ax.text(_x, _y, format(value, ','), ha='center')
        elif h_v == 'h':
            for p in ax.patches:
                _x = p.get_x() + p.get_width() + float(space)
                _y = p.get_x() + p.get_height()
                
                value = int(p.get_width())
                ax.text(_x, _y, format(value, ','), ha='left')
    
    if isinstance(axs, np.ndarray):
        for i, ax in np.ndenumerate(axs):
            _show_on_single_plot(ax)
    else:
        _show_on_single_plot(axs)


# ============================================= WANDB ===========================================
def save_dataset_artifact(run_name, artifact_name, path, 
                          projectName = None, config = None, data_type = "dataset"):
    run = wandb.init(project=projectName,
                     name = run_name,
                     config= config)

    artifact = wandb.Artifact(name = artifact_name,
                              type = data_type)
    artifact.add_file(path)
    
    wandb.log_artifact(artifact)
    wandb.finish()
    print("Artifact has been save successfully!")

def create_wandb_plot(x_data = None, y_data = None, x_name = None, y_name = None,
                      title = None, log = None, plot = "line"):
    data = [
        [label, val] for (label, val) in zip(x_data, y_data)
    ]
    table = wandb.Table(data = data, columns = [x_name, y_name])
    
    if plot == "line":
        wandb.log({ log: wandb.plot.line(table, x_name, y_name, title=title) })
    elif plot == "bar":
        wandb.log({ log: wandb.plot.bar(table, x_name, y_name, title=title) })
    elif plot == "scatter":
        wandb.log({ log: wandb.plot.scatter(table, x_name, y_name, title=title) })

def create_wandb_hist(x_data = None, x_name = None, title = None, log = None):
    data = [[x] for x in x_data]
    table = wandb.Table(data = data, columns=[x_name])
    wandb.log({ log: wandb.plot.histogram(table, x_name, title=title) })
    

def show_stacked_images(image_tensor_batch, target_labels=None):
    
    num_images = image_tensor_batch.size(0)
    sqrt_n = int(math.sqrt(num_images))
    ncols = sqrt_n
    nrows = math.ceil(num_images / ncols)
    fig, axis = plt.subplots(
        nrows=nrows, ncols=ncols, figsize=(12 * ncols, 8 * nrows)
    )
    axis = axis.flatten()
    
    for i in range(num_images):
        image_tensor = image_tensor_batch[i]
        image = image_tensor.cpu().numpy().transpose((1, 2, 0))     # Transpose to HWC
        
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
    
        image = std * image + mean                                  # Unnormalize
        image = np.clip(image, 0, 1)
        
         # Convert RGB to grayscale
        image_gray = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        image_gray = image_gray / 255.0
        
        axis[i].imshow(image_gray, cmap="bone")
        axis[i].axis('off')
        if target_labels is not None:
            axis[i].set_title(f"Target: {target_labels[i].item()}")
    
    plt.tight_layout()
    plt.axis('off')
    plt.show()

 
# ================================= AUGMENTATION =========================================
def transforms(size: int, isTrain=False):
    aug_list = []
    
    if isTrain:
        aug_list += [
            
            # === Spatial: flips & small rotations ===
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Rotate(limit=20, p=0.3),
            
            # === Scale jitter & random crop ===
            # A.RandomResizedCrop(
            #     size=(size, size),
            #     scale=(0.8, 1.0),
            #     ratio=(0.75, 1.33),
            #     p=0.5
            # ),
            
            # === Photometric: brightness / contrast / hue shifts ===
            A.RandomBrightnessContrast(p=0.2),
            A.HueSaturationValue(p=0.3),
            # A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.2),
            
            # === Mild blur ===
            A.OneOf([
                A.MotionBlur(blur_limit=3, p=0.3),
                A.GaussianBlur(blur_limit=4, p=0.4),
            ], p=0.2),
            
        ]
    
    # === Resize & ToTensor ===
    aug_list += [
        A.Resize(size, size),
        ToTensorV2()
    ]
    
    return A.Compose(aug_list, bbox_params=A.BboxParams(format='coco'))


def collate_fn(batch):
    return tuple(zip(*batch))

# Define the DataLoader creation function
def CreateLoader(dataset, batch_size=8, shuffle=False, sampler=None):
    if sampler is not None:
        shuffle = False
        
    return DataLoader(
        dataset, batch_size,
        shuffle, sampler, collate_fn=collate_fn,
        pin_memory=True, drop_last=True
    )

def data_to_device(data, DEVICE):
    images, targets = data
    images = [img.to(DEVICE) for img in images]
    targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]
    return images, targets

def compute_detection_metrics(preds_list, gts_list, iou_thr=0.5):
    """
        Returns:
        mAP50: float  — average precision at IoU=0.50
        f1:    float  — F1 score at IoU=0.50, using your TP/FP/FN logic
    """
    
    # 1) mAP50 via torchmetrics
    metric = MeanAveragePrecision(iou_type="bbox", iou_thresholds=[iou_thr])
    for preds, gt in zip(preds_list, gts_list):
        
        metric.update(preds, [gt])
    res = metric.compute()
    mAP50 = res["map_50"].item()

    # 2) F1 at IoU=0.50
    total_tp = total_fp = total_fn = 0
    for preds, gt in zip(preds_list, gts_list):
        pb, pl = preds[0]["boxes"], preds[0]["labels"]
        gb, gl = gt["boxes"],       gt["labels"]

        # handle edge cases
        if pb.numel() == 0:
            total_fn += gb.size(0)
            continue
        if gb.numel() == 0:
            total_fp += pb.size(0)
            continue

        ious = box_iou(pb, gb)  # [P, G]
        matches = (ious >= iou_thr).nonzero(as_tuple=False)
        # sort by IoU descending
        scores = ious[matches[:,0], matches[:,1]]
        order  = torch.argsort(scores, descending=True)

        tp = 0
        used_p, used_g = set(), set()
        for idx in order:
            p, g = matches[idx,0].item(), matches[idx,1].item()
            if p not in used_p and g not in used_g and pl[p] == gl[g]:
                tp += 1
                used_p.add(p)
                used_g.add(g)

        fp = pb.size(0) - tp
        fn = gb.size(0) - tp
        total_tp += tp
        total_fp += fp
        total_fn += fn

    precision = total_tp / (total_tp + total_fp) if total_tp+total_fp>0 else 0.0
    recall    = total_tp / (total_tp + total_fn) if total_tp+total_fn>0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision+recall)>0 else 0.0

    return mAP50, f1

# MODELS

# Ensemble function: Using N models for voting the best result
# Detection voting = box-level fusion (WBF) or score-averaging + NMS.
def load_ensemble_models(model_paths, model_type, num_classes, size = None, device = 'cpu'):
    
    BASE_DIR = r'D:\Deep_Learning_Object_Detection\randProjects\ObjDet_PlantsCountingRCNN'
    models = []
    
    for path in model_paths:
        if size is None:
            m = model_type(num_classes).to(device)
        else:
            m = model_type(num_classes, size).to(device)
        m.load_state_dict(torch.load(
            os.path.join(BASE_DIR, path),
            map_location=device
        ))
        m.eval()
        
        models.append(m)
    
    return models

def ensemble_wbf(models, image_tensor, iou_threshold=0.4, skip_box_threshold=0.05):
    
    models = [m.eval() for m in models]
    h, w = image_tensor.shape[1:]
    all_boxes, all_scores, all_labels = [], [], []
    
    with torch.no_grad():
        for model in models:
            preds = model([ image_tensor.to(next(model.parameters()).device) ])[0]
            
            boxes  = preds['boxes'].cpu().numpy() / [w, h, w, h]
            scores = preds['scores'].cpu().numpy().tolist()
            labels = preds['labels'].cpu().numpy().tolist()
            
            all_boxes.append(boxes)
            all_scores.append(scores)
            all_labels.append(labels)
    
    # Weighted Boxes Fusion (WBF)
    boxes, scores, labels = weighted_boxes_fusion(
        all_boxes, all_scores, all_labels,
        iou_thr = iou_threshold, skip_box_thr = skip_box_threshold
    )
    
    # convert back to pixel coords
    preds = [{
        'boxes' : torch.tensor(boxes * [w, h, w, h], dtype=torch.float32),
        'scores': torch.tensor(scores, dtype=torch.float32),
        'labels': torch.tensor(labels, dtype=torch.int64),
    }]

    return preds    
            
    

# ==================== FASTER_RCNN + RESTNET50_FPN ========================
def cus_fasterrcnn_resnet50_fpn(num_classes):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = detection.faster_rcnn.FastRCNNPredictor(
        in_features, num_classes
    )
    return model
# ==========================================================================

# ==================== FASTER_RCNN + EFFNET_FPN ================================
def cus_fasterrcnn_effnet(num_classes):
    # 1) Load EfficientNet-B3 that returns a list of feature maps
    timm_backbone = timm.create_model(
        'efficientnet_b3',
        pretrained=True,
        features_only=True,
        out_indices=(0,1,2,3,4),
    )
    in_ch = timm_backbone.feature_info.channels()  # e.g. [40, 80, 192, 384, 1280]
    out_ch = 256

    # 2) Wrap list→dict + FPN, and set out_channels
    class TimmEffnetFPN(torch.nn.Module):
        def __init__(self, body, in_channels_list, out_channels):
            super().__init__()
            self.body = body
            self.fpn  = FeaturePyramidNetwork(
                in_channels_list=in_channels_list,
                out_channels=out_channels
            )
            # THIS is the key line:
            self.out_channels = out_channels

        def forward(self, x):
            feats = self.body(x)            
            feats = { str(i): f for i, f in enumerate(feats) }
            return self.fpn(feats)

    backbone = TimmEffnetFPN(timm_backbone, in_ch, out_ch)

    # 3) Anchor gen tuned for your tree boxes
    anchor_gen = AnchorGenerator(
        sizes=(
            (32,),   # P3
            (64,),   # P4
            (128,),  # P5
            (256,),  # P6
            (512,),  # P7
        ),
        aspect_ratios=(
            (0.5, 1.0, 2.0),  # P3
            (0.5, 1.0, 2.0),  # P4
            (0.5, 1.0, 2.0),  # P5
            (0.5, 1.0, 2.0),  # P6
            (0.5, 1.0, 2.0),  # P7
        )
    )

    # 4) Build the Faster R-CNN
    model = FasterRCNN(
        backbone,
        num_classes=num_classes,
        rpn_anchor_generator=anchor_gen,
        box_score_thresh=0.05,
    )
    return model
# ==========================================================================


# ==================== SWIN-TRANSFORMER ================================
def cus_fasterrcnn_swin(num_classes, size: int):
    timm_backbone = timm.create_model(
        'swin_small_patch4_window7_224',
        pretrained=True,
        features_only=True,
        out_indices=(0, 1, 2, 3),
    )
    timm_backbone.patch_embed.strict_img_size = False
    timm_backbone.patch_embed.dynamic_img_pad = True
    
    in_channels_list = timm_backbone.feature_info.channels()
    out_channels = 256

    # Wrap backbone + FPN
    class TimmSwin(torch.nn.Module):
        def __init__(self, body, in_channels_list, out_channels):
            super().__init__()
            self.body = body
            self.fpn = FeaturePyramidNetwork(
                in_channels_list=in_channels_list,
                out_channels=out_channels
            )
            self.out_channels = out_channels

        def forward(self, x):
            feats = self.body(x)  
            feats = [f.permute(0, 3, 1, 2).contiguous() for f in feats]         
            feats = {str(i): f for i, f in enumerate(feats)}
            return self.fpn(feats)

    backbone = TimmSwin(timm_backbone, in_channels_list, out_channels)

    # Anchor generator tuned for your objects
    anchor_gen = AnchorGenerator(
        sizes=(
            (64,),   # P4
            (128,),  # P5
            (256,),  # P6
            (512,),  # P7
        ),
        aspect_ratios=(
            (0.5, 1.0, 2.0),  # P4
            (0.5, 1.0, 2.0),  # P5
            (0.5, 1.0, 2.0),  # P6
            (0.5, 1.0, 2.0),  # P7
        )
    )

    model = FasterRCNN(
        backbone,
        num_classes=num_classes,
        rpn_anchor_generator=anchor_gen,
        box_score_thresh=0.05,
        min_size=[size],
        max_size=size,
    )
    return model
# ===========================================================================


# Theories

# 1. “Naïve voting” (classification style)
# Input: one model ⇒ one vector of class probabilities (e.g. [0.1, 0.9])

# Ensemble: average those vectors across models ⇒ pick the highest mean

# Output: single class label or probability for the entire image

# Your original mental model: three folds ⇒ three score outputs (0.97, 0.98, 0.96) ⇒ mean ⇒ 0.97. That works great if every model returns exactly one number per image.

# 2. Detection voting (what ensemble_wbf does)
# Input: each model ⇒ a list of boxes with individual confidence scores and labels

# Challenge: boxes from different models don’t line up one-to-one (each model might predict 15 boxes, another 20, with different coordinates)

# Goal: merge those lists into a single final list of boxes, each with a fused confidence and label

# Weighted Boxes Fusion (WBF) steps
# Normalize all box coordinates to [0–1] so they’re comparable across image sizes.

# Cluster boxes from all models whose IoU ≥ iou_threshold into groups—each group corresponds to one “true” object.

# Fuse box coordinates within each cluster by taking a score-weighted average of the corners.

# Fuse scores (and/or apply per-model weights) to get one confidence per fused box.

# Fuse labels (often by picking the label of the highest-scoring box or by majority vote within the cluster).

# Discard any fused boxes with confidence < skip_box_threshold.

# Rescale the result back to pixel coordinates and return it as your ensemble prediction.

# Why this “detection vote” is more powerful
# You don’t simply pick one model’s “best” box and ignore the rest.

# You blend information from all models to get tighter, more accurate bounding boxes.

# You still get a final confidence score per box, but it’s computed from all models’ scores, not just averaged once per image.


# Theory 2
# worry that a “weak” model (one that’s systematically poorer at localization or gives low confidence scores) will pull the fused box away from the “good” models’ predictions. In practice WBF and similar fusion schemes guard against that in two ways:

# Score‐weighted averaging

# When you fuse the corner coordinates of the boxes in a cluster, you weight each model’s box by its confidence score.

# A weak model that only assigns, say, 0.2 confidence to that box will have only 20% of the influence compared to a strong model that gives 0.9.

# so low-score boxes barely shift the average.

# Confidence thresholding

# Before fusion you can apply a skip_box_thr (e.g. 0.05). Any box below that is simply thrown out and doesn’t even enter a cluster.

# If a model is very weak everywhere, most of its boxes will get filtered before they can drag anything.

# What if a model is “mediocre but not awful”?
# Even if a model has middling confidence—say 0.4 where others have 0.8—its boxes still contribute, but only partially:

# Suppose two models each give 
# 𝑥0 = 100 at scores 0.8 and 0.7, and a weak one gives 𝑥0 = 150 at score 0.3.

# The fused 𝑥 0 would be (0.8 * 100 + 0.7 * 100 + 0.3 * 150) / (0.8 + 0.7 + 0.3) = 108.3 so it moves only ~8 pixels toward the weak outlier, not all the way to 150.


# How Gradient Clipping Works:
# When gradients become excessively large, the updates to the model parameters (weights) can become very large, potentially causing the model to diverge instead of converging to a good solution. 