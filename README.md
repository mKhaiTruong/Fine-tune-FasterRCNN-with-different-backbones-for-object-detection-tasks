# 🧠 Fine-tuning a Faster RCNN model with different backbones for Object Detection

## 📁 Dataset
You can use any datasets in COCO format. My best advice is to label each object in Roboflow then download the dataset in COCO format.

---

## 🛠️ Project Pipeline: Dataset and Models

1. **Custom Dataset**
   - A PyTorch VisionDataset wrapper for loading object detection data in COCO format. Its main points are:
    + Loading images and annotations from COCO-style JSON.
    + Filtering out images with no annotations.
    + Using Albumentations for image and bbox transforms.
    + Converting bounding boxes to [x_min, y_min, x_max, y_max] format.
    + Returning normalized image tensors and detection targets (boxes, labels, image_id, area, iscrowd).

2. **Augmentations**:
   - You can find all my preferred augmentations inside utils.py.

3. **Model Explaination**
   - My code sets up a flexible object detection pipeline using Faster R-CNN with various backbones (ResNet-50, EfficientNet-B3, Swin Transformer) for COCO-style datasets. It includes:
      + Custom Dataset Loader: COCODetection class that loads images/annotations in COCO format and supports Albumentations for data augmentation.
      + Backbone Flexibility: ResNet-50 + FPN (via torchvision); EfficientNet-B3; Swin Transformer; all can be found in *utils.py*
      + Custom Anchor Generation: Tailored anchors for improved object fitting across scales.

---

## 🔁 Project Pipeline: Training Function

1. **train_folds**
   - This function handles all necessary functions, such as:
     + saveBestModel: Stores the path leading to N best models. Those models are determined by 'score', which is calculated by 50% mAP and 50% f1 score. In the code, I set N to 3, so there will be 3 stored models.
     + filter_invalid_bboxes: as the name suggests, all the bboxes, which have square equal or less than 0, will be discarded.
     + train_function: simply trains the model and returns the average training loss.
     + evaluate_function: this function is aiming for 3 main goals: calculating the metrics, using those metrics to find 3 best models and storing those metrics for later visualization.
     + add_in_file: stores outputs, especially metrics to a .txt file.
     + and below....

2. **K-Fold Training**
   - This function handles multi-fold training using KFold cross-validation — a robust way to assess generalization. Each fold:
    + Splits the dataset into training and validation.
    + Re-initializes the model, optimizer, and data loaders.
    + Applies multiple best-practice strategies (detailed below).

3. **best-practice strategies, or tricks, to enhance the performance**
   - Anchor Box Optimization (via KMeans): Uses the distribution of bounding box widths/heights across annotations; and applies KMeans clustering to find 9 anchor shapes.
   - Class Imbalance Mitigation: This ensures underrepresented classes are seen more during training.
   - Progressive Resizing: Trains with SMALL_SIZE initially for speed and after some epochs, increases resolution to FULL_SIZE. This is a regularization trick for Swim
   - Mixed Precision + GradScaler: Uses torch.amp.autocast() and GradScaler() for efficient mixed-precision training.

---

## 📊 Project Pipeline: Model Evaluation

  - Uses MeanAveragePrecision metric (from torchmetrics) for MAP@.5, .75, etc.
  - Custom F1-score computation using IoU-based matching.
  - Early stopping if no improvement for PATIENCE epochs.
  - *Evaluation score* = weighted sum of mAP@0.5 and F1.
---

## 🧠 Project Pipeline: Hyperparameter Tuning (objective)

  - The code leverages Optuna to optimize:
      + Learning rate (log-uniform)
      + Batch size (4 or 8)
      + score threshold for prediction filtering
  - For each trial:
      + Trains with train_folds.
      + Loads the top 3 models and runs an ensemble using Weighted Boxes Fusion (WBF).
      + Aggregates predictions and computes final validation F1 and mAP@0.5.
---
