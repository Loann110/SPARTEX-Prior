#########################################################################
# train.py
# Written by Loann KAIKA
# last update : 03/04/2026
#########################################################################

import os
import glob
import joblib
import numpy as np
import cv2
from sklearn.cluster import MiniBatchKMeans
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.kernel_approximation import AdditiveChi2Sampler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, precision_score, recall_score,
    f1_score, jaccard_score, confusion_matrix, classification_report,
    roc_auc_score, average_precision_score
)

from core import load_rgb_image, build_filter_responses, load_mask_binary, compute_lab_image
from core import  segment_superpixels, compute_texton_map, extract_superpixel_feature

################################################
# Params
################################################

TRAIN_IMAGES_DIR = r"c:\Users\loann\Desktop\projets\SPARTeX\dataset_\images"
TRAIN_MASKS_DIR  = r"C:\Users\loann\Desktop\projets\SPARTeX\dataset_\masks"

OUTPUT_DIR       = r"C:\Users\loann\Desktop\projets\SPARTeX\dataset_\model"

#### Superpixel segmentation ####
"""
For precision (thin structure), 
try -> N_SEGMENTS = 800 or 1200
       COMPACTNESS = 8 or 5
       SIGMA_SLIC = 1 or 0.5
"""
N_SEGMENTS = 800 # 
COMPACTNESS = 8
SIGMA_SLIC = 1

#### Textons ####
N_TEXTONS = 16 # you can try another value (32 for ex)
MAX_TOTAL_SAMPLES_FOR_TEXTONS = 150000   # global budget (RAM safe)
RANDOM_STATE = 42
KMEANS_BATCH_SIZE = 4096

#### Color feats ####
N_BINS_LAB = 21

#### SVM ####
SVM_C = 1.0
USE_CALIBRATED_SVM = False  # False = rapid ; True = slow but proba calibrated

#### Evaluation ####
EVALUATE_MODEL = True
VALIDATION_SIZE = 0.2
TRAIN_FINAL_MODEL_ON_ALL_DATA = True  # True = evaluate on split, then retrain on all data before saving

#### Limit ####
TRAIN_IMAGE_LIMIT = None # To debug -> Try small, else None

#### Save models ####
MODEL_PATH = os.path.join(OUTPUT_DIR, "target_prior_svm.joblib")
TEXTON_PATH = os.path.join(OUTPUT_DIR, "texton_kmeans.joblib")

os.makedirs(OUTPUT_DIR, exist_ok=True)

################################################
# Incremental Stats
################################################

def update_running_stats(sum_x, sum_x2, count, X):
    """
    Update global stats without saving all X on RAM.
    """
    if X.size == 0:
        return sum_x, sum_x2, count

    if sum_x is None:
        sum_x = X.sum(axis=0, dtype=np.float64)
        sum_x2 = (X ** 2).sum(axis=0, dtype=np.float64)
    else:
        sum_x += X.sum(axis=0, dtype=np.float64)
        sum_x2 += (X ** 2).sum(axis=0, dtype=np.float64)

    count += X.shape[0]
    return sum_x, sum_x2, count

################################################
# Learning Textons Dictionary (RAM SAFE)
################################################

def train_texton_dictionary( image_paths, n_textons=16, max_total_samples=150000,
                             random_state=42, batch_size_kmeans=4096 ):
    """
    Learn textons without concatenating all the data.
    2 passes :
      1) mean/std global incremental
      2) MiniBatchKMeans.partial_fit
    """
    rng = np.random.RandomState(random_state)

    n_images = len(image_paths)
    if n_images == 0:
        raise ValueError("No image to train textons.")

    samples_per_image = max(n_textons * 4, max_total_samples // n_images)

    print(f"[Textons] Nb images            : {n_images}")
    print(f"[Textons] Total pixels budget  : {max_total_samples}")
    print(f"[Textons] Samples / image     : {samples_per_image}")

   #------------------------------------------------------------
   # First pass : Global stats ---------------------------------
   #------------------------------------------------------------
   
    sum_x = None
    sum_x2 = None
    count = 0

    for i, img_path in enumerate(image_paths, 1):
        img_rgb = load_rgb_image(img_path)
        gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0

        feat_map = build_filter_responses(gray)
        X = feat_map.reshape(-1, feat_map.shape[-1])

        if len(X) > samples_per_image:
            idx = rng.choice(len(X), samples_per_image, replace=False)
            X = X[idx]

        sum_x, sum_x2, count = update_running_stats(sum_x, sum_x2, count, X)

        if i % 200 == 0 or i == n_images:
            print(f"[Pass 1] {i}/{n_images} images processed")

    mean = (sum_x / count).astype(np.float32)
    var = (sum_x2 / count) - (mean.astype(np.float64) ** 2)
    var = np.maximum(var, 1e-8)
    std = np.sqrt(var).astype(np.float32)

    mean = mean[None, :]
    std = std[None, :]

   #------------------------------------------------------------
   # Second pass : incremental KMeans --------------------------
   #------------------------------------------------------------

    kmeans = MiniBatchKMeans( n_clusters=n_textons,
                              batch_size=batch_size_kmeans,
                              n_init=3, random_state=random_state )

    for i, img_path in enumerate(image_paths, 1):
        img_rgb = load_rgb_image(img_path)
        gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0

        feat_map = build_filter_responses(gray)
        X = feat_map.reshape(-1, feat_map.shape[-1])

        if len(X) > samples_per_image:
            idx = rng.choice(len(X), samples_per_image, replace=False)
            X = X[idx]

        Xn = (X - mean) / std
        kmeans.partial_fit(Xn)

        if i % 200 == 0 or i == n_images:
            print(f"[Pass 2] {i}/{n_images} images processed")

    texton_model = {
        "kmeans": kmeans,
        "mean": mean.astype(np.float32),
        "std": std.astype(np.float32),
        "n_textons": n_textons
    }
    return texton_model

################################################
# Collect (image / mask) pairs 
################################################

def collect_image_mask_pairs(images_dir, masks_dir):
    """
    Associate image / mask with the same filename.
    Ex:
      images/img_001.jpg
      masks/img_001.png  or masks/img_001.jpg
    """
    image_paths = sorted(glob.glob(os.path.join(images_dir, "*.*")))
    pairs = []

    for img_path in image_paths:
        base = os.path.splitext(os.path.basename(img_path))[0]

        candidates = [
            os.path.join(masks_dir, base + ".png"),
            os.path.join(masks_dir, base + ".jpg"),
            os.path.join(masks_dir, base + ".jpeg"),
            os.path.join(masks_dir, base + ".bmp"),
            os.path.join(masks_dir, base + ".tif"),
            os.path.join(masks_dir, base + ".tiff"),
        ]

        mask_path = None
        for c in candidates:
            if os.path.exists(c):
                mask_path = c
                break

        if mask_path is not None:
            pairs.append((img_path, mask_path))

    if len(pairs) == 0:
        raise RuntimeError("No pair image/masque founded.")

    return pairs

################################################
# Label of a Superpixel from a GT Mask
################################################

def superpixel_label_from_mask(mask_gt, sp_labels, sp_id, threshold=0.5):
    region_mask = (sp_labels == sp_id)
    ratio_target = mask_gt[region_mask].mean()
    return 1 if ratio_target >= threshold else 0

################################################
# Dataset construction for the SVM
################################################

def build_superpixel_dataset(pairs, texton_model, n_segments=200, compactness=10, sigma=1):
    X = []
    y = []

    for i, (img_path, mask_path) in enumerate(pairs, 1):
        image_rgb = load_rgb_image(img_path)
        mask_gt = load_mask_binary(mask_path)

        if image_rgb.shape[:2] != mask_gt.shape[:2]:
            raise ValueError(f"Size image/masque different for : {img_path}")

        lab = compute_lab_image(image_rgb)
        sp_labels = segment_superpixels(
            image_rgb,
            n_segments=n_segments,
            compactness=compactness,
            sigma=sigma
        )
        texton_map = compute_texton_map(image_rgb, texton_model)

        for sp_id in np.unique(sp_labels):
            feat = extract_superpixel_feature(
                lab=lab,
                sp_labels=sp_labels,
                sp_id=sp_id,
                texton_map=texton_map,
                n_bins_lab=N_BINS_LAB,
                n_textons=texton_model["n_textons"]
            )
            label = superpixel_label_from_mask(mask_gt, sp_labels, sp_id)

            X.append(feat)
            y.append(label)

        if i % 200 == 0 or i == len(pairs):
            print(f"[Dataset SVM] {i}/{len(pairs)} images processed")

    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.int32)
    return X, y


################################################
# Evaluation Metrics
################################################

def _get_prediction_scores(model, X):
    """
    Return continuous scores for ROC-AUC / Average Precision.
    Works with calibrated and non-calibrated models.
    """
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    elif hasattr(model, "decision_function"):
        return model.decision_function(X)
    else:
        return model.predict(X)


def evaluate_target_prior_svm(model, X_val, y_val):
    """
    Evaluate the SVM on validation superpixels.

    Metrics:
      - Accuracy: global correctness
      - Balanced accuracy: better when classes are imbalanced
      - Precision: reliability of target predictions
      - Recall: ability to find target regions
      - F1-score: precision/recall compromise
      - IoU/Jaccard: overlap-like metric for binary labels
      - Confusion matrix: TN, FP, FN, TP
      - ROC-AUC / Average Precision: score quality if possible
    """
    y_pred = model.predict(X_val)
    y_score = _get_prediction_scores(model, X_val)

    acc = accuracy_score(y_val, y_pred)
    bacc = balanced_accuracy_score(y_val, y_pred)
    precision = precision_score(y_val, y_pred, zero_division=0)
    recall = recall_score(y_val, y_pred, zero_division=0)
    f1 = f1_score(y_val, y_pred, zero_division=0)
    iou = jaccard_score(y_val, y_pred, zero_division=0)
    cm = confusion_matrix(y_val, y_pred, labels=[0, 1])

    print("\n[ Evaluation on validation superpixels ]")
    print(f"Accuracy          : {acc:.4f}")
    print(f"Balanced accuracy : {bacc:.4f}")
    print(f"Precision         : {precision:.4f}")
    print(f"Recall            : {recall:.4f}")
    print(f"F1-score          : {f1:.4f}")
    print(f"IoU / Jaccard     : {iou:.4f}")

    print("\nConfusion matrix [labels: 0=non-target, 1=target]")
    print(cm)

    if len(np.unique(y_val)) == 2:
        try:
            roc_auc = roc_auc_score(y_val, y_score)
            avg_precision = average_precision_score(y_val, y_score)
            print(f"ROC-AUC           : {roc_auc:.4f}")
            print(f"Average precision : {avg_precision:.4f}")
        except Exception as e:
            print(f"ROC-AUC/AP skipped: {e}")
    else:
        print("ROC-AUC/AP skipped: validation set contains only one class")

    print("\nClassification report")
    print(classification_report(
        y_val,
        y_pred,
        labels=[0, 1],
        target_names=["non-target", "target"],
        zero_division=0
    ))

################################################
# Train SVM
################################################

def train_target_prior_svm(X, y, C=1.0, use_calibrated=False):
    base_model = Pipeline([
        ("chi2_map", AdditiveChi2Sampler(sample_steps=2)),
        ("svm", LinearSVC(C=C, class_weight="balanced", max_iter=10000))
    ])

    if use_calibrated:
        clf = CalibratedClassifierCV(base_model, method="sigmoid", cv=3)
        clf.fit(X, y)
        return clf
    else:
        base_model.fit(X, y)
        return base_model

################################################
# Main (Training)
################################################

if __name__ == "__main__":

    print(" [ Collecting pairs image/masque ]")
    pairs = collect_image_mask_pairs(TRAIN_IMAGES_DIR, TRAIN_MASKS_DIR)

    if TRAIN_IMAGE_LIMIT is not None:
        pairs = pairs[:TRAIN_IMAGE_LIMIT]

    print(f"Number of pairs founded : {len(pairs)}")

    print("\n[ Learning Textons dictionary ]")
    train_image_paths = [p[0] for p in pairs]
    texton_model = train_texton_dictionary(
        image_paths=train_image_paths,
        n_textons=N_TEXTONS,
        max_total_samples=MAX_TOTAL_SAMPLES_FOR_TEXTONS,
        random_state=RANDOM_STATE,
        batch_size_kmeans=KMEANS_BATCH_SIZE
    )
    joblib.dump(texton_model, TEXTON_PATH)
    print(f"Texton model saved in : {TEXTON_PATH}")

    print("\n[ Building Superpixel dataset for the SVM ]")
    X, y = build_superpixel_dataset(
        pairs=pairs,
        texton_model=texton_model,
        n_segments=N_SEGMENTS,
        compactness=COMPACTNESS,
        sigma=SIGMA_SLIC
    )
    print("X shape:", X.shape)
    print("y shape:", y.shape)
    print("Nb target:", int((y == 1).sum()))
    print("Nb non-target:", int((y == 0).sum()))

    if EVALUATE_MODEL:
        print("\n[ Splitting train / validation data ]")

        unique_classes, class_counts = np.unique(y, return_counts=True)
        can_stratify = len(unique_classes) == 2 and np.all(class_counts >= 2)
        stratify_labels = y if can_stratify else None

        X_train, X_val, y_train, y_val = train_test_split(
            X, y,
            test_size=VALIDATION_SIZE,
            random_state=RANDOM_STATE,
            stratify=stratify_labels
        )

        print("X_train shape:", X_train.shape)
        print("X_val shape  :", X_val.shape)
        print("y_train shape:", y_train.shape)
        print("y_val shape  :", y_val.shape)

        print("\n[ training SVM for evaluation ]")
        eval_svm_model = train_target_prior_svm(
            X_train, y_train,
            C=SVM_C,
            use_calibrated=USE_CALIBRATED_SVM
        )

        evaluate_target_prior_svm(eval_svm_model, X_val, y_val)

        if TRAIN_FINAL_MODEL_ON_ALL_DATA:
            print("\n[ training final SVM on all data ]")
            svm_model = train_target_prior_svm(
                X, y,
                C=SVM_C,
                use_calibrated=USE_CALIBRATED_SVM
            )
        else:
            print("\n[ saving evaluation SVM trained on train split ]")
            svm_model = eval_svm_model

    else:
        print("\n[ training SVM ]")
        svm_model = train_target_prior_svm(
            X, y,
            C=SVM_C,
            use_calibrated=USE_CALIBRATED_SVM
        )

    joblib.dump(svm_model, MODEL_PATH)
    print(f"SVM model saved in : {MODEL_PATH}")

