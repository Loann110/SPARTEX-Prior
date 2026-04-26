#########################################################################
# core.py 
# Written by Loann KAIKA
# last update : 03/04/2026
#########################################################################

import cv2
import numpy as np

from scipy.ndimage import gaussian_filter, gaussian_laplace
from skimage.color import rgb2lab
from skimage.segmentation import slic

################################################
# Utils
################################################

def load_rgb_image(path):
    img_bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise FileNotFoundError(f"[ERROR] Image cannot be read : {path}")
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return img_rgb


def load_mask_binary(path):
    mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"[ERROR] Mask cannot be read : {path}")
    mask = (mask > 127).astype(np.uint8)
    return mask


################################################
# Filter Bank for Textons
################################################

def build_filter_responses(gray_float):
    """
    gray_float: image (float32) normalized [0,1], shape (H,W)
    Return feat_map, shape (H,W,D)
    """
    sigmas = [1, 2, 4]
    responses = []

    for sigma in sigmas:
        # Smoothing
        g = gaussian_filter(gray_float, sigma=sigma)
        responses.append(g)

        # LoG
        log = gaussian_laplace(gray_float, sigma=sigma)
        responses.append(log)

        # 1st order derivatives
        gx = gaussian_filter(gray_float, sigma=sigma, order=[1, 0])
        gy = gaussian_filter(gray_float, sigma=sigma, order=[0, 1])
        responses.append(gx)
        responses.append(gy)

        # 2nd order derivatives
        gxx = gaussian_filter(gray_float, sigma=sigma, order=[2, 0])
        gyy = gaussian_filter(gray_float, sigma=sigma, order=[0, 2])
        gxy = gaussian_filter(gray_float, sigma=sigma, order=[1, 1])
        responses.append(gxx)
        responses.append(gyy)
        responses.append(gxy)

    feat_map = np.stack(responses, axis=-1).astype(np.float32)
    return feat_map

################################################
# Assigning Textons to an Image
################################################

def compute_texton_map(image_rgb, texton_model, predict_chunk_size=200000):
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    feat_map = build_filter_responses(gray)

    H, W, D = feat_map.shape
    X = feat_map.reshape(-1, D)
    Xn = (X - texton_model["mean"]) / texton_model["std"]

    labels = np.empty(Xn.shape[0], dtype=np.int32)

    for start in range(0, Xn.shape[0], predict_chunk_size):
        end = min(start + predict_chunk_size, Xn.shape[0])
        labels[start:end] = texton_model["kmeans"].predict(Xn[start:end])

    texton_map = labels.reshape(H, W)
    return texton_map.astype(np.int32)

################################################
# Superpixel Segmentation ( SLIC Method )
################################################

def segment_superpixels(image_rgb, n_segments=200, compactness=10, sigma=1):
    labels = slic(
        image_rgb,
        n_segments=n_segments,
        compactness=compactness,
        sigma=sigma,
        start_label=0
    )
    return labels.astype(np.int32)

################################################
# Features LAB + Textons
################################################

def compute_lab_image(image_rgb):
    lab = rgb2lab(image_rgb.astype(np.float32) / 255.0).astype(np.float32)
    return lab


def lab_hist_features_from_lab(lab, region_mask, n_bins=21):
    L = lab[:, :, 0][region_mask]
    a = lab[:, :, 1][region_mask]
    b = lab[:, :, 2][region_mask]

    hist_L, _ = np.histogram(L, bins=n_bins, range=(0, 100), density=True)
    hist_a, _ = np.histogram(a, bins=n_bins, range=(-128, 127), density=True)
    hist_b, _ = np.histogram(b, bins=n_bins, range=(-128, 127), density=True)

    return np.concatenate([hist_L, hist_a, hist_b]).astype(np.float32)


def texton_hist_features(texton_map, region_mask, n_textons):
    vals = texton_map[region_mask]
    hist, _ = np.histogram(vals, bins=np.arange(n_textons + 1), density=True)
    return hist.astype(np.float32)


def extract_superpixel_feature(lab, sp_labels, sp_id, texton_map, n_bins_lab=21, n_textons=16):
    region_mask = (sp_labels == sp_id)

    feat_lab = lab_hist_features_from_lab(lab, region_mask, n_bins=n_bins_lab)
    feat_txt = texton_hist_features(texton_map, region_mask, n_textons=n_textons)

    feat = np.concatenate([feat_lab, feat_txt]).astype(np.float32)
    return feat

################################################
# Get Score
################################################

def get_score(model, feat, output_mode="decision"):
    """
    output_mode:
      - 'decision' : score brut du classifieur
      - 'prob'     : vraie proba si modèle calibré, sinon pseudo-proba par sigmoid(score)
      - 'logit'    : logit
    """
    if output_mode == "decision":
        if hasattr(model, "decision_function"):
            return float(model.decision_function(feat)[0])
        else:
            p = float(model.predict_proba(feat)[0, 1])
            p = np.clip(p, 1e-6, 1 - 1e-6)
            return float(np.log(p / (1 - p)))

    elif output_mode == "prob":
        if hasattr(model, "predict_proba"):
            return float(model.predict_proba(feat)[0, 1])
        else:
            s = float(model.decision_function(feat)[0])
            return float(1.0 / (1.0 + np.exp(-s)))

    elif output_mode == "logit":
        if hasattr(model, "predict_proba"):
            p = float(model.predict_proba(feat)[0, 1])
            p = np.clip(p, 1e-6, 1 - 1e-6)
            return float(np.log(p / (1 - p)))
        else:
            return float(model.decision_function(feat)[0])

    else:
        raise ValueError("output_mode doit être 'decision', 'prob' ou 'logit'")
    



