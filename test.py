#########################################################################
# test.py
# Written by Loann KAIKA
# last update : 03/04/2026
#########################################################################

import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import joblib

from core import load_rgb_image, compute_texton_map, segment_superpixels, compute_lab_image
from core import extract_superpixel_feature, get_score

################################################
# Params
################################################

OUTPUT_DIR = r"C:\Users\loann\Desktop\projets\SPARTeX\dataset_\images"

#### Superpixel segmentation ####
N_SEGMENTS = 800
COMPACTNESS = 8
SIGMA_SLIC = 1

#### Color feats ####
N_BINS_LAB = 21

#### SVM ####
USE_CALIBRATED_SVM = True  # False = rapid ; True = slow but proba calibrated

################################################
# Prior Map Generation
################################################

def generate_target_prior_map( image_rgb, texton_model, svm_model,
                               n_segments=200, compactness=15,
                               sigma=1, output_mode="decision" ):
    
    sp_labels = segment_superpixels(
        image_rgb,
        n_segments=n_segments,
        compactness=compactness,
        sigma=sigma
    )

    texton_map = compute_texton_map(image_rgb, texton_model)
    lab = compute_lab_image(image_rgb)

    prior_map = np.zeros(sp_labels.shape, dtype=np.float32)

    for sp_id in np.unique(sp_labels):
        feat = extract_superpixel_feature(
            lab=lab,
            sp_labels=sp_labels,
            sp_id=sp_id,
            texton_map=texton_map,
            n_bins_lab=N_BINS_LAB,
            n_textons=texton_model["n_textons"]    
        ).reshape(1, -1)

        score = get_score(svm_model, feat, output_mode=output_mode)
        prior_map[sp_labels == sp_id] = score

    return prior_map, sp_labels, texton_map


################################################
# Save map as an image
################################################
def save_float_map_as_image(float_map, save_path):
    mn = float_map.min()
    mx = float_map.max()
    norm = (float_map - mn) / (mx - mn + 1e-8)
    u8 = (norm * 255).astype(np.uint8)
    cv2.imwrite(save_path, u8)

################################################
# Visualization
################################################

def show_results(image_rgb, texton_map, sp_labels, prior_map):
    plt.figure(figsize=(16, 5))

    plt.subplot(1, 4, 1)
    plt.imshow(image_rgb)
    plt.title("Image")
    plt.axis("off")

    plt.subplot(1, 4, 2)
    plt.imshow(texton_map, cmap="tab20")
    plt.title("Texton map")
    plt.axis("off")

    plt.subplot(1, 4, 3)
    plt.imshow(sp_labels, cmap="nipy_spectral")
    plt.title("Superpixels")
    plt.axis("off")

    plt.subplot(1, 4, 4)
    plt.imshow(prior_map, cmap="jet")
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.title("Target prior map")
    plt.axis("off")

    plt.tight_layout()
    plt.show()

################################################
# Main (Test)
################################################

if __name__ == "__main__":

    TEST_IMAGE_PATH  = r"C:\Users\loann\Downloads\shadow.png"

    #### Path of the SVM model ####
    SVM_MODEL_PATH = r"C:\Users\loann\Desktop\projets\SPARTeX\dataset_\model\shadow_prior_svm.joblib"

    #### Path of the texton model ####
    TEXTON_MODEL_PATH = r"C:\Users\loann\Desktop\projets\SPARTeX\dataset_\model\shadow_texton_kmeans.joblib"

    print("Loading SVM model...")
    svm_model = joblib.load(SVM_MODEL_PATH)
    print("SVM model loaded from :", SVM_MODEL_PATH)

    print("Loading texton model...")
    texton_model = joblib.load(TEXTON_MODEL_PATH)
    print("Texton model loaded from :", TEXTON_MODEL_PATH)

    print("\n[ TEST on an image ]")
    test_img = load_rgb_image(TEST_IMAGE_PATH)

    test_output_mode = "decision" if not USE_CALIBRATED_SVM else "prob"

    prior_map, sp_labels, texton_map = generate_target_prior_map(
        image_rgb=test_img,
        texton_model=texton_model,
        svm_model=svm_model,
        n_segments=N_SEGMENTS,
        compactness=COMPACTNESS,
        sigma=SIGMA_SLIC,
        output_mode=test_output_mode
    )

    prior_save_path = os.path.join(OUTPUT_DIR, "target_prior_map.png")
    save_float_map_as_image(prior_map, prior_save_path)

    print(f"Prior map saved in : {prior_save_path}")

    show_results(
        image_rgb=test_img,
        texton_map=texton_map,
        sp_labels=sp_labels,
        prior_map=prior_map
    )