# SPARTEX-Prior

**SPARTEX-Prior** is a lightweight classical computer vision framework for generating **target prior maps** from images using **superpixels**, **textons**, **LAB color histograms**, and an **SVM classifier**.

The goal of the framework is not to directly perform deep semantic segmentation, but to produce a useful **prior probability / decision map** that highlights regions likely to belong to a target class. This prior map can then be used alone, visualized, evaluated, or injected as an additional input channel into a CNN / U-Net style segmentation model.

> **SPARTEX** = **SuperPixel-Aware Region TEXton Prior Framework**


## Overview

This framework follows a hybrid image segmentation pipeline:

```text
Input image
   │
   ├── Filter bank responses
   │      └── multi-scale Gaussian, LoG, derivatives
   │
   ├── Texton dictionary
   │      └── MiniBatchKMeans clustering on filter responses
   │
   ├── Superpixel segmentation
   │      └── SLIC regions
   │
   ├── Region feature extraction
   │      ├── LAB color histograms
   │      └── Texton histograms
   │
   ├── Region classification
   │      └── Linear SVM + optional probability calibration
   │
   ├── Model evaluation
   │      └── Accuracy, Balanced Accuracy, Precision, Recall, F1, IoU, ROC-AUC
   │
   └── Target prior map
          └── one score per superpixel region
```

The method is especially useful when the target class can be described using **local texture**, **color distribution**, and **regional structure**, for example:

- shadow prior generation
- cloud / sky region prior generation
- material or surface segmentation
- medical / biological texture segmentation
- defect or anomaly region proposals
- lightweight segmentation preprocessing
- prior map generation before deep learning refinement


## Why this project?

Modern segmentation is often dominated by deep learning, but deep models require large datasets, GPU resources, and careful training. This framework explores a more classical and interpretable alternative:

- no CNN required for the prior map generation
- works with binary masks and images
- can be trained on relatively small datasets
- uses superpixels instead of raw pixel classification
- produces explainable region-level scores
- includes validation metrics for model analysis
- can be combined with deep learning as an auxiliary prior

This makes it interesting for research, prototyping, and low-resource computer vision pipelines.


## Repository structure

```text
.
├── core.py        # Core image processing and feature extraction functions
├── train.py       # Texton dictionary learning + SVM training + evaluation
├── test.py        # Prior map inference and visualization on a test image
└── README.md      # Project documentation
```


## Main components

### 1. Filter bank

The framework first converts the RGB image to grayscale and applies a multi-scale filter bank.

Current implementation uses three scales:

```python
sigmas = [1, 2, 4]
```

For each scale, the following responses are computed:

- Gaussian smoothing
- Laplacian of Gaussian
- first-order derivative in x
- first-order derivative in y
- second-order derivative in x
- second-order derivative in y
- mixed second-order derivative xy

This gives:

```text
3 scales × 7 responses = 21 filter channels
```

The output is a feature map:

```text
(H, W, 21)
```

Each pixel is represented by a local texture / structure descriptor.

> Note: You can adjust the filters to suit your specific needs 


### 2. Texton dictionary

A **texton dictionary** is learned using `MiniBatchKMeans` on sampled filter responses.

To keep memory usage safe, the implementation uses two passes:

1. compute global mean and standard deviation incrementally
2. train `MiniBatchKMeans` using `partial_fit`

The trained texton model stores:

```python
{
    "kmeans": kmeans,
    "mean": mean,
    "std": std,
    "n_textons": n_textons
}
```

Each pixel is then assigned to the closest texton cluster, producing a **texton map**.


### 3. SLIC superpixels

The image is segmented into regions using SLIC superpixels.

Example parameters:

```python
N_SEGMENTS = 800
COMPACTNESS = 8
SIGMA_SLIC = 1
```

For fine structures, higher values such as `800` or `1200` superpixels can be useful. For larger and smoother objects, lower values can be enough.


### 4. Region features

For each superpixel, the framework extracts:

#### LAB color histogram

The RGB image is converted to LAB color space, then histograms are computed for:

- L channel
- a channel
- b channel

#### Texton histogram

The distribution of texton IDs inside the region is also computed.

Final region descriptor:

```text
[LAB histogram] + [Texton histogram]
```

With the default values:

```python
N_BINS_LAB = 21
N_TEXTONS = 16
```

The feature size is:

```text
21 × 3 + 16 = 79 dimensions
```


### 5. SVM prior classifier

A Linear SVM is trained to classify each superpixel as:

```text
0 = non-target region
1 = target region
```

The SVM pipeline uses:

```python
AdditiveChi2Sampler(sample_steps=2)
LinearSVC(C=1.0, class_weight="balanced", max_iter=10000)
```

Optional probability calibration is available using:

```python
CalibratedClassifierCV(method="sigmoid", cv=3)
```

Calibration is slower, but it allows the model to output more meaningful probability values.


### 6. Evaluation metrics

The training script can evaluate the SVM on a validation split before saving the final model.

Example evaluation parameters:

```python
EVALUATE_MODEL = True
VALIDATION_SIZE = 0.2
TRAIN_FINAL_MODEL_ON_ALL_DATA = True
```

The evaluation step computes:

- Accuracy
- Balanced Accuracy
- Precision
- Recall
- F1-score
- IoU / Jaccard score
- Confusion matrix
- ROC-AUC
- Average Precision

The typical training flow is:

```text
1. build the full superpixel dataset
2. split it into train / validation sets
3. train the SVM on the train split
4. evaluate it on the validation split
5. optionally retrain a final model on all available data
6. save the final model
```

When:

```python
TRAIN_FINAL_MODEL_ON_ALL_DATA = True
```

the validation metrics are used only to estimate performance, and then the final SVM is trained again on the complete dataset before being saved.


## Dataset format

The training script expects one image folder and one mask folder.

Example:

```text
dataset/
├── images/
│   ├── img_001.jpg
│   ├── img_002.jpg
│   └── img_003.jpg
│
└── masks/
    ├── img_001.png
    ├── img_002.png
    └── img_003.png
```

Images and masks must have the same base filename.

Accepted mask extensions:

```text
.png, .jpg, .jpeg, .bmp, .tif, .tiff
```

Masks are loaded as grayscale and binarized using:

```python
mask = (mask > 127).astype(np.uint8)
```


## Installation

Create a Python environment and install the required packages:

```bash
pip install numpy opencv-python scipy scikit-image scikit-learn matplotlib joblib
```

Recommended Python version:

```text
Python 3.10+
```


## Training

Open `train.py` and modify the paths:

```python
TRAIN_IMAGES_DIR = r"path/to/images"
TRAIN_MASKS_DIR  = r"path/to/masks"
OUTPUT_DIR       = r"path/to/output/model"
```

Then run:

```bash
python train.py
```

The training process does four main things:

1. collects image / mask pairs
2. learns the texton dictionary
3. builds a superpixel dataset
4. trains and evaluates the SVM classifier

At the end, two models are saved:

```text
texton_kmeans.joblib
target_prior_svm.joblib
```


## Inference / testing

Open `test.py` and modify the paths:

```python
TEST_IMAGE_PATH = r"path/to/test/image.jpg"
SVM_MODEL_PATH = r"path/to/target_prior_svm.joblib"
TEXTON_MODEL_PATH = r"path/to/texton_kmeans.joblib"
OUTPUT_DIR = r"path/to/output"
```

Then run:

```bash
python test.py
```

The script will generate:

- the input image visualization
- the texton map
- the SLIC superpixel map
- the target prior map

The prior map is also saved as:

```text
target_prior_map.png
```


## Output example

The prior map is a floating-point map where each superpixel receives a score from the trained classifier.

Depending on the selected output mode, the score can represent:

```text
decision  -> raw SVM decision score
prob      -> probability or pseudo-probability
logit     -> logit score
```

For visualization, the map is normalized to `[0, 255]` and saved as an image.


## Important parameters

### Superpixel parameters

```python
N_SEGMENTS = 800
COMPACTNESS = 8
SIGMA_SLIC = 1
```

Suggested values:

| Goal | N_SEGMENTS | COMPACTNESS | SIGMA_SLIC |
|---|---:|---:|---:|
| coarse regions | 100 - 300 | 10 - 20 | 1 - 2 |
| general segmentation | 400 - 800 | 8 - 12 | 1 |
| fine structures | 800 - 1200+ | 5 - 8 | 0.5 - 1 |

### Texton parameters

```python
N_TEXTONS = 16
MAX_TOTAL_SAMPLES_FOR_TEXTONS = 150000
KMEANS_BATCH_SIZE = 4096
```

Suggested values:

| Goal | N_TEXTONS |
|---|---:|
| simple textures | 8 - 16 |
| general usage | 16 - 32 |
| complex textures | 32 - 64 |

### SVM parameters

```python
SVM_C = 1.0
USE_CALIBRATED_SVM = False
```

Use calibration when you need probability-like outputs:

```python
USE_CALIBRATED_SVM = True
```

### Evaluation parameters

```python
EVALUATE_MODEL = True
VALIDATION_SIZE = 0.2
TRAIN_FINAL_MODEL_ON_ALL_DATA = True
```

Suggested behavior:

| Goal | Setting |
|---|---|
| quick training only | `EVALUATE_MODEL = False` |
| get validation metrics | `EVALUATE_MODEL = True` |
| save best final-use model | `TRAIN_FINAL_MODEL_ON_ALL_DATA = True` |
| save only validation-trained model | `TRAIN_FINAL_MODEL_ON_ALL_DATA = False` |


## Strengths

- lightweight and interpretable
- no GPU required
- useful with small to medium datasets
- produces region-level prior maps
- includes quantitative evaluation metrics
- can be combined with deep segmentation models
- RAM-safe texton learning using incremental statistics and MiniBatchKMeans


## Limitations

- does not understand high-level semantic context like a deep model
- performance depends strongly on feature quality and dataset consistency
- superpixel quality affects the final prior map
- region-level evaluation is not exactly the same as pixel-level segmentation evaluation
- not ideal for very complex object segmentation without additional features
- probability calibration can be slow on large datasets
- current version is designed for binary target / non-target segmentation


## Example use with a deep model

The generated prior map can be concatenated with the RGB image:

```text
RGB image      -> shape (H, W, 3)
Prior map      -> shape (H, W, 1)
Final input    -> shape (H, W, 4)
```

This can help a neural network focus on regions that are already likely to contain the target class.

## Results

### Cloud dataset

The figure below shows an example result obtained with **SPARTEX-Prior** on a homemade cloud segmentation dataset.

The dataset was built from approximately **1,500 cloud images collected from the web**. The corresponding binary masks were generated using **SAM3 (Segment Anything Model)** and then used to train the superpixel-level SVM classifier.

The example illustrates the complete inference output:

- the original input image
- the learned texton map
- the SLIC superpixel decomposition
- the generated target prior map

![Example result on a homemade cloud dataset](attachment:Figure_4.png)

The prior map highlights image regions that are likely to belong to the target class. In this example, the framework produces a coarse but useful cloud prior map that can be used directly for classical segmentation or as an additional prior channel for a CNN / U-Net refinement stage.


### SBU shadow dataset

SPARTEX-Prior was also tested on an enriched version of the **SBU Shadow Dataset**.

The original dataset was extended with additional images, resulting in a larger shadow segmentation dataset containing approximately **11,500 images**. This enriched dataset was used to train and evaluate the shadow prior model.

The figures below show two example results obtained on the enriched SBU shadow dataset:

![Example result 1 on the enriched SBU shadow dataset](attachment:Figure_5.png)

![Example result 2 on the enriched SBU shadow dataset](attachment:Figure_2.png)

The generated prior maps show that the framework can learn meaningful shadow-related texture and color patterns using only classical computer vision features, superpixels, texton histograms, LAB color histograms, and an SVM classifier.

If you would like to use the enhanced shadow dataset, please feel free to contact me at loannrivoal@gmail.com.

## Author

Created by **Loann KAIKA**.

## License


This project is licensed under the Apache License 2.0.

You are free to use, modify, and distribute this software under the terms of the Apache-2.0 license.

See the [LICENSE](LICENSE) file for more details.
