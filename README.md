<!-- ## Bridging Foundation Models and ASTM Metallurgical Standards for Automated Grain Size Estimation from Microscopy Images

📍 *Accepted at:*  
**Computer Vision for Multimodal Microscopy Image Analysis Workshop (CVPR 2026)**

---

## 🚧 Code Status

The code for this project is currently being finalized and will be released soon.

🔔 Please check back for updates. -->


# ASTM Grain Size Estimator

[![Conference](https://img.shields.io/badge/CVPR_2026-CVMI_Workshop-blue)](https://cvpr.thecvf.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Official repository for the paper: **"Bridging Foundation Models and ASTM Metallurgical Standards for Automated Grain Size Estimation from Microscopy Images"** accepted at the CVMI Workshop, CVPR 2026.

This repository provides a fully automated pipeline for dense instance segmentation and grain size estimation. It adapts **Cellpose-SAM** to microstructures and integrates its topology-aware gradient tracking with an **ASTM E112 Jeffries planimetric module** to directly predict the ASTM E112-25 Grain Size Number ($G$).

---

## 📌 Prerequisites & Dependencies

Our pipeline utilizes a modified Cellpose-SAM architecture and requires several standard Python scientific libraries.

1. **Install the core requirements:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Cellpose-SAM Dependency:** Ensure you have the Cellpose library installed that supports the `"cpsam"` foundation model weights. If you are using a specific fork of Cellpose for SAM integration, clone and install it locally:
   ```bash
   git clone <link-to-your-cellpose-sam-fork>
   cd cellpose
   pip install -e .
   ```

---

## 🚀 Pipeline Overview

The repository is structured into a 4-step modular pipeline to ensure easy replication of our results. Each script is designed to be run sequentially.

### Step 1: Dataset Preparation
This step algorithmically maps and stitches fragmented high-magnification patches into full-field micrographs. It then converts the binary boundary masks into instance-labeled TIFF masks, applying morphological separation to prevent gradient-tracking failure. Finally, it generates CSV files for nested training splits (5%, 10%, 25%, 50%, 75%).

```bash
python 1_prepare_dataset.py \
    --input_patches ./data/raw_patches \
    --input_masks ./data/raw_masks \
    --output_dir ./data/processed
```

### Step 2: Fine-Tuning Cellpose-SAM
Fine-tunes the zero-shot Cellpose-SAM model on the generated splits. Because the smallest splits contain very few images, this script employs an aggressive data augmentation pipeline (geometric distortions, photometric adjustments, and coarse dropout) to ensure robust few-shot scalability and prevent rapid memorization.

```bash
python 2_train.py \
    --data_dir ./data/processed \
    --split split_05_percent \
    --batch_size 2 \
    --epochs 100 \
    --aug_multiplier 30
```

### Step 3: Inference and Segmentation Evaluation
Generates dense segmentation masks for the held-out test set and evaluates standard computer vision metrics. It calculates Average Precision (mAP), Boundary F1 scores, and Grain Count Error for both the zero-shot baseline and your fine-tuned models.

```bash
python 3_inference.py \
    --input_dir ./data/processed/test/images \
    --gt_dir ./data/processed/test/masks \
    --model_path <path-to-fine-tuned-model OR 'cpsam'> \
    --output_dir ./results/inference_output
```

### Step 4: ASTM Jeffries Planimetric Evaluation
Executes the automated Jeffries method to directly calculate the ASTM E112-25 Grain Size Number ($G$). It dynamically calculates the Jeffries multiplier ($f$) based on the physical area of the inscribed test circle.

This script supports two modes:
* **`superimposed`**: Uses the Ground Truth (GT) mask to determine the optimal circle and copies it to the prediction mask (for strict head-to-head benchmarking).
* **`independent`**: Calculates the optimal circle for GT and Prediction autonomously without any shared geometric information (demonstrating the fully autonomous deployment capability).

```bash
python 4_evaluate_jeffries.py \
    --gt_dir ./data/processed/test/masks \
    --pred_dir ./results/inference_output/binary_masks \
    --output_dir ./results/jeffries_eval \
    --mode independent
```

---

## 📊 Results Summary

Our evaluations demonstrate that utilizing just **5% of the training data (2 samples)**, the fine-tuned Cellpose-SAM pipeline successfully maintains topological separation and predicts the ASTM grain size number ($G$) with a Mean Absolute Percentage Error (MAPE) as low as **1.50%**. Robustness testing across varying target grain counts also empirically validates the ASTM 50-grain sampling minimum.

<!-- ---

## 📝 Citation

If you find this code or our paper useful in your research, please consider citing:

```bibtex
@inproceedings{mueez2026bridging,
  title={Bridging Foundation Models and ASTM Metallurgical Standards for Automated Grain Size Estimation from Microscopy Images},
  author={Mueez, Abdul and Vyas, Shruti},
  booktitle={Proceedings of the Computer Vision and Pattern Recognition Conference Workshops (CVMI)},
  year={2026}
}
``` -->