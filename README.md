# DefBreak

**DefBreak: Breaking Hardened Defenses with Adversarial Membership Inference on Vision–Language Models**

> Code repository for the DefBreak framework.
> DefBreak provides a unified, reproducible implementation of adversarial membership inference attacks (MIAs) against modern vision–language models (VLMs), covering both **sample-level** and **user-level** privacy leakage.

---

## 1. Overview

Modern VLMs are frequently fine-tuned on user-generated multimodal data. **DefBreak** systematically studies how such fine-tuning exposes training data to **membership inference attacks (MIAs)**.

This repository provides:

* A **generic pipeline** for attacking VLMs under realistic black-box or limited white-box access.
* Support for both **sample-level** and **user-level** membership inference.
* Reproducible experiment scripts and analysis tools corresponding to the DefBreak paper.

---

## 2. Key Features

* **Dual attack targets**

  * **Sample-level MIA**: determine whether a specific image was used in fine-tuning.
  * **User-level MIA**: determine whether images from a specific user were used in fine-tuning.

* **Flexible access assumptions**

  * Designed for **black-box** or **limited white-box** VLM access.
  * Uses model outputs (logits / probabilities / generated text) rather than requiring full parameters.

* **Rich feature space**

  * Top-K probability projection (over vocabulary or visual tokens).
  * Entropy and Rényi-style uncertainty measures.
  * Aggregated statistics at both sample and user levels.

* **Model-agnostic design**

  * Applicable to various VLM backbones (e.g., MiniGPT-4, CogVLM2, DeepSeek-VL2, Qwen2-VL) with a unified inference interface.

* **Reproducible evaluation**

  * Metrics: accuracy, precision, recall, F1, ROC, AUC.
  * LaTeX-ready tables and figure data for easy integration into papers.

---

## 3. Repository Structure

The intended organization of the codebase:

```
DefBreak/
├─ README.md                 # This file
├─ LICENSE                   # Project license
├─ requirements.txt          # Python dependencies
├─ CITATION.bib              # BibTeX entry for the DefBreak paper
│
├─ configs/
│  ├─ datasets.yaml          # Dataset paths and configuration
│  └─ experiments.yaml       # Model/dataset/attack settings per experiment
│
├─ src/
│  ├─ feature_extraction.py  # Run VLMs to extract DefBreak features
│  ├─ attack_pipeline.py     # Train/evaluate sample-level attacks
│  ├─ user_aggregation.py    # Build user-level features & attacks
│  ├─ evaluation.py          # Metrics, ROC/AUC, table generation
│  ├─ ablation.py            # Ablation and sensitivity studies
│  └─ utils.py               # Shared helpers (I/O, logging, seeding, etc.)
│
├─ scripts/
│  ├─ prepare_dataset.sh
│  ├─ run_feature_extraction.sh
│  ├─ run_attack.sh
│  └─ reproduce_table_results.sh
│
└─ experiments/
   ├─ features/
   ├─ attacks/
   ├─ eval/
   └─ tables/
```

---

## 4. Installation

### Requirements

* Python ≥ 3.8
* CUDA-enabled GPU (recommended for inference at scale)
* PyTorch ≥ 1.13
* `transformers` for model interfaces
* `modelscope` if using ModelScope models
* `numpy`, `pandas`, `scikit-learn`, `tqdm`, `matplotlib`

### Setup

```bash
git clone https://github.com/ensthee/DefBreak.git
cd DefBreak

python -m venv venv
source venv/bin/activate          # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Example `requirements.txt`:

```
torch>=1.13
transformers>=4.28
modelscope>=1.4
scikit-learn>=1.1
numpy>=1.23
pandas>=1.5
tqdm
matplotlib
```

---

## 5. Dataset Preparation

DefBreak works with a unified dataset format. Typical dataset categories used in experiments:

* Captioned image datasets (COCO, Flickr)
* Scene or object datasets (Places)
* User-centric collections (PRID, album-style datasets)
* Large-scale collections for user-level analysis

### Directory layout

```
DATA_ROOT/
  └─ DATASET_NAME/
     ├─ train/
     │  ├─ finetuned/      # images used in fine-tuning (members)
     │  └─ unfinetuned/    # images not used in fine-tuning (non-members)
     └─ test/
        ├─ finetuned/
        └─ unfinetuned/
```

### JSONL metadata

Each dataset should have a `data.jsonl` where each line is a JSON object:

```json
{
  "images": ["/path/to/image.jpg"],
  "query": "",
  "response": "a person walking on the street",
  "member": true,
  "user_id": "user_001"
}
```

Minimum fields:

* `images` — list of image paths
* `member` — boolean
* `user_id` — identifier for user aggregation (optional for sample-only experiments)

Use `scripts/prepare_dataset.sh` to scan folders and produce `data.jsonl`.

---

## 6. Model Preparation

DefBreak is model-agnostic provided the VLM can produce logits/probabilities or text outputs for a given image (and optional prompt). Supported model families (examples):

* MiniGPT-style
* CogVLM2-style
* DeepSeek-VL2-style
* Qwen2-VL-style

Fine-tuning (full or adapter/LoRA) should be performed externally. Optionally merge adapters into a single checkpoint before running feature extraction:

```bash
swift export --merge_lora true \
  --ckpt_dir /path/to/lora_ckpt \
  --output /path/to/merged_model
```

Configure model and dataset paths in `configs/experiments.yaml` and `configs/datasets.yaml`.

---

## 7. Running DefBreak

### 7.1 Feature extraction

```bash
python src/feature_extraction.py \
  --model_path /path/to/model \
  --dataset_jsonl /path/to/data.jsonl \
  --output_dir experiments/features/DATASET/MODEL/ \
  --topk 50
```

Typical features:

* Top-K probability vectors
* Entropy / Rényi scores
* Per-token / per-class statistics

### 7.2 Sample-level attack

```bash
python src/attack_pipeline.py \
  --features_dir experiments/features/DATASET/MODEL/ \
  --attack_type sample \
  --out_dir experiments/attacks/sample/DATASET/MODEL/
```

### 7.3 User-level attack

Aggregate features per user:

```bash
python src/user_aggregation.py \
  --features_dir experiments/features/DATASET/MODEL/ \
  --user_map configs/user_map.json \
  --agg_method mean \
  --out_dir experiments/features_user/DATASET/MODEL/
```

Run attack on aggregated features:

```bash
python src/attack_pipeline.py \
  --features_dir experiments/features_user/DATASET/MODEL/ \
  --attack_type user \
  --out_dir experiments/attacks/user/DATASET/MODEL/
```

### 7.4 Evaluation

```bash
python src/evaluation.py \
  --preds experiments/attacks/sample/DATASET/MODEL/preds.csv \
  --labels experiments/attacks/sample/DATASET/MODEL/labels.csv \
  --out_dir experiments/eval/sample/DATASET/MODEL/
```

Generated outputs:

* `metrics.json` (ACC / PREC / REC / F1 / AUC)
* `roc.png` (ROC curve)
* `result_table.tex` (LaTeX table row)

---

## 8. Reproducing Paper Results

Wrapper scripts are provided:

```bash
bash scripts/run_feature_extraction.sh
bash scripts/run_attack.sh
bash scripts/reproduce_table_results.sh
```

These iterate over model × dataset × attack configurations specified in `configs/experiments.yaml` and store outputs under `experiments/`.

---

## 9. Relation to Existing VLM MIAs

DefBreak unifies multiple signal types (top-K projection, uncertainty measures, aggregated statistics) and explicitly supports user-level inference. The repository includes scripts to compare DefBreak to representative baselines (VL-MIA, SPV-MIA, MaxRényi, etc.) and to generate comparison tables.

---

## 10. Citation

If you use DefBreak, please cite:

```bibtex
@article{defbreak,
  title   = {DefBreak: Breaking Hardened Defenses with Adversarial Membership Inference on Vision--Language Models},
  author  = {Kang, Hanwen and Li, Haozhe and Yao, Zhongjiang and others},
  journal = {Preprint},
  year    = {2025},
  url     = {https://github.com/ensthee/DefBreak}
}
```

Update `CITATION.bib` with final publication metadata when available.

---

## 11. License

This project is released under an open-source license. See `LICENSE` for exact terms.

---

## 12. Contributing & Contact

Contributions welcome:

* Open issues for bugs / reproducibility problems.
* Submit pull requests for new model adapters, dataset parsers, or evaluation features.
* Keep PRs focused and include small runnable examples when possible.

For questions or collaboration, open an issue in the repository:
[https://github.com/ensthee/DefBreak](https://github.com/ensthee/DefBreak)

---
