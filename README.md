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

* **Rich feature space**

  * Top-K probability projection (over vocabulary or visual tokens).
  * Entropy and Rényi-style uncertainty measures.
  * Aggregated statistics at both sample and user levels.

* **Model-agnostic design**

  * Applicable to various VLM backbones (e.g., MiniGPT-4, CogVLM2, DeepSeek-VL2, Qwen2-VL) with a unified inference interface.

* **Reproducible evaluation**

  * Metrics: accuracy, precision, AUC.

---

## 3. Repository Structure

The intended organization of the codebase:

```
DefBreak
├── ablation
├── dataset
│   ├── sample
│   │   ├── SA1B
│   │   ├── cc_sbu_align
│   │   ├── coco
│   │   ├── fashion
│   │   ├── flickr_8k
│   │   └── tti
│   └── user
│       ├── place365
│       └── prid
├── main
│   ├── classifier
│   │   ├── sample
│   │   │   ├── test_cnn.py
│   │   │   └── train_cnn.py
│   │   └── user
│   │       └── mlp.py
│   └── feature_extraction
│       ├── sample
│       │   ├── cogvlm2_generate.py
│       │   ├── deepseek_generate_perplexity.py
│       │   ├── minigpt_sample.py
│       │   └── qwen2vl_generate_perplexity.py
│       └── user
│           ├── cogvlm2_user.py
│           ├── deepseek_user.py
│           ├── minigpt_user.py
│           └── qwen2vl_user.py
└── models
    ├── cogvlm2-llama3-chat-19B
    ├── deepseek-vl2-tiny
    └── qwen2vl-2B-instruct

```

---

## 4. Installation

During attacks, please activate the environment corresponding to the target model and follow the official configuration files for that model.

The CNN/MLP training scripts (e.g., train_cnn.py, train_mlp.py) depend on a dedicated cnn environment.
Please create the environment using the provided cnn/requirements.txt to ensure full reproducibility of our experiments.

```
conda create -n cnn python=3.10
conda activate cnn
pip install -r cnn/requirements.txt
```


### Attack Target Models

The DefBreak implementation and experiments target commonly used VLM backbones. The following model resources are used as attack targets (links point to the upstream model repos/pages):

MiniGPT-4 — https://github.com/Vision-CAIR/MiniGPT-4/

Qwen2-VL (example instruct checkpoint) — https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct

CogVLM2 — https://github.com/zai-org/CogVLM2

DeepSeek-VL2 (tiny) — https://huggingface.co/deepseek-ai/deepseek-vl2-tiny

### Setup

```bash
git clone https://github.com/ensthee/DefBreak.git
cd DefBreak
```
---

## 5. Dataset Preparation

The project evaluates both sample-level and user-level datasets. All datasets are publicly available and listed below.

Sample-Level Datasets
1. CC-SBU Align

Official link:
https://github.com/Vision-CAIR/MiniGPT-4/tree/main/dataset

2. Flickr8k

Kaggle:
https://www.kaggle.com/datasets/adityajn105/flickr8k

3. COCO 2014

MS COCO official site:
https://cocodataset.org/#home

4. SA-1B (Segment Anything Dataset)

Meta AI:
https://ai.meta.com/datasets/segment-anything/

5.Fashion Image Caption-3500 (HuggingFace)

https://huggingface.co/datasets/GHonem/fashion_image_caption-3500

User-Level Datasets
1. PRID-2011

Person Re-ID dataset (camera A/B):
https://www.tugraz.at/institute/icg/research/team-bischof/learning-recognition-surveillance/downloads/prid11/

2. Places365

Scene understanding dataset:
https://github.com/CSAILVision/places365



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

To fine-tune MiniGPT4, we use a lightweight caption dataset stored in a file named filter_cap.json

This format stores a list of image–caption pairs:

```
{
  "annotations": [
    {
      "image_id": "sample_image_0001",
      "caption": "A person is walking outdoors."
    },
    {
      "image_id": "sample_image_0002",
      "caption": "A person is walking outdoors."
    }
  ]
}
```
Instruction-tuning Format (Swift / LoRA / custom SFT)

When used for instruction-based fine-tuning, the dataset follows this structure:

```
{
  "query": "<ImageHere> Describe this image.",
  "response": "A person is walking outdoors.",
  "images": [
    "path/to/image.png"
  ]
}
```
---

## 6. Model Preparation

DefBreak is model-agnostic provided the VLM can produce logits/probabilities or text outputs for a given image (and optional prompt). Supported model families (examples):
### MiniGPT-4
`torchrun --nproc-per-node NUM_GPU train.py --cfg-path train_configs/minigptv2_finetune.yaml`

### DeepSeek-VL2 (full fine-tune)
```
CUDA_VISIBLE_DEVICES=2 \
swift sft \
  --model /PATH/TO/models/deepseek-vl2-tiny \
  --model_type deepseek_vl2 \
  --train_type full \
  --dataset /PATH/TO/datasets/cc_sbu_align/data.jsonl \
  --learning_rate 1e-5 \
  --output_dir /PATH/TO/finetuned_models/deepseek \
  --num_train_epochs 1 \
  --save_total_limit -1 \
  --save_strategy epoch
```

### Qwen2-VL (full fine-tune)
```
CUDA_VISIBLE_DEVICES=3 \
swift sft \
  --model /PATH/TO/models/qwen2vl-2B-instruct \
  --model_type qwen2_vl \
  --train_type full \
  --dataset /PATH/TO/datasets/coco2014_mini/data.jsonl \
  --learning_rate 1e-5 \
  --output_dir /PATH/TO/finetuned_models/qwen2 \
  --num_train_epochs 1 \
  --save_total_limit -1 \
  --save_strategy epoch
```

### CogVLM2 (full fine-tune)
```
CUDA_VISIBLE_DEVICES=3 \
swift sft \
  --model /PATH/TO/models/cogvlm2-llama3-chat-19B \
  --model_type cogvlm2 \
  --train_type full \
  --dataset /PATH/TO/datasets/user-level/prid_seperate/data.jsonl \
  --learning_rate 1e-4 \
  --output_dir /PATH/TO/finetuned_models/cogvlm2 \
  --num_train_epochs 1 \
  --save_total_limit -1 \
  --save_strategy epoch
```
Fine-tuning (full or adapter/LoRA) should be performed externally. Optionally merge adapters into a single checkpoint before running feature extraction:

```bash
swift export --merge_lora true \
  --ckpt_dir /path/to/lora_ckpt \
  --output /path/to/merged_model
```
---

## 7. Running DefBreak

### 7.1 Sample-level Feature extraction

Run MiniGPT-4 feature extraction
```
python minigpt_sample.py \
  --cfg-path configs/minigpt4_eval.yaml \
  --base-input-dir path/to/dataset_root \
  --base-output-dir path/to/output_features \
  --splits train test \
  --conditions finetuned \
  --gpu-id 0 \
  --max-new-tokens 1024 \
  --top-k-features 120000 \
  --temperature 0.2
```
Run CogVLM2 feature extraction
```
python cogvlm2_sample.py \
  --model_path THUDM/cogvlm2-llama3-chat-19B \
  --quant 0 \
  --input_dir path/to/dataset_root \
  --output_dir path/to/output_features \
```
Run DeepSeek-VL2 feature extraction
```
python deepseek_sample.py
```
Run Qwen2-VL feature extraction
```
python qwen2vl_sample.py
```
For Qwen2-VL and DeepSeek-VL2,
Before running, edit the configuration section in your script if needed:

```
BASE_INPUT_DIR = "path/to/sample_level_dataset"         
BASE_OUTPUT_DIR = "path/to/output_features"            
MODEL_PATH = "path/to/qwen2vl_checkpoint"             
USE_FLASH_ATTENTION = False

SPLITS = ["test", "train"]
CONDITIONS = ["finetuned", "unfinetuned"]
```

### 7.2 Sample-level attack
Run CNN feature classifier training
```bash
python train_cnn.py \
  --train_data path/to/feature_dir \
  --model_path tti_best_cnn_model.pth \
  --batch_size 32 \
  --epochs 100 \
  --learning_rate 3e-4
```
### 7.3 User-level Feature extraction

Run MiniGPT-4 feature extraction
```
python minigpt_user.py \
  --cfg-path configs/minigpt4_eval.yaml \
  --base-input-dir path/to/dataset_root \
  --base-output-dir path/to/output_features \
  --splits train test \
  --conditions finetuned \
  --gpu-id 0 \
  --max-new-tokens 1024 \
  --top-k-features 120000 \
  --temperature 0.2
```
Run CogVLM2 feature extraction
```
python cogvlm2_user.py \
  --model_path THUDM/cogvlm2-llama3-chat-19B \
  --quant 0 \
  --input_dir path/to/dataset_root \
  --output_dir path/to/output_features \
```
Run DeepSeek-VL2 feature extraction
```
python deepseek_user.py

```
Run Qwen2-VL feature extraction
```
python qwen2vl_user.py

```
Before running, edit the configuration section in your script if needed:
```
BASE_INPUT_DIR = "path/to/user_level_dataset"         
BASE_OUTPUT_DIR = "path/to/output_features"           
MODEL_PATH = "path/to/qwen2vl_checkpoint"             
USE_FLASH_ATTENTION = False
SPLITS = ["test", "train"]
CONDITIONS = ["finetuned", "unfinetuned"]
```
### 7.4 User-level attack
Run MLP feature classifier training
```bash
python train_cnn.py \
  --train_data path/to/feature_dir \
  --model_path best_model.pth \
  --batch_size 32 \
  --epochs 100 \
  --learning_rate 3e-4
```
---

## 8. Contact & Collaboration

For questions, suggestions, or collaboration opportunities, please open an issue in the repository:

https://github.com/ensthee/DefBreak

We welcome contributions from the community.

---
