#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Feature extraction for DeepSeek VL2 model (user-level version).
Recursively processes subfolders, extracts logits, calculates self-perplexity,
and saves averaged probability distributions.
"""

import torch
import torch.nn.functional as F
from modelscope import AutoModelForCausalLM
from deepseek_vl2.models import DeepseekVLV2Processor
from deepseek_vl2.utils.io import load_pil_images
from PIL import Image, ImageOps
import numpy as np
import os
from tqdm import tqdm
import glob
import traceback


# ----------------------------- #
#      Model Initialization     #
# ----------------------------- #
def get_model_outputs(model_path, device="cuda"):
    print(f"Loading model from: {model_path}")
    processor = DeepseekVLV2Processor.from_pretrained(model_path)
    tokenizer = processor.tokenizer

    model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)
    model = model.to(torch.bfloat16).cuda().eval()

    print("Model and processor loaded.")
    return model, processor, tokenizer


# ----------------------------- #
#        Input Processing       #
# ----------------------------- #
def process_inputs_for_generation(model, processor, tokenizer, image_path):
    conversation = [
        {
            "role": "<|User|>",
            "content": "<image>\nDescribe the image.",
            "images": [image_path],
        },
        {"role": "<|Assistant|>", "content": ""},
    ]

    try:
        image = Image.open(image_path)
        image = ImageOps.exif_transpose(image)
        if image.mode != "RGB":
            image = image.convert("RGB")
        if image.width < 10 or image.height < 10:
            raise ValueError(f"Invalid image size: {image.size}")
    except Exception as e:
        print(f"Skipping invalid image {image_path}: {e}")
        raise e

    pil_images = load_pil_images(conversation)

    prepare_inputs = processor(
        conversations=conversation,
        images=pil_images,
        force_batchify=True,
        system_prompt=""
    ).to(model.device)

    return prepare_inputs


# ----------------------------- #
#     Perplexity Calculation    #
# ----------------------------- #
def calculate_self_perplexity(scores, generated_ids, min_logit_threshold=None):
    log_probs = []
    num_generated_tokens = len(scores)
    if num_generated_tokens == 0:
        return 1.0

    prompt_len = generated_ids.shape[1] - num_generated_tokens

    for i in range(num_generated_tokens):
        step_logits = scores[i][0].float()
        if min_logit_threshold is not None:
            step_logits[step_logits < min_logit_threshold] = min_logit_threshold

        step_log_softmax = F.log_softmax(step_logits, dim=-1)
        actual_token_id = generated_ids[0, prompt_len + i].item()
        log_prob = step_log_softmax[actual_token_id].item()
        log_probs.append(max(log_prob, -20.0))

    avg_neg_log_prob = -sum(log_probs) / len(log_probs)
    perplexity = np.exp(avg_neg_log_prob)
    if np.isinf(perplexity) or np.isnan(perplexity):
        return 1e10
    return perplexity


# ----------------------------- #
#   Probability + Feature Save  #
# ----------------------------- #
def save_generated_distribution_and_perplexity(
    model,
    processor,
    tokenizer,
    inputs,
    save_path,
    max_new_tokens=512,
    min_logit_threshold=-10.0,
    top_k_features=120000,
):
    model.eval()
    with torch.no_grad():
        try:
            inputs_embeds = model.prepare_inputs_embeds(**inputs)
            outputs = model.language.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=inputs.attention_mask,
                pad_token_id=tokenizer.eos_token_id,
                bos_token_id=tokenizer.bos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                max_new_tokens=max_new_tokens,
                output_scores=True,
                return_dict_in_generate=True,
                do_sample=False,
                use_cache=False,
            )
        except Exception as e:
            print(f"Error during model.generate for {save_path}: {e}")
            traceback.print_exc()
            return None

    scores = outputs.get("scores", None)
    generated_ids = outputs.get("sequences", None)
    if scores is None or generated_ids is None:
        print(f"Missing scores/sequences for {save_path}")
        return None

    perplexity = calculate_self_perplexity(scores, generated_ids, min_logit_threshold)

    if not scores:
        vocab_size = len(tokenizer)
        mean_probs_np = np.zeros(vocab_size)
        num_generated = 0
    else:
        num_generated = len(scores)
        all_probs = []
        vocab_size = scores[0].shape[-1]
        for step_logits in scores:
            step_logits_processed = step_logits[0].float().cpu()
            if min_logit_threshold is not None:
                step_logits_processed[step_logits_processed < min_logit_threshold] = min_logit_threshold
            step_probs = torch.softmax(step_logits_processed, dim=0)
            all_probs.append(step_probs)
        mean_probs_np = torch.mean(torch.stack(all_probs), dim=0).numpy() if all_probs else np.zeros(vocab_size)

    feature_probs = np.zeros(top_k_features)
    feature_probs[:min(len(mean_probs_np), top_k_features)] = mean_probs_np[:top_k_features]
    combined_features = np.append(feature_probs, perplexity)

    np.savetxt(save_path, combined_features, fmt="%.10f")

    return {
        "mean_probabilities_feature": feature_probs,
        "perplexity": perplexity,
        "num_generated": num_generated,
    }


# ----------------------------- #
#         Image Process         #
# ----------------------------- #
def process_image(image_path, model, processor, tokenizer, img_num, output_dir):
    try:
        if not os.path.exists(image_path):
            return False

        inputs = process_inputs_for_generation(model, processor, tokenizer, image_path)
        output_path = os.path.join(output_dir, f"file_{img_num}.txt")
        result = save_generated_distribution_and_perplexity(model, processor, tokenizer, inputs, output_path)
        if result is None:
            return False

        summary_path = os.path.join(output_dir, "perplexity_summary.csv")
        if not os.path.exists(summary_path):
            with open(summary_path, "w") as f:
                f.write("image_name,self_perplexity,num_generated_tokens\n")
        with open(summary_path, "a") as f:
            img_name = os.path.basename(image_path)
            f.write(f"{img_name},{result['perplexity']:.10f},{result['num_generated']}\n")

        return True
    except Exception:
        traceback.print_exc()
        return False


# ----------------------------- #
#    Recursive Directory Walk   #
# ----------------------------- #
def process_directory(base_input_dir, base_output_dir, split, condition, model, processor, tokenizer):
    input_dir = os.path.join(base_input_dir, split, condition)
    output_dir = os.path.join(base_output_dir, split, condition)
    if not os.path.isdir(input_dir):
        return 0

    successful = 0
    patterns = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp"]

    for root, _, _ in os.walk(input_dir):
        rel_path = os.path.relpath(root, input_dir)
        save_dir = os.path.join(output_dir, rel_path)
        os.makedirs(save_dir, exist_ok=True)

        img_paths = []
        for p in patterns:
            img_paths.extend(glob.glob(os.path.join(root, p)))
        if not img_paths:
            continue

        print(f"Processing {split}/{condition}/{rel_path} ({len(img_paths)} images)")
        for i, img in tqdm(enumerate(img_paths), total=len(img_paths), desc=f"{split}/{condition}/{rel_path}"):
            if process_image(img, model, processor, tokenizer, i, save_dir):
                successful += 1

    print(f"Finished {split}/{condition}: {successful} images processed.")
    return successful


# ----------------------------- #
#              Main             #
# ----------------------------- #
def main():
    BASE_INPUT_DIR = "/path/to/input"
    BASE_OUTPUT_DIR = "/path/to/output"
    MODEL_PATH = "/path/to/model"

    SPLITS = ["train", "test"]
    CONDITIONS = ["finetuned", "unfinetuned"]

    print("Initializing DeepSeek VL2...")
    model, processor, tokenizer = get_model_outputs(MODEL_PATH)

    total = 0
    for s in SPLITS:
        for c in CONDITIONS:
            total += process_directory(BASE_INPUT_DIR, BASE_OUTPUT_DIR, s, c, model, processor, tokenizer)

    print("All processing complete!")
    print(f"Total processed images: {total}")
    print(f"Results saved in: {BASE_OUTPUT_DIR}")


if __name__ == "__main__":
    main()
