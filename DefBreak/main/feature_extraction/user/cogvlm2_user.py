#!/usr/bin/env python3
"""
Feature extraction for CogVLM2 model (user-level version).
Recursively processes subfolders, extracts logits, calculates perplexity,
and saves probability distributions.
"""

import torch
import torch.nn.functional as F
import argparse
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import numpy as np
import os
from tqdm import tqdm
import glob
import traceback


# ----------------------------- #
#       Model Initialization    #
# ----------------------------- #
def get_model_outputs(model_path, quantization=0, device='cuda'):
    print(f"Loading model from: {model_path}")
    torch_type = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    if quantization == 4:
        model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=torch_type, trust_remote_code=True,
            quantization_config=BitsAndBytesConfig(load_in_4bit=True),
            low_cpu_mem_usage=True
        ).eval()
    elif quantization == 8:
        model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=torch_type, trust_remote_code=True,
            quantization_config=BitsAndBytesConfig(load_in_8bit=True),
            low_cpu_mem_usage=True
        ).eval()
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=torch_type, trust_remote_code=True
        ).eval().to(device)

    print("Model and tokenizer loaded.")
    return model, tokenizer, torch_type


# ----------------------------- #
#       Input Processing        #
# ----------------------------- #
from PIL import Image, ImageOps

def process_inputs_for_generation(model, tokenizer, query, history=None, image=None, device='cuda', torch_type=None):
    if history is None:
        history = []

    if image is not None:
        try:
            image = ImageOps.exif_transpose(image)
            if image.mode != "RGB":
                image = image.convert("RGB")
            if image.width < 10 or image.height < 10:
                raise ValueError(f"Image too small: {image.size}")
        except Exception as e:
            print(f"Invalid image encountered: {e}")
            raise e

        input_by_model = model.build_conversation_input_ids(
            tokenizer,
            query=query,
            history=history,
            images=[image],
            template_version='chat'
        )
    else:
        input_by_model = model.build_conversation_input_ids(
            tokenizer,
            query=query,
            history=history,
            template_version='chat'
        )

    inputs = {
        'input_ids': input_by_model['input_ids'].unsqueeze(0).to(device),
        'token_type_ids': input_by_model['token_type_ids'].unsqueeze(0).to(device),
        'attention_mask': input_by_model['attention_mask'].unsqueeze(0).to(device),
        'images': [[input_by_model['images'][0].to(device).to(torch_type)]] if image is not None else None,
    }
    return inputs, input_by_model


# ----------------------------- #
#       Perplexity Calc         #
# ----------------------------- #
def calculate_self_perplexity(scores, generated_ids, min_logit_threshold=None):
    log_probs = []
    num_generated_tokens = len(scores)
    if num_generated_tokens == 0:
        return 1.0

    prompt_len = generated_ids.shape[1] - num_generated_tokens

    for i in range(num_generated_tokens):
        step_logits = scores[i]
        if step_logits.ndim == 3:
            step_logits = step_logits[:, -1, :]
        step_logits = step_logits[0].float()

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
#       Probability Save        #
# ----------------------------- #
def save_generated_distribution_and_perplexity(
    model,
    tokenizer,
    inputs,
    save_path="probabilities_mean.txt",
    min_logit_threshold=-10.0,
    top_k_features=120000,
):
    try:
        image_tensor = None
        if "images" in inputs and inputs["images"] is not None:
            image_tensor = inputs["images"][0][0]
        else:
            print(f"No image found for {save_path}")
            return None

        device = next(model.parameters()).device
        for k in ["input_ids", "token_type_ids", "attention_mask"]:
            if k in inputs and isinstance(inputs[k], torch.Tensor):
                inputs[k] = inputs[k].to(dtype=torch.long, device=device)

        try:
            vision_module = getattr(getattr(model, "model", model), "vision")
            vision_dtype = next(vision_module.parameters()).dtype
        except Exception:
            vision_dtype = next(model.parameters()).dtype

        image_tensor = image_tensor.to(device=device, dtype=vision_dtype)

        with torch.no_grad():
            outputs = model(
                **{k: v for k, v in inputs.items() if k != "images"},
                images=[[image_tensor]],
                output_hidden_states=False,
                output_attentions=False,
                return_dict=True,
            )

        logits = outputs.logits[0].float().cpu()
        probs = torch.softmax(logits, dim=-1)
        mean_probs = probs.mean(dim=0).numpy()

        feature_probs = np.zeros(top_k_features)
        take = min(len(mean_probs), top_k_features)
        feature_probs[:take] = mean_probs[:take]

        entropy = -np.sum(mean_probs * np.log(mean_probs + 1e-12))
        perplexity = np.exp(entropy / len(mean_probs))

        combined = np.append(feature_probs, perplexity)
        np.savetxt(save_path, combined, fmt="%.10f")

        return {
            "mean_probabilities_feature": feature_probs,
            "perplexity": perplexity,
            "num_generated": logits.shape[0],
        }

    except Exception as e:
        print(f"Error during forward() for {save_path}: {e}")
        traceback.print_exc()
        return None


# ----------------------------- #
#       Single Image Process    #
# ----------------------------- #
def process_image(image_path, model, tokenizer, img_num, output_dir, device='cuda', torch_type=None):
    try:
        image = Image.open(image_path).convert("RGB")
    except Exception:
        return False

        return False

    query = "describe the image."
    try:
        inputs, _ = process_inputs_for_generation(model, tokenizer, query, image=image, device=device, torch_type=torch_type)
    except Exception:
        traceback.print_exc()
        return False

    output_path = os.path.join(output_dir, f"file_{img_num}.txt")
    result = save_generated_distribution_and_perplexity(model, tokenizer, inputs, output_path)

    if result is None:
        return False

    summary_path = os.path.join(output_dir, "perplexity_summary.csv")
    if not os.path.exists(summary_path):
        with open(summary_path, "w") as f:
            f.write("image_name,self_perplexity,num_generated_tokens\n")
    with open(summary_path, "a") as f:
        image_name = os.path.basename(image_path)
        f.write(f"{image_name},{result['perplexity']:.10f},{result['num_generated']}\n")
    return True


# ----------------------------- #
#   Recursive Directory Logic   #
# ----------------------------- #
def process_directory(base_input_dir, base_output_dir, split, condition, model, tokenizer, device='cuda', torch_type=None):
    input_dir = os.path.join(base_input_dir, split, condition)
    output_dir = os.path.join(base_output_dir, split, condition)

    if not os.path.isdir(input_dir):
        print(f"Input directory not found: {input_dir}")
        return 0

    successful = 0
    image_patterns = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp"]

    for root, _, _ in os.walk(input_dir):
        rel_path = os.path.relpath(root, input_dir)
        save_dir = os.path.join(output_dir, rel_path)
        os.makedirs(save_dir, exist_ok=True)

        image_paths = []
        for pattern in image_patterns:
            image_paths.extend(glob.glob(os.path.join(root, pattern)))
        if not image_paths:
            continue

        print(f"\nProcessing {split}/{condition}/{rel_path} ({len(image_paths)} images)")
        for img_num, image_path in tqdm(
            enumerate(image_paths), total=len(image_paths), desc=f"{split}/{condition}/{rel_path}"
        ):
            if process_image(image_path, model, tokenizer, img_num, save_dir, torch_type):
                successful += 1

    print(f"Finished {split}/{condition}: {successful} images processed.")
    return successful


# ----------------------------- #
#              Main             #
# ----------------------------- #
def main():
    parser = argparse.ArgumentParser(description="CogVLM2 Feature Extraction (user-level recursive version)")
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--quant', type=int, choices=[0, 4, 8], default=0)
    parser.add_argument('--input_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    model, tokenizer, torch_type = get_model_outputs(args.model_path, args.quant, device)
    splits = ["train", "test"]
    conditions = ["finetuned", "unfinetuned"]

    total_processed = 0
    for split in splits:
        for condition in conditions:
            total_processed += process_directory(args.input_dir, args.output_dir, split, condition, model, tokenizer, device, torch_type)

    print("\nAll processing complete")
    print(f"Total images processed: {total_processed}")
    print(f"Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
