"""
Feature extraction for model.
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

def get_model_outputs(
    model_path="MODEL_PATH",
    quantization=0,
    device='cuda',
):
    torch_type = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    if quantization == 4:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch_type,
            trust_remote_code=True,
            quantization_config=BitsAndBytesConfig(load_in_4bit=True),
            low_cpu_mem_usage=True
        ).eval()
    elif quantization == 8:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch_type,
            trust_remote_code=True,
            quantization_config=BitsAndBytesConfig(load_in_8bit=True),
            low_cpu_mem_usage=True
        ).eval()
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch_type,
            trust_remote_code=True
        ).eval().to(device)

    return model, tokenizer, torch_type

def process_inputs_for_generation(model, tokenizer, query, history=None, image=None, device='cuda', torch_type=None):
    if history is None:
        history = []
    if image is None:
        input_by_model = model.build_conversation_input_ids(
            tokenizer,
            query=query,
            history=history,
            template_version='chat'
        )
    else:
        input_by_model = model.build_conversation_input_ids(
            tokenizer,
            query=query,
            history=history,
            images=[image],
            template_version='chat'
        )

    inputs = {
        'input_ids': input_by_model['input_ids'].unsqueeze(0).to(device),
        'token_type_ids': input_by_model['token_type_ids'].unsqueeze(0).to(device),
        'attention_mask': input_by_model['attention_mask'].unsqueeze(0).to(device),
        'images': [[input_by_model['images'][0].to(device).to(torch_type)]] if image is not None else None,
    }
    return inputs, input_by_model

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
        MIN_LOG_PROB = -20.0
        log_prob = max(log_prob, MIN_LOG_PROB)
        log_probs.append(log_prob)

    if not log_probs:
        return 1.0
    avg_neg_log_prob = -sum(log_probs) / len(log_probs)
    perplexity = np.exp(avg_neg_log_prob)
    if np.isinf(perplexity) or np.isnan(perplexity):
        return 1e10
    return perplexity

def save_generated_distribution_and_perplexity(
    model,
    tokenizer,
    inputs,
    save_path="probabilities_mean.txt",
    max_new_tokens=1024,
    min_logit_threshold=-10.0,
    min_prob_threshold=1e-6,
    top_k_features=120000,
    device='cuda',
    pad_token_id=128002
):
    model.eval()
    with torch.no_grad():
        try:
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                output_scores=True,
                return_dict_in_generate=True,
                do_sample=False,
                pad_token_id=pad_token_id,
            )
        except Exception:
            traceback.print_exc()
            return None

    scores = outputs.get("scores", None)
    generated_ids = outputs.get("sequences", None)
    if scores is None or generated_ids is None:
        return None

    perplexity = calculate_self_perplexity(scores, generated_ids, min_logit_threshold)

    if not scores:
        vocab_size = model.config.vocab_size
        mean_probs_np = np.zeros(vocab_size)
        num_generated = 0
    else:
        num_generated = len(scores)
        all_step_probabilities = []
        vocab_size = scores[0].shape[-1]

        for step_logits in scores:
            step_logits_processed = step_logits[0].float().cpu()
            if min_logit_threshold is not None:
                step_logits_processed[step_logits_processed < min_logit_threshold] = min_logit_threshold
            step_probs = torch.softmax(step_logits_processed, dim=0)
            all_step_probabilities.append(step_probs)

        if all_step_probabilities:
            mean_probabilities = torch.mean(torch.stack(all_step_probabilities), dim=0)
            mean_probs_np = mean_probabilities.numpy()
        else:
            mean_probs_np = np.zeros(vocab_size)

    current_vocab_size = len(mean_probs_np)
    if current_vocab_size >= top_k_features:
        feature_probs = mean_probs_np[:top_k_features]
    else:
        feature_probs = np.zeros(top_k_features)
        feature_probs[:current_vocab_size] = mean_probs_np

    combined_features = np.append(feature_probs, perplexity)

    try:
        np.savetxt(save_path, combined_features, fmt="%.10f")
    except Exception:
        traceback.print_exc()
        return None

    return {
        "mean_probabilities_feature": feature_probs,
        "perplexity": perplexity,
        "num_generated": num_generated,
        "save_path": save_path,
    }

def process_image(image_path, model, tokenizer, img_num, output_dir, device='cuda', torch_type=None):
    try:
        image = Image.open(image_path).convert("RGB")
    except Exception:
        return False

    query = "describe the image."

    try:
        inputs, input_by_model = process_inputs_for_generation(
            model, tokenizer, query, history=[], image=image,
            device=device, torch_type=torch_type
        )
    except Exception:
        traceback.print_exc()
        return False

    output_path = os.path.join(output_dir, f"file_{img_num}.txt")
    result = save_generated_distribution_and_perplexity(
        model,
        tokenizer,
        inputs,
        output_path,
        max_new_tokens=1024,
        top_k_features=120000,
        device=device,
        pad_token_id=128002
    )

    if result is None:
        return False

    perplexity_summary_path = os.path.join(output_dir, "perplexity_summary.csv")
    image_name = os.path.basename(image_path)

    if not os.path.exists(perplexity_summary_path):
        try:
            with open(perplexity_summary_path, "w") as f:
                f.write("image_name,self_perplexity,num_generated_tokens\n")
        except Exception:
            pass

    try:
        with open(perplexity_summary_path, "a") as f:
            f.write(f"{image_name},{result['perplexity']:.10f},{result['num_generated']}\n")
    except Exception:
        pass

    return True

def process_directory(base_input_dir, base_output_dir, split, condition, model, tokenizer, device='cuda', torch_type=None):
    input_dir = os.path.join(base_input_dir, split, condition)
    output_dir = os.path.join(base_output_dir, split, condition)

    if not os.path.isdir(input_dir):
        return 0

    try:
        os.makedirs(output_dir, exist_ok=True)
    except OSError:
        return 0

    image_patterns = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp"]
    image_paths = []
    for pattern in image_patterns:
        image_paths.extend(glob.glob(os.path.join(input_dir, pattern)))

    if not image_paths:
        return 0

    perplexity_summary_path = os.path.join(output_dir, "perplexity_summary.csv")
    if os.path.exists(perplexity_summary_path):
        os.remove(perplexity_summary_path)

    successful = 0
    for img_num, image_path in tqdm(
        enumerate(image_paths),
        total=len(image_paths),
        desc=f"Processing {split}/{condition}"
    ):
        if process_image(image_path, model, tokenizer, img_num, output_dir, device, torch_type):
            successful += 1

    return successful

def main():
    parser = argparse.ArgumentParser(description="Feature Extraction")
    parser.add_argument('--model_path', type=str, default="MODEL_PATH")
    parser.add_argument('--quant', type=int, choices=[0, 4, 8], default=0)
    parser.add_argument('--input_dir', type=str, default="/path/to/input")
    parser.add_argument('--output_dir', type=str, default="/path/to/output")
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    splits = ["train", "test"]
    conditions = ["unfinetuned"]

    try:
        model, tokenizer, torch_type = get_model_outputs(
            model_path=args.model_path,
            quantization=args.quant,
            device=device
        )
    except Exception:
        traceback.print_exc()
        return

    total_processed = 0
    total_images_found = 0

    for split in splits:
        for condition in conditions:
            current_input_dir = os.path.join(args.input_dir, split, condition)
            if os.path.isdir(current_input_dir):
                image_patterns = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp"]
                current_images = 0
                for pattern in image_patterns:
                    current_images += len(glob.glob(os.path.join(current_input_dir, pattern)))
                total_images_found += current_images
            else:
                current_images = 0

            successful_count = process_directory(
                args.input_dir, args.output_dir, split, condition,
                model, tokenizer, device, torch_type
            )
            total_processed += successful_count

    print("Done")
    print(f"Total images found: {total_images_found}")
    print(f"Total images processed: {total_processed}")
    print(f"Features saved to: {args.output_dir}")

if __name__ == "__main__":
    main()
