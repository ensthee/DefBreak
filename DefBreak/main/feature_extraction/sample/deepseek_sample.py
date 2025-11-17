import torch
import torch.nn.functional as F
from modelscope import AutoModelForCausalLM
from deepseek_vl2.models import DeepseekVLV2Processor, DeepseekVLV2ForCausalLM
from deepseek_vl2.utils.io import load_pil_images
from PIL import Image
import numpy as np
import os
from tqdm import tqdm
import glob
import traceback


def get_model_outputs(
    model_path="/path/to/model",
    device="cuda",
):
    print(f"Loading model from: {model_path}")
    processor = DeepseekVLV2Processor.from_pretrained(model_path)
    tokenizer = processor.tokenizer
    model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)
    model = model.to(torch.bfloat16).cuda().eval()
    return model, processor, tokenizer


def process_inputs_for_generation(model, processor, tokenizer, image_path):
    conversation = [
        {
            "role": "<|User|>",
            "content": "<image>\nDescribe the image.",
            "images": [image_path],
        },
        {"role": "<|Assistant|>", "content": ""},
    ]
    pil_images = load_pil_images(conversation)
    prepare_inputs = processor(
        conversations=conversation,
        images=pil_images,
        force_batchify=True,
        system_prompt=""
    ).to(model.device)
    return prepare_inputs


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
        log_prob = max(log_prob, -20.0)
        log_probs.append(log_prob)

    avg_neg_log_prob = -sum(log_probs) / len(log_probs)
    perplexity = np.exp(avg_neg_log_prob)
    if np.isinf(perplexity) or np.isnan(perplexity):
        return 1e10
    return perplexity


def save_generated_distribution_and_perplexity(
    model,
    processor,
    tokenizer,
    inputs,
    save_path="probabilities_mean.txt",
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
                use_cache=True
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
        vocab_size = len(tokenizer)
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


def process_image(image_path, model, processor, tokenizer, img_num, output_dir):
    if not os.path.exists(image_path):
        return False
    try:
        with Image.open(image_path) as test_img:
            test_img.verify()
    except Exception:
        return False

    try:
        inputs = process_inputs_for_generation(model, processor, tokenizer, image_path)
    except Exception:
        traceback.print_exc()
        return False

    output_path = os.path.join(output_dir, f"file_{img_num}.txt")
    result = save_generated_distribution_and_perplexity(
        model,
        processor,
        tokenizer,
        inputs,
        output_path,
        max_new_tokens=512,
        top_k_features=120000,
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


def process_directory(base_input_dir, base_output_dir, split, condition, model, processor, tokenizer):
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
        desc=f"Processing {split}/{condition}",
    ):
        if process_image(image_path, model, processor, tokenizer, img_num, output_dir):
            successful += 1

    return successful


def main():
    BASE_INPUT_DIR = "/path/to/input"
    BASE_OUTPUT_DIR = "/path/to/output"
    MODEL_PATH = "/path/to/model"

    SPLITS = ["train", "test"]
    CONDITIONS = ["finetuned", "unfinetuned"]

    try:
        model, processor, tokenizer = get_model_outputs(model_path=MODEL_PATH)
    except Exception:
        traceback.print_exc()
        return

    total_processed = 0
    total_images_found = 0

    for split in SPLITS:
        for condition in CONDITIONS:
            current_input_dir = os.path.join(BASE_INPUT_DIR, split, condition)
            if os.path.isdir(current_input_dir):
                image_patterns = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp"]
                current_images = 0
                for pattern in image_patterns:
                    current_images += len(glob.glob(os.path.join(current_input_dir, pattern)))
                total_images_found += current_images

            successful_count = process_directory(
                BASE_INPUT_DIR, BASE_OUTPUT_DIR, split, condition, model, processor, tokenizer
            )
            total_processed += successful_count

    print("Done")
    print(f"Total images found: {total_images_found}")
    print(f"Total processed: {total_processed}")
    print(f"Saved to: {BASE_OUTPUT_DIR}")


if __name__ == "__main__":
    main()
