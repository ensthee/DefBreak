import torch
import torch.nn.functional as F
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
from PIL import Image
import numpy as np
import os
from tqdm import tqdm
import glob
import traceback

# Model
def get_model_outputs(
    model_path="/path/to/model",
    use_flash_attention=False,
):
    print(f"Loading model from: {model_path}")
    if use_flash_attention:
        print("Using Flash Attention 2")
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map="auto",
        )
    else:
        print("Using default attention implementation")
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_path, torch_dtype="auto", device_map="auto"
        )
    print("Model loaded.")

    print(f"Loading processor from: {model_path}")
    min_pixels = 336 * 336
    max_pixels = 512 * 512
    processor = AutoProcessor.from_pretrained(
        model_path, min_pixels=min_pixels, max_pixels=max_pixels
    )
    print("Processor loaded.")
    return model, processor


# Inputs
def process_inputs_for_generation(model, processor, messages):
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)

    model_inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    return model_inputs.to(model.device)


# Perplexity
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
        log_prob = max(step_log_softmax[actual_token_id].item(), -20.0)
        log_probs.append(log_prob)

    if not log_probs:
        return 1.0
    avg_neg_log_prob = -sum(log_probs) / len(log_probs)
    perplexity = np.exp(avg_neg_log_prob)

    if np.isinf(perplexity) or np.isnan(perplexity):
        return 1e10

    return perplexity


# Save features
def save_generated_distribution_and_perplexity(
    model,
    processor,
    inputs,
    save_path="probabilities_mean.txt",
    max_new_tokens=1024,
    min_logit_threshold=-10.0,
    min_prob_threshold=1e-6,
    top_k_features=120000,
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
            )
        except Exception as e:
            print(f"Error during model.generate for {save_path}: {e}")
            traceback.print_exc()
            return None

    scores = outputs.get("scores", None)
    generated_ids = outputs.get("sequences", None)

    if scores is None or generated_ids is None:
        print(f"Warning: generate() did not return scores or sequences for {save_path}.")
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
                step_logits_processed[step_logits_processed < min_logit_threshold] = (
                    min_logit_threshold
                )

            step_probs = torch.softmax(step_logits_processed, dim=0)
            all_step_probabilities.append(step_probs)

        if all_step_probabilities:
            mean_probs_np = torch.mean(
                torch.stack(all_step_probabilities), dim=0
            ).numpy()
        else:
            mean_probs_np = np.zeros(vocab_size)

    current_vocab_size = len(mean_probs_np)
    if current_vocab_size >= top_k_features:
        feature_probs = mean_probs_np[:top_k_features]
    else:
        print(
            f"Warning: Vocab size {current_vocab_size} < top_k_features {top_k_features}."
        )
        feature_probs = np.zeros(top_k_features)
        feature_probs[:current_vocab_size] = mean_probs_np

    combined_features = np.append(feature_probs, perplexity)

    try:
        np.savetxt(save_path, combined_features, fmt="%.10f")
    except Exception as e:
        print(f"Error saving features to {save_path}: {e}")
        traceback.print_exc()
        return None

    return {
        "mean_probabilities_feature": feature_probs,
        "perplexity": perplexity,
        "num_generated": num_generated,
        "save_path": save_path,
    }


# Image process
def process_image(image_path, model, processor, img_num, output_dir):
    try:
        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            print(f"Error opening image {image_path}: {e}")
            return False

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": "describe the image."},
                ],
            }
        ]

        try:
            inputs = process_inputs_for_generation(model, processor, messages)
        except Exception as e:
            print(f"Error processing inputs for {image_path}: {e}")
            traceback.print_exc()
            return False

        output_path = os.path.join(output_dir, f"file_{img_num}.txt")
        result = save_generated_distribution_and_perplexity(
            model,
            processor,
            inputs,
            output_path,
            max_new_tokens=1024,
            top_k_features=120000,
        )

        if result is None:
            print(f"Failed to save features for {image_path}.")
            return False

        summary_path = os.path.join(output_dir, "perplexity_summary.csv")
        image_name = os.path.basename(image_path)

        if not os.path.exists(summary_path):
            try:
                with open(summary_path, "w") as f:
                    f.write("image_name,self_perplexity,num_generated_tokens\n")
            except Exception:
                pass

        try:
            with open(summary_path, "a") as f:
                f.write(
                    f"{image_name},{result['perplexity']:.10f},{result['num_generated']}\n"
                )
        except Exception:
            pass

        return True

    except Exception as e:
        print(f"Unexpected error processing image {image_path}: {str(e)}")
        traceback.print_exc()
        return False


# Recursive directory
def process_directory_recursive(
    base_input_dir, base_output_dir, split, condition, model, processor
):
    input_root = os.path.join(base_input_dir, split, condition)
    output_root = os.path.join(base_output_dir, split, condition)

    if not os.path.isdir(input_root):
        print(f"Input directory not found: {input_root}.")
        return 0

    total_success = 0
    total_images = 0

    print(f"Scanning recursively under {input_root}")

    for root, dirs, files in os.walk(input_root):
        image_paths = []
        for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp"):
            image_paths.extend(glob.glob(os.path.join(root, ext)))

        if not image_paths:
            continue

        rel_path = os.path.relpath(root, input_root)
        output_dir = os.path.join(output_root, rel_path)

        os.makedirs(output_dir, exist_ok=True)
        print(f"Processing folder: {rel_path}")
        print(f"Found {len(image_paths)} images")

        summary_path = os.path.join(output_dir, "perplexity_summary.csv")
        if os.path.exists(summary_path):
            os.remove(summary_path)

        success_count = 0
        for img_num, image_path in tqdm(
            enumerate(image_paths), total=len(image_paths), desc=f"{rel_path}"
        ):
            ok = process_image(image_path, model, processor, img_num, output_dir)
            if ok:
                success_count += 1

        total_success += success_count
        total_images += len(image_paths)

        print(f"{rel_path}: {success_count}/{len(image_paths)} processed.")

    print(f"Total processed: {total_success}/{total_images}")
    return total_success


# Main
def main():
    BASE_INPUT_DIR = "/path/to/input"
    BASE_OUTPUT_DIR = "/path/to/output"
    MODEL_PATH = "/path/to/model"
    USE_FLASH_ATTENTION = False

    SPLITS = ["test", "train"]
    CONDITIONS = ["finetuned", "unfinetuned"]

    print("Initializing model and processor...")
    try:
        model, processor = get_model_outputs(
            model_path=MODEL_PATH, use_flash_attention=USE_FLASH_ATTENTION
        )
    except Exception as e:
        print(f"Failed to initialize model or processor: {e}")
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
                    current_images += len(
                        glob.glob(os.path.join(current_input_dir, pattern))
                    )
                total_images_found += current_images
            else:
                current_images = 0

            successful_count = process_directory_recursive(
                BASE_INPUT_DIR, BASE_OUTPUT_DIR, split, condition, model, processor
            )
            total_processed += successful_count

    print("Processing complete!")
    print(
        f"Total images found across all specified input directories: {total_images_found}"
    )
    print(f"Total images successfully processed: {total_processed}")
    print(f"Features saved to base directory: {BASE_OUTPUT_DIR}")


if __name__ == "__main__":
    main()
