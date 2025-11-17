from io import BytesIO
import os
import random
import glob
import logging
import requests
import torch
import torch.nn.functional as F
import numpy as np
import torch.backends.cudnn as cudnn
from tqdm import tqdm
import traceback

from minigpt4.common.config import Config
from minigpt4.common.dist_utils import get_rank
from minigpt4.common.registry import registry
from minigpt4.conversation.conversation_backup import Chat, CONV_VISION_Vicuna0

from minigpt4.datasets.builders import *
from minigpt4.models import *
from minigpt4.processors import *
from minigpt4.runners import *
from minigpt4.tasks import *
import argparse
from PIL import Image


def parse_args():
    parser = argparse.ArgumentParser(description="MiniGPT-4 Perplexity Generation")
    parser.add_argument("--cfg-path", required=True)
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--base-input-dir", required=True)
    parser.add_argument("--base-output-dir", required=True)
    parser.add_argument("--splits", nargs="+", default=["train", "test"])
    parser.add_argument("--conditions", nargs="+", default=["finetuned"])
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--top-k-features", type=int, default=120000)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-length", type=int, default=2000)
    parser.add_argument("--options", nargs="+")
    args = parser.parse_args()
    return args


def setup_seeds(config):
    seed = config.run_cfg.seed + get_rank()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    cudnn.benchmark = False
    cudnn.deterministic = True


def load_image(image_file):
    if image_file.startswith("http") or image_file.startswith("https"):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        image = Image.open(image_file).convert("RGB")
    return image


def get_model_and_chat(cfg_path, gpu_id):
    print("Loading MiniGPT-4 model...")
    cfg = Config(argparse.Namespace(cfg_path=cfg_path, options=None))
    model_config = cfg.model_cfg
    model_config.device_8bit = gpu_id
    model_cls = registry.get_model_class(model_config.arch)
    model = model_cls.from_config(model_config).to(f"cuda:{gpu_id}")
    vis_processor_cfg = cfg.datasets_cfg.cc_sbu_align.vis_processor.train
    vis_processor = registry.get_processor_class(vis_processor_cfg.name).from_config(
        vis_processor_cfg
    )
    chat = Chat(model, vis_processor, device=f"cuda:{gpu_id}")
    print("MiniGPT-4 model loaded successfully.")
    return model, chat


def calculate_self_perplexity_from_logits(
    logits_sequence, generated_tokens, min_logit_threshold=None
):
    if len(logits_sequence) == 0 or len(generated_tokens) == 0:
        return 1.0

    log_probs = []

    for step_log_probs, actual_token_id in zip(logits_sequence, generated_tokens):
        log_prob = step_log_probs[actual_token_id].item()
        log_prob = max(log_prob, -20.0)
        log_probs.append(log_prob)

    if not log_probs:
        return 1.0

    avg_neg_log_prob = -sum(log_probs) / len(log_probs)
    perplexity = np.exp(avg_neg_log_prob)
    if np.isinf(perplexity) or np.isnan(perplexity):
        return 1e10
    return perplexity


def generate_with_logits_tracking(
    chat, img, user_message, temperature=0.2, max_new_tokens=300
):
    try:
        chat_state = CONV_VISION_Vicuna0.copy()
        img_list = []
        chat.upload_img(img, chat_state, img_list)
        chat.encode_img(img_list)
        chat.ask(user_message, chat_state)

        try:
            chat_state.append_message(chat_state.roles[1], None)
            prompt = chat_state.get_prompt()
            embs = chat.model.get_context_emb(prompt, img_list)

            with torch.no_grad():
                outputs = chat.model.llama_model.generate(
                    inputs_embeds=embs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    do_sample=True if temperature > 0 else False,
                    output_scores=True,
                    return_dict_in_generate=True,
                    stopping_criteria=chat.stopping_criteria,
                    pad_token_id=chat.model.llama_tokenizer.eos_token_id,
                )

            generated_ids = outputs.sequences
            scores = outputs.scores if hasattr(outputs, "scores") else []

            generated_text = chat.model.llama_tokenizer.decode(
                generated_ids[0], skip_special_tokens=True
            )
            generated_text = generated_text.split("###")[0]
            generated_text = generated_text.split("Assistant:")[-1].strip()

            if len(scores) > 0:
                num_new_tokens = len(scores)
                total_tokens = generated_ids.shape[1]
                prompt_token_len = total_tokens - num_new_tokens
                new_token_ids = generated_ids[0][prompt_token_len:].cpu().numpy()

                logits_sequence = []
                for score in scores:
                    step_probs = F.softmax(score[0].float(), dim=-1)
                    step_log_probs = torch.log(step_probs + 1e-8)
                    logits_sequence.append(step_log_probs.cpu())

                return generated_text, logits_sequence, new_token_ids
            else:
                return generated_text, [], []

        except Exception:
            try:
                chat_state_b = CONV_VISION_Vicuna0.copy()
                img_list_b = []
                chat.upload_img(img, chat_state_b, img_list_b)
                chat.encode_img(img_list_b)
                chat.ask(user_message, chat_state_b)

                gen_text, gen_token_ids = chat.answer(
                    conv=chat_state_b,
                    img_list=img_list_b,
                    temperature=temperature,
                    max_new_tokens=max_new_tokens,
                )
                return gen_text, [], gen_token_ids
            except Exception:
                return "", [], []

    except Exception:
        traceback.print_exc()
        return "", [], []


def save_generated_distribution_and_perplexity(
    chat,
    img,
    save_path="probabilities_mean.txt",
    max_new_tokens=1024,
    temperature=0.2,
    max_length=2000,
    min_logit_threshold=-10.0,
    top_k_features=120000,
    user_message="Describe the image in detail",
):
    try:
        generated_text, logits_sequence, generated_tokens = (
            generate_with_logits_tracking(
                chat, img, user_message, temperature, max_new_tokens
            )
        )

        if not logits_sequence or len(generated_tokens) == 0:
            vocab_size = 32000
            mean_probs_np = np.zeros(vocab_size)
            perplexity = 1.0
            num_generated = 0
        else:
            perplexity = calculate_self_perplexity_from_logits(
                logits_sequence, generated_tokens, min_logit_threshold
            )
            num_generated = len(logits_sequence)
            all_step_probabilities = []
            vocab_size = logits_sequence[0].shape[-1]

            for step_log_probs in logits_sequence:
                step_probs = torch.exp(step_log_probs)
                all_step_probabilities.append(step_probs)

            if all_step_probabilities:
                mean_probabilities = torch.mean(
                    torch.stack(all_step_probabilities), dim=0
                )
                mean_probs_np = mean_probabilities.detach().numpy()
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
            "generated_text": generated_text,
        }

    except Exception:
        traceback.print_exc()
        return None


def process_image(image_path, chat, img_num, output_dir, args):
    try:
        try:
            image = load_image(image_path)
        except Exception:
            return False

        output_path = os.path.join(output_dir, f"file_{img_num}.txt")
        result = save_generated_distribution_and_perplexity(
            chat,
            image,
            output_path,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            max_length=args.max_length,
            top_k_features=args.top_k_features,
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
                f.write(
                    f"{image_name},{result['perplexity']:.10f},{result['num_generated']}\n"
                )
        except Exception:
            pass

        return True

    except Exception:
        traceback.print_exc()
        return False


def process_directory(base_input_dir, base_output_dir, split, condition, chat, args):
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
        if process_image(image_path, chat, img_num, output_dir, args):
            successful += 1

    return successful


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler("minigpt_log.txt"), logging.StreamHandler()],
    )

    args = parse_args()

    print("Initializing MiniGPT-4 model...")
    try:
        model, chat = get_model_and_chat(args.cfg_path, args.gpu_id)
    except Exception:
        traceback.print_exc()
        return

    total_processed = 0
    total_images_found = 0

    for split in args.splits:
        for condition in args.conditions:
            current_input_dir = os.path.join(args.base_input_dir, split, condition)
            if os.path.isdir(current_input_dir):
                image_patterns = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp"]
                current_images = 0
                for pattern in image_patterns:
                    current_images += len(glob.glob(os.path.join(current_input_dir, pattern)))
                total_images_found += current_images

            successful_count = process_directory(
                args.base_input_dir, args.base_output_dir, split, condition, chat, args
            )
            total_processed += successful_count

    print("Done")
    print(f"Total images found: {total_images_found}")
    print(f"Total processed: {total_processed}")
    print(f"Saved to: {args.base_output_dir}")


if __name__ == "__main__":
    main()
