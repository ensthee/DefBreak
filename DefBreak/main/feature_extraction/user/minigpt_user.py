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


# ----------------------------- #
#        Argument Parser        #
# ----------------------------- #
def parse_args():
    parser = argparse.ArgumentParser(description="MiniGPT-4 Perplexity Generation")
    parser.add_argument("--cfg-path", required=True, help="config file")
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--base-input-dir", required=True)
    parser.add_argument("--base-output-dir", required=True)
    parser.add_argument("--splits", nargs="+", default=["train", "test"])
    parser.add_argument("--conditions", nargs="+", default=["unfinetuned"])
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--top-k-features", type=int, default=120000)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max-length", type=int, default=2000)
    parser.add_argument("--options", nargs="+")
    return parser.parse_args()


# ----------------------------- #
#           Utilities           #
# ----------------------------- #
def setup_seeds(config):
    seed = config.run_cfg.seed + get_rank()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    cudnn.benchmark = False
    cudnn.deterministic = True


def load_image(image_file):
    if image_file.startswith("http"):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        image = Image.open(image_file).convert("RGB")
    return image


# ----------------------------- #
#         Model Loading         #
# ----------------------------- #
def get_model_and_chat(cfg_path, gpu_id):
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
    return model, chat


# ----------------------------- #
#      Perplexity Function      #
# ----------------------------- #
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

    avg_neg_log_prob = -sum(log_probs) / len(log_probs)
    perplexity = np.exp(avg_neg_log_prob)
    if np.isinf(perplexity) or np.isnan(perplexity):
        return 1e10
    return perplexity


# ----------------------------- #
#     MiniGPT4 Text Generate    #
# ----------------------------- #
def generate_with_logits_tracking(chat, img, user_message, temperature=1.0, max_length=2000):
    try:
        chat_state = CONV_VISION_Vicuna0.copy()
        img_list = []

        chat.upload_img(img, chat_state, img_list)
        chat.encode_img(img_list)
        chat.ask(user_message, chat_state)

        try:
            probabilities = chat.answer_logit(
                conv=chat_state,
                img_list=img_list,
                temperature=temperature,
                max_length=max_length,
            )
        except Exception:
            try:
                generated_text, generated_token_ids = chat.answer(
                    conv=chat_state,
                    img_list=img_list,
                    temperature=temperature,
                    max_length=max_length,
                )
                return generated_text, [], generated_token_ids
            except Exception:
                return "", [], []

        if (
            probabilities is not None
            and len(probabilities.shape) == 2
            and probabilities.shape[0] > 0
        ):
            epsilon = 1e-8
            logits_sequence = [
                torch.log(probabilities[i] + epsilon).cpu()
                for i in range(probabilities.shape[0])
            ]
            generated_token_ids = torch.argmax(probabilities, dim=-1).cpu().numpy()
            generated_text = chat.model.llama_tokenizer.decode(
                generated_token_ids, skip_special_tokens=True
            )
            return generated_text, logits_sequence, generated_token_ids
        else:
            try:
                generated_text, generated_token_ids = chat.answer(
                    conv=chat_state,
                    img_list=img_list,
                    temperature=temperature,
                    max_length=max_length,
                )
                return generated_text, [], generated_token_ids
            except Exception:
                return "", [], []

    except Exception:
        traceback.print_exc()
        return "", [], []


# ----------------------------- #
#   Save Probability Features   #
# ----------------------------- #
def save_generated_distribution_and_perplexity(
    chat,
    img,
    save_path="probabilities_mean.txt",
    max_new_tokens=1024,
    temperature=1.0,
    max_length=2000,
    min_logit_threshold=-10.0,
    top_k_features=32000,
    user_message="Describe the image in detail",
):
    try:
        generated_text, logits_sequence, generated_tokens = (
            generate_with_logits_tracking(
                chat, img, user_message, temperature, max_length
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
            mean_probabilities = torch.mean(
                torch.stack(all_step_probabilities), dim=0
            )
            mean_probs_np = mean_probabilities.detach().numpy()

        current_vocab_size = len(mean_probs_np)
        if current_vocab_size >= top_k_features:
            feature_probs = mean_probs_np[:top_k_features]
        else:
            feature_probs = np.zeros(top_k_features)
            feature_probs[:current_vocab_size] = mean_probs_np

        combined_features = np.append(feature_probs, perplexity)

        np.savetxt(save_path, combined_features, fmt="%.10f")

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


# ----------------------------- #
#         Image Process         #
# ----------------------------- #
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

        summary_path = os.path.join(output_dir, "perplexity_summary.csv")
        if not os.path.exists(summary_path):
            with open(summary_path, "w") as f:
                f.write("image_name,self_perplexity,num_generated_tokens\n")
        with open(summary_path, "a") as f:
            f.write(
                f"{os.path.basename(image_path)},{result['perplexity']:.10f},{result['num_generated']}\n"
            )

        return True
    except Exception:
        traceback.print_exc()
        return False


# ----------------------------- #
#     Directory Recursion       #
# ----------------------------- #
def process_directory(base_input_dir, base_output_dir, split, condition, chat, args):
    input_root = os.path.join(base_input_dir, split, condition)
    output_root = os.path.join(base_output_dir, split, condition)

    if not os.path.isdir(input_root):
        return 0

    total_success = 0
    patterns = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp"]

    for class_dir, _, _ in os.walk(input_root):
        rel_path = os.path.relpath(class_dir, input_root)
        output_dir = os.path.join(output_root, rel_path)

        image_paths = []
        for p in patterns:
            image_paths.extend(glob.glob(os.path.join(class_dir, p)))

        if not image_paths:
            continue

        os.makedirs(output_dir, exist_ok=True)

        summary_file = os.path.join(output_dir, "perplexity_summary.csv")
        if os.path.exists(summary_file):
            os.remove(summary_file)

        successful = 0
        for img_num, image_path in tqdm(
            enumerate(image_paths),
            total=len(image_paths),
            desc=f"{split}/{condition}/{rel_path}",
        ):
            if process_image(image_path, chat, img_num, output_dir, args):
                successful += 1

        total_success += successful

    return total_success


# ----------------------------- #
#              Main             #
# ----------------------------- #
def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler("minigpt_tdsc_log.txt"), logging.StreamHandler()],
    )

    args = parse_args()
    try:
        model, chat = get_model_and_chat(args.cfg_path, args.gpu_id)
    except Exception:
        traceback.print_exc()
        return

    total_processed = 0
    for split in args.splits:
        for condition in args.conditions:
            total_processed += process_directory(
                args.base_input_dir, args.base_output_dir, split, condition, chat, args
            )

    print(f"Total images processed: {total_processed}")
    print(f"Features saved under: {args.base_output_dir}")


if __name__ == "__main__":
    main()
