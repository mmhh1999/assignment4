import argparse
import os
import os.path as osp
import time

import torch
import torch.nn as nn
from PIL import Image
from SDS import SDS
from tqdm import tqdm
from utils import get_cosine_schedule_with_warmup, prepare_embeddings, seed_everything


def optimize_an_image(
    sds,
    prompt,
    neg_prompt="",
    img=None,
    log_interval=100,
    args=None
):
    """
    Optimize an image to match the prompt.
    """
    
    # --- [START] --- 根本修复 1：替换 Step 1 ---
    #
    # Step 1. Create text embeddings from prompt
    # 我们彻底绕过 'prepare_embeddings' (来自 utils.py)，因为它不可靠。
    # 我们将直接调用 'sds.get_text_embeddings()'，
    # 我们从 SDS.py 中知道它需要一个 *列表* (list) 作为输入。
    
    # 'prompt' 是一个字符串 (例如 "a hamburger")，我们把它包装成一个列表
    text_cond = sds.get_text_embeddings([prompt])
    
    # 'neg_prompt' 是一个字符串 (例如 "")，我们也把它包装成一个列表
    text_uncond = sds.get_text_embeddings([neg_prompt])
    
    # 我们手动创建 'embeddings' 字典，供后续代码使用
    embeddings = {
        "text_cond": text_cond,
        "text_uncond": text_uncond
    }
    
    # --- [END] --- 根本修复 1 结束 ---
    
    sds.text_encoder.to("cpu")  # free up GPU memory
    torch.cuda.empty_cache()

    # Step 2. Initialize latents to optimize
    latents = nn.Parameter(torch.randn(1, 4, 64, 64, device=sds.device))

    # Step 3. Create optimizer and loss function
    optimizer = torch.optim.AdamW([latents], lr=1e-1, weight_decay=0)
    total_iter = 2000
    scheduler = get_cosine_schedule_with_warmup(optimizer, 100, int(total_iter * 1.5))

    # Step 4. Training loop to optimize the latents
    for i in tqdm(range(total_iter)):
        optimizer.zero_grad()
        # Forward pass to compute the loss
        
        ### YOUR CODE HERE ###
        
        
        # pull out embeddings and ensure device
        if isinstance(embeddings, dict):
            text_cond = embeddings.get("text_cond", None)
            text_uncond = embeddings.get("text_uncond", None)
        else:
            # (这种情况不应该再发生了)
            if len(embeddings) == 2:
                text_cond, text_uncond = embeddings
            else:
                text_cond, text_uncond = embeddings, None

        if text_cond is None:
            # (这个错误现在真的不应该再发生了)
            raise ValueError("text_cond (conditional embedding) is None. 'get_text_embeddings' failed.")

        text_cond = text_cond.to(sds.device)
        if text_uncond is not None:
            text_uncond = text_uncond.to(sds.device)

        if args.sds_guidance:
            if text_uncond is None:
                print("Warning: sds_guidance is True, but text_uncond is None. Using no guidance.")
                loss = sds.sds.sds_loss(
                    latents=latents,
                    text_embeddings=text_cond,
                    text_embeddings_uncond=None,
                    guidance_scale=1.0,
                    grad_scale=1.0,
                )
            else:
                loss = sds.sds_loss(
                    latents=latents,
                    text_embeddings=text_cond,
                    text_embeddings_uncond=text_uncond,
                    guidance_scale=100,
                    grad_scale=1.0,
                )
        else:
            loss = sds.sds_loss(
                latents=latents,
                text_embeddings=text_cond,
                text_embeddings_uncond=None,
                guidance_scale=1.0,
                grad_scale=1.0,
            )
        
        ### END YOUR CODE HERE ###

        # Backward pass
        loss.backward()
        optimizer.step()
        scheduler.step()

        # clamping the latents to avoid over saturation
        latents.data = latents.data.clip(-1, 1)

        if i % log_interval == 0 or i == total_iter - 1:
            # Decode the image to visualize the progress
            img = sds.decode_latents(latents.detach())
            # Save the image
            output_im = Image.fromarray(img.astype("uint8"))
            
            # (使用 args.prompt 保证文件名正确)
            prompt_name = args.prompt.replace(' ', '_')
            output_path = os.path.join(
                sds.output_dir,
                f"output_{prompt_name}_iter_{i}.png",
            )
            output_im.save(output_path)

    return img


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, default="a hamburger")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="output")
    parser.add_argument("--sds_guidance", type=int, default=0, choices=[0, 1], help="boolen option to add guidance to the SDS loss")
    parser.add_argument(
        "--postfix",
        type=str,
        default="",
        help="postfix for the output directory to differentiate multiple runs",
    )
    args = parser.parse_args()

    seed_everything(args.seed)

    # create output directory
    args.output_dir = osp.join(args.output_dir, "image")
    output_dir = os.path.join(
        args.output_dir, args.prompt.replace(" ", "_") + args.postfix
    )
    
    # --- [START] --- 根本修复 2：修复 'exist_OK' 错误 ---
    # os.makedirs(output_dir, exist_OK=True) # 这一行在 Python 2 中会崩溃
    
    # 这是 Python 2 兼容的写法：
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # --- [END] --- 根本修复 2 结束 ---

    # initialize SDS
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sds = SDS(sd_version="2.1", device=device, output_dir=output_dir)

    # optimize an image
    prompt = args.prompt
    start_time = time.time()
    
    # (保持这个调用不变，使用字符串 'prompt')
    img = optimize_an_image(sds, prompt=prompt, args=args)
    
    print(f"Optimization took {time.time() - start_time:.2f} seconds")

    # save the output image
    img = Image.fromarray(img.astype("uint8"))
    output_path = os.path.join(output_dir, f"output.png")
    print(f"Saving image to {output_path}")
    img.save(output_path)