import argparse
import os
import os.path as osp
import time

import pytorch3d
import torch
from implicit import ColorField
from PIL import Image
from pytorch3d.renderer import TexturesVertex
from SDS import SDS
from tqdm import tqdm
from utils import (
    get_cosine_schedule_with_warmup,
    get_mesh_renderer_soft,
    init_mesh,
    prepare_embeddings,
    render_360_views,
    seed_everything,
)


def optimize_mesh_texture(
    sds,
    mesh_path,
    prompt,
    neg_prompt="",
    device="cpu",
    log_interval=100,
    save_mesh=True,
    args=None,
):
    """
    Optimize the texture map of a mesh to match the prompt.
    """
    # --- [START] --- 修复 1: 替换 prepare_embeddings ---
    # Step 1. Create text embeddings from prompt
    # 我们绕过 'prepare_embeddings' (来自 utils.py)，因为它不可靠。
    # 我们将直接调用 'sds.get_text_embeddings()'，它需要一个 *列表* (list)。
    
    # 'prompt' 是一个字符串, 'neg_prompt' 也是
    text_cond = sds.get_text_embeddings([prompt])
    text_uncond = sds.get_text_embeddings([neg_prompt])
    
    embeddings = {
        "text_cond": text_cond,
        "text_uncond": text_uncond
    }
    # --- [END] --- 修复 1 结束 ---

    sds.text_encoder.to("cpu")  # free up GPU memory
    torch.cuda.empty_cache()

    # Step 2. Load the mesh
    mesh, vertices, faces, aux = init_mesh(mesh_path, device=device)
    vertices = vertices.unsqueeze(0).to(device)  # (N_v, 3) -> (1, N_v, 3)
    faces = faces.unsqueeze(0).to(device)  # (N_f, 3) -> (1, N_f, 3)

    # Step 2.1 Initialize a randome texture map (optimizable parameter)
    # create a texture field with implicit function
    color_field = ColorField().to(device)  # input (1, N_v, xyz) -> output (1, N_v, rgb)
    mesh = pytorch3d.structures.Meshes(
        verts=vertices,
        faces=faces,
        textures=TexturesVertex(verts_features=color_field(vertices)),
    )
    mesh = mesh.to(device)

    # Step 3.1 Initialize the renderer
    lights = pytorch3d.renderer.PointLights(location=[[0, 0, -3]], device=device)
    renderer = get_mesh_renderer_soft(image_size=512, device=device, lights=lights)

    # For logging purpose, render 360 views of the initial mesh
    if save_mesh:
        render_360_views(
            mesh.detach(),
            renderer,
            device=device,
            output_path=osp.join(sds.output_dir, "initial_mesh.gif"),
        )

    # Step 3.2. Initialize the cameras
    # check the size of the mesh so that it is in the field of view
    print(
        f"check mesh range: {vertices.min()}, {vertices.max()}, center {vertices.mean(1)}"
    )

    ### YOUR CODE HERE ### (这个部分我们留空，因为我们在循环中采样)
    # create a list of query cameras as the training set
    # Note: to create the dataset, you can either pre-define a list of query cameras as below or randomly sample a camera pose on the fly in the training loop.
    query_cameras = [] # optional

    # Step 4. Create optimizer training parameters
    optimizer = torch.optim.AdamW(color_field.parameters(), lr=5e-4, weight_decay=0)
    total_iter = 2000
    scheduler = get_cosine_schedule_with_warmup(optimizer, 100, int(total_iter * 1.5))

    # Step 5. Training loop to optimize the texture map
    loss_dict = {}
    for i in tqdm(range(total_iter)):
        # Initialize optimizer
        optimizer.zero_grad()

        # Update the textures
        mesh.textures = TexturesVertex(verts_features=color_field(vertices))

        ### YOUR CODE HERE ###
        
        # --- [START] --- 填充 Q2.2 的核心逻辑 ---
        
        # 1. 提取 embeddings (和 Q2.1 一样)
        if isinstance(embeddings, dict):
            text_cond = embeddings.get("text_cond", None)
            text_uncond = embeddings.get("text_uncond", None)
        text_cond = text_cond.to(sds.device)
        if text_uncond is not None:
            text_uncond = text_uncond.to(sds.device)
            
        # 2. 动态随机采样一个相机视角
        # 固定距离，随机高度和角度
        dist = 2.7 
        # 随机高度 (elevation)，例如 0 到 60 度
        elev = torch.rand(1) * 60
        # 随机方位角 (azimuth)，0 到 360 度
        azim = torch.rand(1) * 360
        
        # 获取相机位姿
        R, T = pytorch3d.renderer.look_at_view_transform(dist, elev, azim, device=device)
        cameras = pytorch3d.renderer.FoVPerspectiveCameras(R=R, T=T, fov=60, device=device)

        # Forward pass
        # 3. 渲染这个随机视角
        # renderer 的输出是 (B, H, W, 4) RGBA
        rend_rgba = renderer(mesh, cameras=cameras)
        
        # SDS 需要 (B, 3, H, W) 的 RGB 图像，范围 [0, 1]
        rend = rend_rgba[..., :3].permute(0, 3, 1, 2) # 取 RGB 并 B,C,H,W

        # 4. 将渲染图像编码为 latents
        latents = sds.encode_imgs(rend)
        
        # 5. 计算 SDS loss (我们总是使用 "有指导" 的版本)
        loss = sds.sds_loss(
            latents=latents,
            text_embeddings=text_cond,
            text_embeddings_uncond=text_uncond,
            guidance_scale=100, # 总是使用强指导
            grad_scale=1.0,
        )
        # --- [END] --- Q2.2 核心逻辑结束 ---
        
        ### END YOUR CODE HERE ###

        # Backward pass
        loss.backward()
        optimizer.step()
        scheduler.step()

        # clamping the latents to avoid over saturation
        latents.data = latents.data.clip(-1, 1)
        if i % log_interval == 0 or i == total_iter - 1:
            # save the loss
            loss_dict[i] = loss.item()

            # save the image
            img = sds.decode_latents(latents.detach())
            output_im = Image.fromarray(img.astype("uint8"))
            
            # --- [START] --- 修复 2: 修复 prompt[0] 文件名 bug ---
            # 原始代码: f"output_{prompt[0].replace(' ', '_')}_iter_{i}.png",
            # 这会导致文件名错误 (例如 "a.png")
            prompt_name = prompt.replace(' ', '_')
            output_path = os.path.join(
                sds.output_dir,
                f"output_{prompt_name}_iter_{i}.png",
            )
            # --- [END] --- 修复 2 结束 ---
            
            output_im.save(output_path)

    if save_mesh:
        render_360_views(
            mesh.detach(),
            renderer,
            device=device,
            output_path=osp.join(sds.output_dir, f"final_mesh.gif"),
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, default="a hamburger")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="output")
    parser.add_argument(
        "--postfix",
        type=str,
        default="",
        help="postfix for the output directory to differentiate multiple runs",
    )

    parser.add_argument(
        "-m",
        "--mesh_path",
        type=str,
        default="data/cow.obj",
        help="Path to the input mesh",
    )
    args = parser.parse_args()

    seed_everything(args.seed)

    # create output directory
    args.output_dir = osp.join(args.output_dir, "mesh")
    output_dir = os.path.join(
        args.output_dir, args.prompt.replace(" ", "_") + args.postfix
    )
    
    # --- [START] --- 修复 3: 修复 Python 2 'exist_OK' bug ---
    # 原始代码: os.makedirs(output_dir, exist_ok=True)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # --- [END] --- 修复 3 结束 ---

    # initialize SDS
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sds = SDS(sd_version="2.1", device=device, output_dir=output_dir)

    # optimize the texture map of a mesh
    start_time = time.time()
    assert (
        args.mesh_path is not None
    ), "mesh_path should be provided for optimizing the texture map for a mesh"
    optimize_mesh_texture(
        sds, mesh_path=args.mesh_path, prompt=args.prompt, device=device, args=args
    )
    print(f"Optimization took {time.time() - start_time:.2f} seconds")