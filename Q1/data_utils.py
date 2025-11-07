import os
import torch
import imageio
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from plyfile import PlyData
from torch.utils.data import Dataset
from pytorch3d.renderer.cameras import PerspectiveCameras, look_at_view_transform

SH_C0 = 0.28209479177387814
CMAP_JET = plt.get_cmap("jet")
CMAP_MIN_NORM, CMAP_MAX_NORM = 5.0, 7.0

class TruckDataset(Dataset):

    def __init__(self, root, split):
        super().__init__()
        self.root = root
        self.split = split
        if self.split not in ("train", "test"):
            raise ValueError(f"Invalid split: {self.split}")

        self.masks = []
        self.points = []
        self.images = []
        self.cameras = []

        imgs_root = os.path.join(root, "imgs")
        poses_root = os.path.join(root, "poses")
        points_root = os.path.join(root, "points")
        self.points_path = os.path.join(points_root, "points_10000.npy")

        data_img_size = None
        num_files = len(os.listdir(imgs_root))
        test_idxs = np.linspace(0, num_files, 7).astype(np.int32)[1:-1]
        test_idxs_set = set(test_idxs.tolist())
        train_idxs = [i for i in range(num_files) if i not in test_idxs_set]
        idxs = train_idxs if self.split == "train" else test_idxs

        for i in idxs:
            img_path = os.path.join(imgs_root, f"frame{i+1:06d}.jpg")
            npy_path = os.path.join(poses_root, f"frame{i+1:06d}.npy")

            img_ = imageio.v3.imread(img_path).astype(np.float32) / 255.0

            mask = None
            if img_.shape[-1] == 3:
                img = torch.tensor(img_)  # (H, W, 3)
            else:
                img = torch.tensor(img_[..., :3])  # (H, W, 3)
                mask = torch.tensor(img_[..., 3:4])  # (H, W, 1)
                
            img_size = img.shape[:2]
            h, w = img_size
            
            # Checking if all data samples have the same image size
            if data_img_size is None:
                data_img_size = (w,h) 
            else:
                if data_img_size[0] != img_size[1] or data_img_size[1] != img_size[0]:
                    raise RuntimeError

            pose = np.load(npy_path)
            R, T, F, C = pose[:9].reshape((3,3)), pose[9:12], pose[12:14], pose[14:16]
            
            # Screen space camera
            F = F * min(img_size) / 2 
            C = w / 2 - C[0] * min(img_size) / 2, h / 2 - C[1] * min(img_size) / 2  

            camera = PerspectiveCameras(
                focal_length=torch.tensor(F, dtype=torch.float)[None], 
                principal_point=torch.tensor(C, dtype=torch.float)[None],
                R=torch.tensor(R, dtype=torch.float)[None], 
                T=torch.tensor(T, dtype=torch.float)[None],
                in_ndc=False,
                image_size=((h,w),)
            )

            self.images.append(img)
            self.cameras.append(camera)
            if mask is not None:
                self.masks.append(mask)

        self.img_size = data_img_size

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        masks = None
        if len(self.masks) > 0:
            masks = self.masks[idx]
        return self.images[idx], self.cameras[idx], masks

    @staticmethod
    def collate_fn(batch):
        images = torch.stack([x[0] for x in batch], dim=0)
        cameras = [x[1] for x in batch]

        masks = [x[2] for x in batch if x[2] is not None]
        if len(masks) == 0:
            masks = None
        else:
            masks = torch.stack(masks, dim=0)

        return images, cameras, masks


def colour_depth_q1_render(depth):
    normalized_depth = (depth - CMAP_MIN_NORM) / (CMAP_MAX_NORM - CMAP_MIN_NORM + 1e-8)
    coloured_depth = CMAP_JET(normalized_depth)[:, :, :3]  # (H, W, 3)
    coloured_depth = (np.clip(coloured_depth, 0.0, 1.0) * 255.0).astype(np.uint8)

    return coloured_depth

def visualize_renders(scene, gt_viz_img, cameras, img_size):

    imgs = []
    viz_size = (256, 256)
    with torch.no_grad():
        for cam in cameras:
            pred_img, _, _ = scene.render(
                cam, img_size=img_size,
                bg_colour=(0.0, 0.0, 0.0),
                per_splat=-1,
            )
            img = torch.clamp(pred_img, 0.0, 1.0) * 255.0
            img = img.clone().detach().cpu().numpy().astype(np.uint8)

            if img_size[0] != viz_size[0] or img_size[1] != viz_size[1]:
                img = np.array(Image.fromarray(img).resize(viz_size))

            imgs.append(img)

    pred_viz_img = np.concatenate(imgs, axis=1)
    viz_frame = np.concatenate((pred_viz_img, gt_viz_img), axis=0)
    return viz_frame

def load_gaussians_from_ply(path):
    # Modified from https://github.com/thomasantony/splat
    max_sh_degree = 3
    plydata = PlyData.read(path)
    xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                    np.asarray(plydata.elements[0]["y"]),
                    np.asarray(plydata.elements[0]["z"])),  axis=1)
    opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

    features_dc = np.zeros((xyz.shape[0], 3, 1))
    features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
    features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
    features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

    extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
    extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
    assert len(extra_f_names) == 3 * (max_sh_degree + 1) ** 2 - 3
    features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
    for idx, attr_name in enumerate(extra_f_names):
        features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
    # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
    features_extra = features_extra.reshape((features_extra.shape[0], 3, (max_sh_degree + 1) ** 2 - 1))
    features_extra = np.transpose(features_extra, [0, 2, 1])

    scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
    scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
    scales = np.zeros((xyz.shape[0], len(scale_names)))
    for idx, attr_name in enumerate(scale_names):
        scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

    rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
    rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
    rots = np.zeros((xyz.shape[0], len(rot_names)))
    for idx, attr_name in enumerate(rot_names):
        rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

    xyz = xyz.astype(np.float32)
    rots = rots.astype(np.float32)
    scales = scales.astype(np.float32)
    opacities = opacities.astype(np.float32)
    shs = np.concatenate([
        features_dc.reshape(-1, 3),
        features_extra.reshape(len(features_dc), -1)
    ], axis=-1).astype(np.float32)
    shs = shs.astype(np.float32)

    dc_vals = shs[:, :3]
    dc_colours = np.maximum(dc_vals * SH_C0 + 0.5, np.zeros_like(dc_vals))

    output = {
        "xyz": xyz, "rot": rots, "scale": scales,
        "sh": shs, "opacity": opacities, "dc_colours": dc_colours
    }
    return output

def colours_from_spherical_harmonics(spherical_harmonics, gaussian_dirs):
    """
    [Q 1.3.1] Computes view-dependent color from SH coefficients and directions.

    Args:
        spherical_harmonics: torch.Tensor of shape (N, 48)
            Layout matches this file's loader:
              - first 3 numbers are DC for (R,G,B)
              - remaining 45 numbers are 15 higher-order bases × 3 channels,
                flattened in (basis-major, then channels) order.
        gaussian_dirs: torch.Tensor of shape (N, 3)
            World-space unit directions from camera center to Gaussian means.

    Returns:
        colours: torch.Tensor of shape (N, 3) in [0,1]
    """
    if spherical_harmonics.dim() != 2 or spherical_harmonics.shape[1] != 48:
        raise ValueError(f"Expected SH of shape (N, 48), got {tuple(spherical_harmonics.shape)}")
    if gaussian_dirs.dim() != 2 or gaussian_dirs.shape[1] != 3:
        raise ValueError(f"Expected dirs of shape (N, 3), got {tuple(gaussian_dirs.shape)}")

    # Normalize directions to be safe
    dirs = torch.nn.functional.normalize(gaussian_dirs, dim=-1)
    x, y, z = dirs[:, 0], dirs[:, 1], dirs[:, 2]
    N = dirs.shape[0]

    # Real SH constants up to degree 3 (total 16 bases)
    c0 = 0.28209479177387814

    c1 = 0.4886025119029199

    c2_0 = 1.0925484305920792
    c2_1 = 1.0925484305920792
    c2_2 = 0.31539156525252005
    c2_3 = 1.0925484305920792
    c2_4 = 0.5462742152960396

    c3_0 = 0.5900435899266435
    c3_1 = 2.890611442640554
    c3_2 = 0.4570457994644658
    c3_3 = 0.3731763325901154
    c3_4 = 0.4570457994644658
    c3_5 = 1.445305721320277
    c3_6 = 0.5900435899266435

    # Build SH basis vector Y: (N, 16)
    xx, yy, zz = x * x, y * y, z * z
    xy, yz, xz = x * y, y * z, x * z

    Y = torch.stack([
        # l = 0 (1)
        torch.full_like(x, c0),

        # l = 1 (3)
        -c1 * y,
         c1 * z,
        -c1 * x,

        # l = 2 (5)
         c2_0 * xy,
        -c2_1 * yz,
         c2_2 * (2.0 * zz - xx - yy),
        -c2_3 * xz,
         c2_4 * (xx - yy),

        # l = 3 (7)
        -c3_0 * y * (3.0 * xx - yy),
         c3_1 * xy * z,
        -c3_2 * y * (4.0 * zz - xx - yy),
         c3_3 * z * (2.0 * zz - 3.0 * xx - 3.0 * yy),
        -c3_4 * x * (4.0 * zz - xx - yy),
         c3_5 * z * (xx - yy),
        -c3_6 * x * (xx - 3.0 * yy),
    ], dim=1)  # (N, 16)

    # Reshape coefficients to (N, 16, 3) to match Y ordering
    sh_dc = spherical_harmonics[:, 0:3]                 # (N, 3)
    sh_rest = spherical_harmonics[:, 3:]                # (N, 45)
    sh_rest = sh_rest.view(N, 15, 3)                    # (N, 15, 3)
    sh = torch.cat([sh_dc.unsqueeze(1), sh_rest], dim=1)  # (N, 16, 3)

    # Evaluate: sum_k Y_k * sh_k for each channel
    colours = torch.einsum('nk,nkc->nc', Y, sh)         # (N, 3)

    # Clamp to displayable range
    colours = torch.clamp(colours, 0.0, 1.0)
    return colours

    # Ensure tensor types/placement match directions
    device = gaussian_dirs.device
    sh = spherical_harmonics.to(device=device, dtype=gaussian_dirs.dtype)

    # Reshape to (N, 16, 3): first basis = DC(3), then 15 bases × 3 channels
    dc = sh[:, :3]                               # (N,3)
    rest = sh[:, 3:].view(-1, 15, 3)             # (N,15,3)
    sh_16x3 = torch.cat([dc.unsqueeze(1), rest], dim=1)  # (N,16,3)

    # Evaluate SH bases (N,16)
    bases = _eval_sh_bases_deg3(gaussian_dirs)   # (N,16)

    # Linear combination per channel: sum_b sh[n,b,c] * Y_b(dir_n)
    colours = torch.einsum('nbc,nb->nc', sh_16x3, bases)  # (N,3)

    # Add 0.5 bias (consistent with DC-only path: dc*SH_C0 + 0.5) and clamp
    colours = torch.clamp(colours + 0.5, 0.0, 1.0)

    return colours
