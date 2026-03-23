# import argparse
# import os

# import cv2
# import numpy as np
# import torch
# import torch.nn.functional as F

# # ==============================
# # Constants (from your message)
# # ==============================
# FOCAL_LENGTH = 4285.7      # in pixels
# BASELINE = 0.76          # in meters

# CHECKPOINT_PATH = "/mnt/c/Users/prath/OneDrive/Desktop/Assignments/Thesis/CNN/PSMNet/best_checkpoint5.tar"


# def build_model(device):
#     """
#     Build PSMNet model and load weights.
#     Adjust the import below to match your repo structure if needed.
#     """
#     # If your stackhourglass is in models/stackhourglass.py with class PSMNet:
#     from models.stackhourglass import PSMNet
#     model = PSMNet(maxdisp=192)

#     checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)

#     # Try common checkpoint formats
#     if isinstance(checkpoint, dict):
#         if "state_dict" in checkpoint:
#             state_dict = checkpoint["state_dict"]
#         elif "model" in checkpoint:
#             state_dict = checkpoint["model"]
#         else:
#             state_dict = checkpoint
#     else:
#         state_dict = checkpoint

#     # Remove "module." prefix if it was trained with DataParallel
#     new_state_dict = {}
#     for k, v in state_dict.items():
#         if k.startswith("module."):
#             new_state_dict[k[7:]] = v
#         else:
#             new_state_dict[k] = v

#     model.load_state_dict(new_state_dict, strict=False)
#     model.to(device)
#     model.eval()
#     return model


# def load_image_as_tensor(path, device):
#     """
#     Load an RGB image, convert to float tensor [1,3,H,W].
#     Adjust normalization to match how you trained PSMNet.
#     """
#     img = cv2.imread(path, cv2.IMREAD_COLOR)
#     if img is None:
#         raise FileNotFoundError(f"Could not read image: {path}")

#     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     img = img.astype(np.float32) / 255.0

#     # If you used ImageNet normalization during training, uncomment this:
#     # mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
#     # std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
#     # img = (img - mean) / std

#     img = np.transpose(img, (2, 0, 1))  # HWC -> CHW
#     tensor = torch.from_numpy(img).unsqueeze(0).to(device)  # [1,3,H,W]
#     return tensor


# def disparity_to_depth(disparity, focal_length=FOCAL_LENGTH, baseline=BASELINE):
#     """
#     disparity: numpy array [H, W], in pixels
#     Returns depth in meters: depth = f * B / d
#     """
#     disp = disparity.astype(np.float32)
#     depth = np.zeros_like(disp, dtype=np.float32)

#     valid = disp > 0.0
#     depth[valid] = focal_length * baseline / disp[valid]

#     return depth


# def save_depth_maps(depth_m, out_prefix="depth"):
#     """
#     Save:
#       1) depth in millimeters as 16-bit PNG (depth_mm.png)
#       2) normalized depth visualization as 8-bit PNG (depth_vis.png)
#     """
#     # 1) Metric depth in mm (16-bit)
#     depth_mm = depth_m * 1000.0  # m -> mm
#     depth_mm = np.clip(depth_mm, 0, 65535).astype(np.uint16)
#     cv2.imwrite(f"{out_prefix}_mm.png", depth_mm)

#     # 2) Visualization (normalize for display)
#     valid = depth_m > 0
#     if np.any(valid):
#         max_depth = np.percentile(depth_m[valid], 99)  # robust max
#         vis = np.zeros_like(depth_m, dtype=np.float32)
#         vis[valid] = depth_m[valid] / (max_depth + 1e-8)
#         vis = np.clip(vis, 0.0, 1.0)
#     else:
#         vis = depth_m

#     vis = (vis * 255.0).astype(np.uint8)
#     cv2.imwrite(f"{out_prefix}_vis.png", vis)


# def run_inference(left_path, right_path, out_prefix):
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     model = build_model(device)

#     left = load_image_as_tensor(left_path, device)   # [1,3,H,W]
#     right = load_image_as_tensor(right_path, device) # [1,3,H,W]

#     # ---------------------------
#     # PADDING TO MULTIPLE OF 16
#     # ---------------------------
#     _, _, H, W = left.shape
#     pad_h = (16 - H % 16) % 16
#     pad_w = (16 - W % 16) % 16

#     # F.pad pads in (left, right, top, bottom) order for 4D tensors
#     if pad_h != 0 or pad_w != 0:
#         left_padded = F.pad(left, (0, pad_w, 0, pad_h))
#         right_padded = F.pad(right, (0, pad_w, 0, pad_h))
#     else:
#         left_padded = left
#         right_padded = right

#     with torch.no_grad():
#         disp = model(left_padded, right_padded)

#     # Handle common output shapes
#     # PSMNet often returns [B, H, W] or [B, 1, H, W]
#     if disp.dim() == 4:
#         # [B,1,H,W] or [B,C,H,W]
#         disp = disp[:, 0, :, :]
#     disp = disp.squeeze(0)  # [H_padded, W_padded]

#     # ---------------------------
#     # CROP BACK TO ORIGINAL SIZE
#     # ---------------------------
#     disp = disp[:H, :W].cpu().numpy()  # [H, W]

#     # Convert disparity -> depth (meters)
#     depth_m = disparity_to_depth(disp)

#     # Save depth maps
#     save_depth_maps(depth_m, out_prefix=out_prefix)


# def main():
#     parser = argparse.ArgumentParser(description="PSMNet depth map export")
#     parser.add_argument("--left", required=True, help="Path to left image")
#     parser.add_argument("--right", required=True, help="Path to right image")
#     parser.add_argument(
#         "--out_prefix",
#         default="depth",
#         help="Output prefix for saved depth maps (default: depth)",
#     )
#     args = parser.parse_args()

#     run_inference(args.left, args.right, args.out_prefix)


# if __name__ == "__main__":
#     main()

import argparse
import os

import cv2
import numpy as np
import torch
import torch.nn.functional as F

# ==============================
# Constants
# ==============================
FOCAL_LENGTH = 4285.7
BASELINE = 0.76

CHECKPOINT_PATH = "/mnt/c/Users/prath/OneDrive/Desktop/Assignments/Thesis/CNN/PSMNet/best_checkpoint5.tar"


def build_model(device):
    from models.stackhourglass import PSMNet
    model = PSMNet(maxdisp=192)

    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)

    if isinstance(checkpoint, dict):
        if "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        elif "model" in checkpoint:
            state_dict = checkpoint["model"]
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint

    # Remove DataParallel prefixes
    new_state_dict = {}
    for k, v in state_dict.items():
        new_state_dict[k[7:] if k.startswith("module.") else k] = v

    model.load_state_dict(new_state_dict, strict=False)
    model.to(device)
    model.eval()
    return model


def load_image_as_tensor(path, device):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0

    img = np.transpose(img, (2, 0, 1))  # HWC→CHW
    tensor = torch.from_numpy(img).unsqueeze(0).to(device)
    return tensor


def disparity_to_depth(disparity, focal_length=FOCAL_LENGTH, baseline=BASELINE):
    disp = disparity.astype(np.float32)
    depth = np.zeros_like(disp, dtype=np.float32)

    valid = disp > 0
    depth[valid] = focal_length * baseline / disp[valid]

    return depth


def save_disparity_npy(disparity, out_prefix):
    """Save disparity map as .npy"""
    np.save(f"{out_prefix}_disp.npy", disparity)
    print(f"[Saved disparity] → {out_prefix}_disp.npy")


def save_depth_maps(depth_m, out_prefix):
    depth_mm = (depth_m * 1000.0).clip(0, 65535).astype(np.uint16)
    cv2.imwrite(f"{out_prefix}_mm.png", depth_mm)

    valid = depth_m > 0
    if np.any(valid):
        max_depth = np.percentile(depth_m[valid], 99)
        vis = np.zeros_like(depth_m, dtype=np.float32)
        vis[valid] = depth_m[valid] / (max_depth + 1e-8)
    else:
        vis = depth_m

    vis = np.clip(vis, 0, 1)
    vis = (vis * 255).astype(np.uint8)
    cv2.imwrite(f"{out_prefix}_vis.png", vis)


def run_inference(left_path, right_path, out_prefix):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(device)

    left = load_image_as_tensor(left_path, device)
    right = load_image_as_tensor(right_path, device)

    _, _, H, W = left.shape
    pad_h = (16 - H % 16) % 16
    pad_w = (16 - W % 16) % 16

    left_padded = F.pad(left, (0, pad_w, 0, pad_h))
    right_padded = F.pad(right, (0, pad_w, 0, pad_h))

    with torch.no_grad():
        disp = model(left_padded, right_padded)

    if disp.dim() == 4:
        disp = disp[:, 0, :, :]

    disp = disp.squeeze(0)
    disp = disp[:H, :W].cpu().numpy()

    # ==========================
    # NEW: SAVE DISPARITY AS .NPY
    # ==========================
    save_disparity_npy(disp, out_prefix)

    # Convert to depth
    depth_m = disparity_to_depth(disp)
    save_depth_maps(depth_m, out_prefix)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--left", required=True)
    parser.add_argument("--right", required=True)
    parser.add_argument("--out_prefix", default="output")
    args = parser.parse_args()

    run_inference(args.left, args.right, args.out_prefix)


if __name__ == "__main__":
    main()
