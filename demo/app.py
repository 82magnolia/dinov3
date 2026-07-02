import os
import sys

import cv2
import gradio as gr
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image
from scipy.interpolate import RBFInterpolator
from scipy.ndimage import map_coordinates
from sklearn.decomposition import PCA
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

PATCH_SIZE = 16
IMAGE_SIZE = 448
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

MODEL_OPTIONS = {
    "ViT-S/16 (fastest)": ("dinov3_vits16", 12),
    "ViT-S/16+ (fast)": ("dinov3_vits16plus", 12),
    "ViT-B/16 (balanced)": ("dinov3_vitb16", 12),
    "ViT-L/16 (best quality)": ("dinov3_vitl16", 24),
}

PRETRAINED_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pretrained")

_model_cache: dict = {}


def _get_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def _scan_pretrained() -> list[str]:
    """Return full paths of all .pth files inside demo/pretrained/, sorted by name."""
    if not os.path.isdir(PRETRAINED_DIR):
        return []
    return sorted(
        os.path.join(PRETRAINED_DIR, f)
        for f in os.listdir(PRETRAINED_DIR)
        if f.endswith(".pth")
    )


def _load_model(model_name: str, weights_path: str):
    cache_key = (model_name, weights_path)
    if cache_key not in _model_cache:
        if not weights_path or not os.path.isfile(weights_path):
            raise gr.Error(
                f"Weights file not found: '{weights_path}'. "
                "Drop a .pth file into demo/pretrained/ and click Refresh."
            )
        device = _get_device()
        repo_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        model = torch.hub.load(
            repo_or_dir=repo_dir,
            model=model_name,
            source="local",
            weights=weights_path,
        )
        model.eval()
        model.to(device)
        _model_cache[cache_key] = model
    return _model_cache[cache_key], _get_device()


def _resize(image: Image.Image) -> torch.Tensor:
    w, h = image.size
    h_p = IMAGE_SIZE // PATCH_SIZE
    w_p = round((w * IMAGE_SIZE) / (h * PATCH_SIZE))
    return TF.to_tensor(TF.resize(image, (h_p * PATCH_SIZE, w_p * PATCH_SIZE)))


def _extract_features(model, image: Image.Image, n_layers: int, device: str) -> torch.Tensor:
    img = _resize(image.convert("RGB"))
    img = TF.normalize(img, mean=IMAGENET_MEAN, std=IMAGENET_STD).unsqueeze(0).to(device)
    with torch.inference_mode():
        with torch.autocast(device_type=device.split(":")[0], dtype=torch.float32):
            feats = model.get_intermediate_layers(img, n=range(n_layers), reshape=True, norm=True)
    return feats[-1].squeeze().detach().cpu()  # [D, H, W]


def _stratify_points(pts: torch.Tensor, threshold: float) -> np.ndarray:
    n = len(pts)
    sentinel = threshold + 1.0
    sq_norms = (pts * pts).sum(dim=1)
    dists = -2.0 * pts @ pts.T
    dists.add_(sq_norms[:, None]).add_(sq_norms[None, :])
    dists.fill_diagonal_(sentinel)
    keep = np.ones(n, dtype=bool)
    mask = (dists <= threshold).float()
    ones = torch.ones(n)
    counts = mask @ ones
    while counts.any():
        worst = int(counts.argmax())
        keep[worst] = False
        dists[worst, :] = sentinel
        dists[:, worst] = sentinel
        mask = (dists <= threshold).float()
        counts = mask @ ones
    return np.where(keep)[0]


def run_matching(
    image_left: Image.Image,
    image_right: Image.Image,
    model_choice: str,
    weights_path: str,
    num_points: int,
    stratify_threshold: float,
):
    if image_left is None or image_right is None:
        raise gr.Error("Please upload both a left and a right image.")

    model_name, n_layers = MODEL_OPTIONS[model_choice]
    model, device = _load_model(model_name, weights_path or "")

    feat_l = _extract_features(model, image_left, n_layers, device)   # [D, H1, W1]
    feat_r = _extract_features(model, image_right, n_layers, device)  # [D, H2, W2]
    dim = feat_l.shape[0]

    feat_l_n = F.normalize(feat_l, p=2, dim=0)
    feat_r_n = F.normalize(feat_r, p=2, dim=0)

    heatmaps = torch.einsum(
        "k f, f h w -> k h w",
        feat_l_n.view(dim, -1).T,   # [N1, D]
        feat_r_n,                    # [D, H2, W2]
    )  # [N1, H2, W2]

    h1, w1 = feat_l.shape[1], feat_l.shape[2]
    h2, w2 = feat_r.shape[1], feat_r.shape[2]
    n1 = h1 * w1

    idx_l = torch.arange(n1)
    locs_l = (torch.stack([idx_l // w1, idx_l % w1], dim=-1).float() + 0.5) * PATCH_SIZE  # [N1, 2] (row,col) resized

    idx_r = heatmaps.flatten(-2).argmax(-1)  # [N1]
    locs_r = (torch.stack([idx_r // w2, idx_r % w2], dim=-1).float() + 0.5) * PATCH_SIZE  # [N1, 2] (row,col) resized

    # ---------- PCA colour map ----------
    x_l = feat_l.view(dim, -1).T.numpy()   # [N1, D]
    x_r = feat_r.view(dim, -1).T.numpy()   # [N2, D]

    pca = PCA(n_components=3, whiten=True)
    pca.fit(x_l)

    def to_rgb(x, h, w):
        proj = torch.from_numpy(pca.transform(x)).view(h, w, 3)
        return torch.sigmoid(proj * 2.0).permute(2, 0, 1)  # [3, H, W]

    rgb_l = to_rgb(x_l, h1, w1)
    rgb_r = to_rgb(x_r, h2, w2)

    # ---------- Dense figure ----------
    fig_dense, (da1, da2) = plt.subplots(1, 2, figsize=(12, 5), dpi=120)
    da1.imshow(rgb_l.permute(1, 2, 0).numpy())
    da1.set_title("Left — Dense Correspondences", fontsize=11)
    da1.axis("off")
    da2.imshow(rgb_r.permute(1, 2, 0).numpy())
    da2.set_title("Right — Dense Correspondences", fontsize=11)
    da2.axis("off")
    plt.tight_layout()

    # ---------- Sparse selection ----------
    scale_l = image_left.height / IMAGE_SIZE
    scale_r = image_right.height / IMAGE_SIZE

    keep = _stratify_points(locs_l * scale_l, stratify_threshold ** 2)
    if len(keep) > num_points:
        rng = np.random.default_rng(42)
        keep = np.sort(rng.choice(keep, size=num_points, replace=False))

    pts_l = locs_l[keep].numpy()   # [K, 2] (row, col) in resized space
    pts_r = locs_r[keep].numpy()

    # Convert to original image pixel space and store for RANSAC stage
    pts_l_orig = pts_l * scale_l   # [K, 2] (row, col) original px
    pts_r_orig = pts_r * scale_r
    new_match_state = {
        "pts_l": pts_l_orig,
        "pts_r": pts_r_orig,
        "img_left": image_left,
        "img_right": image_right,
    }

    # ---------- Sparse figure ----------
    ri = (pts_l[:, 0] / PATCH_SIZE).astype(int).clip(0, h1 - 1)
    ci = (pts_l[:, 1] / PATCH_SIZE).astype(int).clip(0, w1 - 1)
    colors = rgb_l[:, ri, ci].T.numpy()  # [K, 3]

    fig_sparse, (sa1, sa2) = plt.subplots(1, 2, figsize=(16, 8), dpi=120)
    sa1.imshow(image_left)
    sa1.set_title("Left Image", fontsize=11)
    sa1.axis("off")
    sa2.imshow(image_right)
    sa2.set_title("Right Image", fontsize=11)
    sa2.axis("off")

    for (row_l, col_l), (row_r, col_r), color in zip(pts_l, pts_r, colors):
        xl = col_l * scale_l
        yl = row_l * scale_l
        xr = col_r * scale_r
        yr = row_r * scale_r
        con = ConnectionPatch(
            xyA=(xl, yl), xyB=(xr, yr),
            coordsA="data", coordsB="data",
            axesA=sa1, axesB=sa2,
            color=color, linewidth=0.8, alpha=0.85,
        )
        sa2.add_artist(con)
        sa1.plot(xl, yl, "o", color=color, markersize=4)
        sa2.plot(xr, yr, "o", color=color, markersize=4)

    plt.tight_layout()
    return fig_sparse, fig_dense, new_match_state


def fit_and_warp(state: dict, transform_type: str, reproj_threshold: float):
    if state is None or state.get("pts_l") is None:
        raise gr.Error("Run 'Find Correspondences' first.")

    pts_l_rc = state["pts_l"]    # [K, 2] (row, col) float64, original px
    pts_r_rc = state["pts_r"]    # [K, 2] (row, col) float64, original px
    img_left  = state["img_left"]
    img_right = state["img_right"]

    K = len(pts_l_rc)
    img_left_np  = np.array(img_left.convert("RGB"))   # H1 x W1 x 3 uint8
    img_right_np = np.array(img_right.convert("RGB"))  # H2 x W2 x 3 uint8

    # cv2 uses (x,y) = (col,row) and float32
    pts_l_xy = pts_l_rc[:, ::-1].astype(np.float32)
    pts_r_xy = pts_r_rc[:, ::-1].astype(np.float32)
    dsize = (img_right.width, img_right.height)  # cv2 (width, height)
    h2, w2 = img_right.height, img_right.width

    if transform_type == "Affine":
        M, mask = cv2.estimateAffine2D(
            pts_l_xy, pts_r_xy,
            method=cv2.RANSAC,
            ransacReprojThreshold=reproj_threshold,
        )
        if M is None:
            raise gr.Error("Affine RANSAC failed — too few or degenerate matches.")
        inlier_count = int(mask.sum())
        warped = cv2.warpAffine(img_left_np, M, dsize)

    elif transform_type == "Homography":
        H, mask = cv2.findHomography(
            pts_l_xy, pts_r_xy,
            cv2.RANSAC,
            ransacReprojThreshold=reproj_threshold,
        )
        if H is None:
            raise gr.Error("Homography RANSAC failed — too few or degenerate matches.")
        inlier_count = int(mask.sum())
        warped = cv2.warpPerspective(img_left_np, H, dsize)

    elif transform_type in ("RBF (Affine init)", "RBF (Homography init)"):
        if transform_type == "RBF (Affine init)":
            M_init, mask = cv2.estimateAffine2D(
                pts_l_xy, pts_r_xy,
                method=cv2.RANSAC,
                ransacReprojThreshold=reproj_threshold,
            )
        else:
            M_init, mask = cv2.findHomography(
                pts_l_xy, pts_r_xy,
                cv2.RANSAC,
                ransacReprojThreshold=reproj_threshold,
            )
        if M_init is None or mask is None:
            raise gr.Error(f"{transform_type} RANSAC (inlier selection for RBF) failed.")
        inlier_bool = mask.ravel().astype(bool)
        inlier_count = int(inlier_bool.sum())
        if inlier_count < 4:
            raise gr.Error(f"Too few inliers ({inlier_count}) to fit RBF — need at least 4.")

        pts_l_in = pts_l_rc[inlier_bool]   # [I, 2] (row, col)
        pts_r_in = pts_r_rc[inlier_bool]   # [I, 2] (row, col)

        # Inverse mapping: for each right-image pixel, where does it come from in left?
        rbf = RBFInterpolator(
            pts_r_in,
            pts_l_in - pts_r_in,   # displacement: right -> left
            kernel="thin_plate_spline",
        )

        rows = np.arange(h2, dtype=np.float64)
        cols = np.arange(w2, dtype=np.float64)
        grid_col, grid_row = np.meshgrid(cols, rows)   # H2 x W2 each
        query_rc = np.column_stack([grid_row.ravel(), grid_col.ravel()])  # [H2*W2, 2]

        disp = rbf(query_rc)   # [H2*W2, 2] (delta_row, delta_col)
        src_row = query_rc[:, 0] + disp[:, 0]
        src_col = query_rc[:, 1] + disp[:, 1]

        warped = np.stack([
            map_coordinates(
                img_left_np[:, :, c],
                [src_row, src_col],
                order=1,
                mode="nearest",
            ).reshape(h2, w2)
            for c in range(3)
        ], axis=-1).astype(np.uint8)

    else:
        raise gr.Error(f"Unknown transform type: {transform_type}")

    inlier_text = f"{inlier_count} / {K} inliers"
    # Return warped twice: once for display, once for warped_state
    return warped, warped, inlier_text


def blend_images(warped_state, img_right: Image.Image, alpha: float):
    if warped_state is None or img_right is None:
        return None
    right_np = np.array(img_right.convert("RGB"))
    blended = np.clip(
        alpha * warped_state.astype(np.float32)
        + (1.0 - alpha) * right_np.astype(np.float32),
        0, 255,
    ).astype(np.uint8)
    return blended


# ─────────────────────────── Gradio UI ───────────────────────────

DESCRIPTION = """
# DINOv3 Keypoint Matching

Upload a **left** and **right** image of the same object or scene and click
**Find Correspondences**. The demo uses DINOv3 patch features to compute
dense and sparse visual correspondences without any task-specific fine-tuning.

* **Sparse**: coloured lines connect matched keypoints between the two images.
* **Dense**: patches are coloured by their position in PCA feature space — matching patches share the same colour.
* **Geometric Warp**: fit an Affine, Homography, or RBF warp via RANSAC and overlay the result.

> **Note**: DINOv3 weights are gated by Meta. Request access at
> [ai.meta.com/resources/models-and-libraries/dinov3-downloads](https://ai.meta.com/resources/models-and-libraries/dinov3-downloads/),
> download the `.pth` file, place it in `demo/pretrained/`, then click **Refresh**.
"""

with gr.Blocks(title="DINOv3 Keypoint Matching", theme=gr.themes.Soft()) as demo:
    gr.Markdown(DESCRIPTION)

    match_state  = gr.State(value=None)
    warped_state = gr.State(value=None)

    with gr.Row():
        img_left = gr.Image(
            label="Left Image",
            type="pil",
            sources=["upload", "clipboard"],
            height=320,
        )
        img_right = gr.Image(
            label="Right Image",
            type="pil",
            sources=["upload", "clipboard"],
            height=320,
        )

    with gr.Row():
        model_dd = gr.Dropdown(
            label="Model",
            choices=list(MODEL_OPTIONS.keys()),
            value="ViT-B/16 (balanced)",
        )
        n_pts_sl = gr.Slider(
            label="Max sparse keypoints",
            minimum=10,
            maximum=200,
            step=10,
            value=100,
        )
        stratify_sl = gr.Slider(
            label="Stratify threshold (px)",
            minimum=10,
            maximum=200,
            step=5,
            value=20,
        )

    with gr.Row():
        weights_dd = gr.Dropdown(
            label="Weights (demo/pretrained/)",
            choices=_scan_pretrained(),
            value=(_scan_pretrained() or [None])[0],
            info="Place .pth files in demo/pretrained/ and click Refresh to list them.",
            scale=5,
        )
        refresh_btn = gr.Button("Refresh", scale=1)

    refresh_btn.click(
        fn=lambda: gr.Dropdown(choices=_scan_pretrained(), value=(_scan_pretrained() or [None])[0]),
        inputs=[],
        outputs=[weights_dd],
    )

    run_btn = gr.Button("Find Correspondences", variant="primary")

    sparse_out = gr.Plot(label="Sparse Correspondences")
    with gr.Accordion("Dense Correspondences (PCA colour map)", open=False):
        dense_out = gr.Plot(label="Dense Correspondences")

    with gr.Accordion("Geometric Warp (RANSAC)", open=False):
        with gr.Row():
            transform_dd = gr.Dropdown(
                label="Transform type",
                choices=["Affine", "Homography", "RBF (Affine init)", "RBF (Homography init)"],
                value="Affine",
            )
            fit_btn = gr.Button("Fit & Warp", variant="primary")
        reproj_sl = gr.Slider(
            label="RANSAC reprojection threshold (px)",
            minimum=1.0,
            maximum=50.0,
            step=0.5,
            value=3.0,
        )
        inlier_txt = gr.Textbox(label="Inlier count", interactive=False)
        alpha_sl = gr.Slider(
            label="Blend  (1 = warped left,  0 = right only)",
            minimum=0.0,
            maximum=1.0,
            step=0.05,
            value=0.5,
        )
        warp_img_out = gr.Image(
            label="Warped Left overlaid on Right",
            type="numpy",
            interactive=False,
        )

    run_btn.click(
        fn=run_matching,
        inputs=[img_left, img_right, model_dd, weights_dd, n_pts_sl, stratify_sl],
        outputs=[sparse_out, dense_out, match_state],
    )

    fit_btn.click(
        fn=fit_and_warp,
        inputs=[match_state, transform_dd, reproj_sl],
        outputs=[warp_img_out, warped_state, inlier_txt],
    )

    alpha_sl.change(
        fn=blend_images,
        inputs=[warped_state, img_right, alpha_sl],
        outputs=[warp_img_out],
    )

if __name__ == "__main__":
    demo.launch()
