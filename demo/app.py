import os
import sys

import gradio as gr
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image
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
STRATIFY_DISTANCE_THRESHOLD = 60.0

MODEL_OPTIONS = {
    "ViT-S/16 (fastest)": ("dinov3_vits16", 12),
    "ViT-S/16+ (fast)": ("dinov3_vits16plus", 12),
    "ViT-B/16 (balanced)": ("dinov3_vitb16", 12),
    "ViT-L/16 (best quality)": ("dinov3_vitl16", 24),
}

_model_cache: dict = {}


def _get_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def _load_model(model_name: str, weights_path: str):
    cache_key = (model_name, weights_path)
    if cache_key not in _model_cache:
        if not weights_path or not os.path.isfile(weights_path):
            raise gr.Error(
                f"Weights file not found: '{weights_path}'. "
                "Please request access at https://ai.meta.com/resources/models-and-libraries/dinov3-downloads/ "
                "and provide the path to the downloaded .pth file."
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
    dtype = torch.float32
    with torch.inference_mode():
        with torch.autocast(device_type=device.split(":")[0], dtype=dtype):
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
):
    if image_left is None or image_right is None:
        raise gr.Error("Please upload both a left and a right image.")

    model_name, n_layers = MODEL_OPTIONS[model_choice]
    model, device = _load_model(model_name, weights_path.strip())

    feat_l = _extract_features(model, image_left, n_layers, device)   # [D, H1, W1]
    feat_r = _extract_features(model, image_right, n_layers, device)  # [D, H2, W2]
    dim = feat_l.shape[0]

    feat_l_n = F.normalize(feat_l, p=2, dim=0)
    feat_r_n = F.normalize(feat_r, p=2, dim=0)

    # For each left patch find the best-matching right patch
    heatmaps = torch.einsum(
        "k f, f h w -> k h w",
        feat_l_n.view(dim, -1).T,   # [N1, D]
        feat_r_n,                    # [D, H2, W2]
    )  # [N1, H2, W2]

    h1, w1 = feat_l.shape[1], feat_l.shape[2]
    h2, w2 = feat_r.shape[1], feat_r.shape[2]
    n1 = h1 * w1

    idx_l = torch.arange(n1)
    # patch-centre pixel coords in the resized image
    locs_l = (torch.stack([idx_l // w1, idx_l % w1], dim=-1).float() + 0.5) * PATCH_SIZE  # [N1, 2]

    idx_r = heatmaps.flatten(-2).argmax(-1)  # [N1]
    locs_r = (torch.stack([idx_r // w2, idx_r % w2], dim=-1).float() + 0.5) * PATCH_SIZE  # [N1, 2]

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

    # ---------- Sparse figure ----------
    scale_l = image_left.height / IMAGE_SIZE
    scale_r = image_right.height / IMAGE_SIZE

    keep = _stratify_points(locs_l * scale_l, STRATIFY_DISTANCE_THRESHOLD ** 2)
    if len(keep) > num_points:
        rng = np.random.default_rng(42)
        keep = np.sort(rng.choice(keep, size=num_points, replace=False))

    pts_l = locs_l[keep].numpy()   # [K, 2]  (row, col) in resized space
    pts_r = locs_r[keep].numpy()

    # colour from PCA map
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
    return fig_sparse, fig_dense


# ─────────────────────────── Gradio UI ───────────────────────────

DESCRIPTION = """
# DINOv3 Keypoint Matching

Upload a **left** and **right** image of the same object or scene and click
**Find Correspondences**. The demo uses DINOv3 patch features to compute
dense and sparse visual correspondences without any task-specific fine-tuning.

* **Sparse**: coloured lines connect matched keypoints between the two images.
* **Dense**: patches are coloured by their position in PCA feature space — matching patches share the same colour.

> **Note**: DINOv3 weights are gated by Meta. Request access at
> [ai.meta.com/resources/models-and-libraries/dinov3-downloads](https://ai.meta.com/resources/models-and-libraries/dinov3-downloads/)
> and paste the local `.pth` path below.
"""

with gr.Blocks(title="DINOv3 Keypoint Matching", theme=gr.themes.Soft()) as demo:
    gr.Markdown(DESCRIPTION)

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
            value=60,
        )

    weights_tb = gr.Textbox(
        label="Path to weights (.pth)",
        placeholder="/path/to/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth",
        info="Local path to the downloaded DINOv3 checkpoint. Must match the selected model architecture.",
    )

    run_btn = gr.Button("Find Correspondences", variant="primary")

    sparse_out = gr.Plot(label="Sparse Correspondences")
    with gr.Accordion("Dense Correspondences (PCA colour map)", open=False):
        dense_out = gr.Plot(label="Dense Correspondences")

    run_btn.click(
        fn=run_matching,
        inputs=[img_left, img_right, model_dd, weights_tb, n_pts_sl],
        outputs=[sparse_out, dense_out],
    )

if __name__ == "__main__":
    demo.launch()
