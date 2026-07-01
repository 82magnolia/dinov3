# DINOv3 Keypoint Matching — Gradio Demo

A local web demo that lets you drag-and-drop two images and instantly see
DINOv3-based dense and sparse visual correspondences.

## What it does

For a pair of uploaded images the app:

1. Extracts patch-level features from a DINOv3 ViT backbone.
2. Finds the best-matching patch in the right image for every patch in the left
   image using cosine similarity.
3. **Sparse view** — draws coloured lines connecting matched keypoints on the
   original images.
4. **Dense view** — colours every patch by its position in a shared 3-component
   PCA feature space so that semantically matching regions share the same colour.

---

## Prerequisites

You need Python 3.9+ and a working copy of this repository.

### 1 — Install the DINOv3 package

From the **repository root**:

```bash
pip install -e .
```

### 2 — Install demo dependencies

```bash
pip install gradio scikit-learn matplotlib
```

If you are using the **`pm_touch` conda environment**, most dependencies are
already present. Only install the missing packages:

```bash
conda activate pm_touch
pip install gradio torchmetrics termcolor
```

GPU inference is strongly recommended. The demo falls back to CPU automatically,
but feature extraction on CPU is noticeably slower (expect ~5–15 s per image
pair with ViT-B/16 on a modern CPU).

If you have a CUDA GPU, make sure the PyTorch version installed matches your
CUDA toolkit:

```bash
# Example for CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### 3 — Download model weights (required)

DINOv3 weights are **gated by Meta** and cannot be auto-downloaded. You must
request access and download them manually:

1. Fill out the access form at:
   **https://ai.meta.com/resources/models-and-libraries/dinov3-downloads/**

2. After approval you will receive an email with download URLs. Use `wget` to
   download (a browser will not work):

   ```bash
   # Example for ViT-B/16 (default model used by the demo)
   wget -O ~/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth \
       "<URL from the email>"
   ```

   Download the file that matches the model you plan to use in the UI:

   | UI label | Expected filename pattern |
   |---|---|
   | ViT-S/16 (fastest) | `dinov3_vits16_pretrain_lvd1689m-*.pth` |
   | ViT-S/16+ (fast) | `dinov3_vits16plus_pretrain_lvd1689m-*.pth` |
   | ViT-B/16 (balanced) | `dinov3_vitb16_pretrain_lvd1689m-*.pth` |
   | ViT-L/16 (best quality) | `dinov3_vitl16_pretrain_lvd1689m-*.pth` |

3. Note the full local path to the downloaded file — you will paste it into
   the **"Path to weights"** text box in the app.

---

## Running the app

From the **repository root** (or from inside `demo/`):

```bash
# from repo root
python demo/app.py

# or
cd demo && python app.py
```

Gradio will print a local URL such as `http://127.0.0.1:7860` — open it in
your browser.

To expose the demo on your local network (e.g. to access it from another
machine):

```bash
python demo/app.py --share   # generates a temporary public tunnel URL
```

> The `--share` flag is passed through to Gradio's `launch()`. Alternatively
> edit `demo.launch(share=True)` at the bottom of `app.py`.

---

## Model options

| UI label | Architecture | Embed dim | Layers | Notes |
|---|---|---|---|---|
| ViT-S/16 (fastest) | ViT-Small | 384 | 12 | Good for quick experiments |
| ViT-S/16+ (fast) | ViT-Small+ | 384 | 12 | SwiGLU FFN variant |
| ViT-B/16 (balanced) | ViT-Base | 768 | 12 | Default; good speed/quality trade-off |
| ViT-L/16 (best quality) | ViT-Large | 1024 | 24 | Best results, slower |

Weights are downloaded automatically from Meta's CDL on first use and cached
in `~/.cache/torch/hub/checkpoints/`.

---

## Controls

| Control | Description |
|---|---|
| **Left / Right Image** | Drag-and-drop or click to upload; paste from clipboard also works |
| **Model** | Backbone to use for feature extraction |
| **Max sparse keypoints** | How many correspondence lines to draw (10–200) |
| **Path to weights (.pth)** | Full local path to the downloaded DINOv3 checkpoint; must match the selected model |
| **Find Correspondences** | Runs the pipeline and displays both outputs |

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| `ModuleNotFoundError: No module named 'dinov3'` | Run `pip install -e .` from the repo root |
| `ModuleNotFoundError: No module named 'gradio'` | Run `pip install gradio` |
| `HTTP Error 403` when loading weights | The weights are gated — follow Step 3 above to request access and download manually |
| "Weights file not found" error in the UI | The path in the text box doesn't point to an existing file; check the path |
| Wrong architecture error / key mismatch | The `.pth` file doesn't match the selected model; choose the matching model in the dropdown |
| Out-of-memory on GPU | Switch to a smaller model (ViT-S) or reduce `IMAGE_SIZE` in `app.py` |
| Slow on CPU | Expected; use a GPU or choose ViT-S/16 |
