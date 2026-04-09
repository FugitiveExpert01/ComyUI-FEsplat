# ComfyUI-SfM-COLMAP

A ComfyUI custom node that runs **Structure-from-Motion** on a batch of images
using **COLMAP 4.0 with ALIKED + LightGlue** — the best commercially-licensed
SfM pipeline as of 2026.

## Why ALIKED + LightGlue?

| Method | Licence | Speed | Quality on hard scenes |
|---|---|---|---|
| **ALIKED + LightGlue** ✅ | MIT / Apache | Fast | Best — handles low texture, reflections, large viewpoint changes |
| SIFT + LightGlue | MIT | Moderate | Good |
| MASt3R / VGGT | Non-commercial ⚠️ | Fastest | Best overall |

ALIKED + LightGlue is the best option you can legally use in commercial projects.

## Installation

### 1. Install COLMAP 4.0+

```bash
# Linux
sudo apt install colmap

# macOS
brew install colmap

# Windows
# Download from https://github.com/colmap/colmap/releases
# Add the folder containing colmap.exe to your PATH
```

### 2. Install this node

Drop the `comfyui-sfm-node/` folder into:
```
ComfyUI/custom_nodes/comfyui-sfm-node/
```

Then install Python deps:
```bash
pip install -r custom_nodes/comfyui-sfm-node/requirements.txt
```

### 3. Restart ComfyUI

The node appears under **SplatNode > SfM** as:
`SfM · COLMAP (ALIKED + LightGlue)`

## Node Inputs

| Input | Type | Description |
|---|---|---|
| `images` | IMAGE | ComfyUI batch tensor from any image loader |
| `feature_type` | dropdown | `ALIKED` (recommended) or `SIFT` |
| `matcher_type` | dropdown | `lightglue` (recommended), `exhaustive`, `sequential`, `vocab_tree` |
| `mapper` | dropdown | `incremental` (quality) or `glomap` (2–10× faster) |
| `camera_model` | dropdown | `SIMPLE_RADIAL` for most cameras; `OPENCV` for wide-angle |
| `output_dir` | string | Where to save COLMAP database, images, and sparse model |
| `image_folder_override` | string | (Optional) Point at an existing image folder on disk instead |

## Node Outputs

| Output | Type | Description |
|---|---|---|
| `sfm_result` | SFM_RESULT | Dict with paths, camera poses, point cloud array — feed to training nodes |
| `status_log` | STRING | Full log of what ran — connect to a text display node |

## Typical Workflow

```
Load Images ──► SfM · COLMAP (ALIKED + LightGlue) ──► Train_Vanilla3DGS ──► Export PLY
```

## Mapper Choice Guide

- **incremental** — default, best quality, handles tricky scenes
- **glomap** — use for clean, well-overlapping datasets where you want speed

## When to Use SIFT Instead of ALIKED

- Very large datasets (1000+ images) where ALIKED GPU memory is a concern
- Scenes with very uniform textures where learned features don't generalise
- Debugging / compatibility testing

## Output Files

After running, `output_dir` will contain:
```
output/sfm/
├── images/           ← your input images (if not using override)
├── database.db       ← COLMAP feature database
├── sparse/0/         ← binary sparse model
├── sparse_txt/       ← human-readable cameras.txt, images.txt, points3D.txt
└── sfm_summary.json  ← JSON summary of results + config used
```
