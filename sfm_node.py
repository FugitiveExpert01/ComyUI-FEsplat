"""
ComfyUI-SfM-COLMAP
A custom ComfyUI node that takes a batch of images and runs COLMAP SfM
using ALIKED + LightGlue — the best commercially-licensed pipeline as of 2026.

Install:
  pip install pycolmap opencv-python-headless numpy Pillow

COLMAP binary must be on PATH:
  Windows: https://github.com/colmap/colmap/releases
  Linux:   sudo apt install colmap
  macOS:   brew install colmap
"""

import os
import json
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image


# ── helpers ────────────────────────────────────────────────────────────────────

def tensor_to_pil(image_tensor):
    """Convert ComfyUI IMAGE tensor (B,H,W,C float32 0-1) to list of PIL images."""
    images = []
    # image_tensor shape: (batch, H, W, C)
    batch = image_tensor.shape[0] if len(image_tensor.shape) == 4 else 1
    t = image_tensor if len(image_tensor.shape) == 4 else image_tensor.unsqueeze(0)
    for i in range(batch):
        arr = (t[i].cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
        images.append(Image.fromarray(arr))
    return images


def run(cmd, cwd=None, check=True):
    """Run a subprocess, return stdout. Raises on failure if check=True."""
    result = subprocess.run(
        cmd, cwd=cwd, capture_output=True, text=True
    )
    if check and result.returncode != 0:
        raise RuntimeError(
            f"Command failed: {' '.join(str(c) for c in cmd)}\n"
            f"STDERR:\n{result.stderr}\n"
            f"STDOUT:\n{result.stdout}"
        )
    return result.stdout


def colmap_bin():
    """Locate the colmap binary."""
    binary = shutil.which("colmap")
    if binary is None:
        raise EnvironmentError(
            "COLMAP binary not found on PATH.\n"
            "Install: https://colmap.github.io/install.html\n"
            "  Linux:   sudo apt install colmap\n"
            "  macOS:   brew install colmap\n"
            "  Windows: add colmap.exe folder to PATH"
        )
    return binary


def parse_cameras_txt(cameras_txt: Path) -> list[dict]:
    """Parse COLMAP cameras.txt into a list of dicts."""
    cameras = []
    with open(cameras_txt) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            cam_id = int(parts[0])
            model = parts[1]
            width = int(parts[2])
            height = int(parts[3])
            params = [float(p) for p in parts[4:]]
            cameras.append({
                "camera_id": cam_id,
                "model": model,
                "width": width,
                "height": height,
                "params": params,
            })
    return cameras


def parse_images_txt(images_txt: Path) -> list[dict]:
    """
    Parse COLMAP images.txt.
    Each registered image has: IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID NAME
    followed by a line of 2D points (we skip it).
    Returns list of pose dicts.
    """
    poses = []
    with open(images_txt) as f:
        lines = [l.strip() for l in f if l.strip() and not l.startswith("#")]
    i = 0
    while i < len(lines):
        parts = lines[i].split()
        if len(parts) < 9:
            i += 1
            continue
        image_id = int(parts[0])
        qw, qx, qy, qz = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
        tx, ty, tz = float(parts[5]), float(parts[6]), float(parts[7])
        camera_id = int(parts[8])
        name = parts[9] if len(parts) > 9 else f"image_{image_id}"
        poses.append({
            "image_id": image_id,
            "camera_id": camera_id,
            "name": name,
            "quaternion_wxyz": [qw, qx, qy, qz],
            "translation_xyz": [tx, ty, tz],
        })
        i += 2  # skip the keypoint line
    return poses


def parse_points3d_txt(points_txt: Path) -> dict:
    """Parse COLMAP points3D.txt. Returns summary stats + raw point array."""
    points = []
    with open(points_txt) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            # POINT3D_ID X Y Z R G B ERROR TRACK[]
            if len(parts) >= 7:
                x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                points.append([x, y, z])
    arr = np.array(points, dtype=np.float32) if points else np.zeros((0, 3), dtype=np.float32)
    return {
        "num_points": len(points),
        "points_xyz": arr,
        "centroid": arr.mean(axis=0).tolist() if len(points) > 0 else [0, 0, 0],
    }


# ── main node class ─────────────────────────────────────────────────────────────

class SfM_COLMAP_ALIKED:
    """
    Structure-from-Motion using COLMAP 4.0 with ALIKED + LightGlue.

    Best commercially-licensed SfM pipeline as of 2026:
      - ALIKED: learned local features, robust on low-texture / reflective scenes
      - LightGlue: transformer-based matcher, high inlier ratio
      - GLOMAP mapper option: 2-10× faster for clean datasets
      - Outputs: camera poses JSON + point cloud numpy array
    """

    CATEGORY = "SplatNode/SfM"

    RETURN_TYPES  = ("SFM_RESULT", "STRING")
    RETURN_NAMES  = ("sfm_result",  "status_log")

    FUNCTION = "run_sfm"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),  # ComfyUI batch tensor (B,H,W,C)

                # ── Feature extraction ──────────────────────────
                "feature_type": (
                    ["ALIKED", "SIFT"],
                    {"default": "ALIKED"},
                ),

                # ── Matching ────────────────────────────────────
                "matcher_type": (
                    ["lightglue", "exhaustive", "sequential", "vocab_tree"],
                    {"default": "lightglue"},
                ),

                # ── Reconstruction mapper ────────────────────────
                "mapper": (
                    ["incremental", "glomap"],
                    {"default": "incremental"},
                ),

                # ── Camera model ─────────────────────────────────
                "camera_model": (
                    ["SIMPLE_RADIAL", "OPENCV", "PINHOLE", "SIMPLE_PINHOLE"],
                    {"default": "SIMPLE_RADIAL"},
                ),

                # ── Output path ──────────────────────────────────
                "output_dir": (
                    "STRING",
                    {"default": "output/sfm", "multiline": False},
                ),
            },
            "optional": {
                # Override: point at an existing image folder on disk
                # instead of using the tensor input
                "image_folder_override": (
                    "STRING",
                    {"default": "", "multiline": False},
                ),
            },
        }

    # ── execution ───────────────────────────────────────────────────────────────

    def run_sfm(
        self,
        images,
        feature_type,
        matcher_type,
        mapper,
        camera_model,
        output_dir,
        image_folder_override="",
    ):
        log_lines = []

        def log(msg):
            print(f"[SfM_COLMAP] {msg}")
            log_lines.append(msg)

        colmap = colmap_bin()
        log(f"COLMAP binary: {colmap}")

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # ── 1. Write images to disk ─────────────────────────────────────────────
        if image_folder_override and Path(image_folder_override).is_dir():
            image_dir = Path(image_folder_override)
            log(f"Using image folder override: {image_dir}")
        else:
            image_dir = output_path / "images"
            image_dir.mkdir(exist_ok=True)
            pil_images = tensor_to_pil(images)
            log(f"Writing {len(pil_images)} images to {image_dir}")
            for idx, img in enumerate(pil_images):
                img.save(image_dir / f"frame_{idx:05d}.jpg", quality=95)

        database_path = output_path / "database.db"
        sparse_dir    = output_path / "sparse"
        sparse_dir.mkdir(exist_ok=True)

        # ── 2. Feature extraction ───────────────────────────────────────────────
        log(f"Extracting features: {feature_type}")

        extract_cmd = [
            colmap, "feature_extractor",
            "--database_path", str(database_path),
            "--image_path",    str(image_dir),
            "--ImageReader.camera_model", camera_model,
            "--ImageReader.single_camera", "1",
        ]

        if feature_type == "ALIKED":
            extract_cmd += [
                "--FeatureExtraction.use_gpu", "1",
                "--FeatureExtraction.num_threads", "-1",
                # ALIKED is invoked via COLMAP's deep feature interface
                "--SiftExtraction.use_gpu", "0",          # disable SIFT GPU path
                "--ImageReader.camera_model", camera_model,
            ]
            # COLMAP 4.0 ALIKED flag
            extract_cmd += ["--FeatureExtraction.feature_type", "ALIKED"]
        else:
            extract_cmd += [
                "--SiftExtraction.use_gpu", "1",
            ]

        run(extract_cmd)
        log("Feature extraction complete.")

        # ── 3. Feature matching ─────────────────────────────────────────────────
        log(f"Matching features: {matcher_type}")

        if matcher_type == "lightglue":
            match_cmd = [
                colmap, "lightglue_matches",
                "--database_path", str(database_path),
            ]
        elif matcher_type == "exhaustive":
            match_cmd = [
                colmap, "exhaustive_matcher",
                "--database_path", str(database_path),
                "--SiftMatching.use_gpu", "1",
            ]
        elif matcher_type == "sequential":
            match_cmd = [
                colmap, "sequential_matcher",
                "--database_path", str(database_path),
                "--SiftMatching.use_gpu", "1",
                "--SequentialMatching.overlap", "10",
                "--SequentialMatching.loop_detection", "1",
            ]
        else:  # vocab_tree
            match_cmd = [
                colmap, "vocab_tree_matcher",
                "--database_path", str(database_path),
                "--SiftMatching.use_gpu", "1",
            ]

        run(match_cmd)
        log("Feature matching complete.")

        # ── 4. Sparse reconstruction (mapper) ───────────────────────────────────
        log(f"Running mapper: {mapper}")

        if mapper == "glomap":
            # GLOMAP is bundled inside COLMAP 4.0
            mapper_cmd = [
                colmap, "glomap_mapper",
                "--database_path", str(database_path),
                "--image_path",    str(image_dir),
                "--output_path",   str(sparse_dir),
            ]
        else:
            mapper_cmd = [
                colmap, "mapper",
                "--database_path", str(database_path),
                "--image_path",    str(image_dir),
                "--output_path",   str(sparse_dir),
                "--Mapper.num_threads",         "-1",
                "--Mapper.init_min_tri_angle",  "4",
                "--Mapper.multiple_models",     "0",  # force single model
            ]

        run(mapper_cmd)
        log("Sparse reconstruction complete.")

        # ── 5. Convert sparse model to TXT ──────────────────────────────────────
        # COLMAP writes to sparse/0/ by default
        model_dir = sparse_dir / "0"
        if not model_dir.exists():
            # Some versions write directly to sparse/
            model_dir = sparse_dir

        txt_dir = output_path / "sparse_txt"
        txt_dir.mkdir(exist_ok=True)

        run([
            colmap, "model_converter",
            "--input_path",  str(model_dir),
            "--output_path", str(txt_dir),
            "--output_type", "TXT",
        ])
        log(f"Model converted to TXT: {txt_dir}")

        # ── 6. Parse results ────────────────────────────────────────────────────
        cameras = parse_cameras_txt(txt_dir / "cameras.txt")
        poses   = parse_images_txt(txt_dir  / "images.txt")
        pc_data = parse_points3d_txt(txt_dir / "points3D.txt")

        log(f"Registered images : {len(poses)}")
        log(f"Sparse point count: {pc_data['num_points']}")
        log(f"Scene centroid    : {[round(v,3) for v in pc_data['centroid']]}")

        # ── 7. Build result dict ────────────────────────────────────────────────
        sfm_result = {
            # Paths — downstream nodes (Train_3DGS etc.) read these
            "colmap_dir":    str(output_path),
            "image_dir":     str(image_dir),
            "sparse_dir":    str(txt_dir),
            "database_path": str(database_path),

            # Parsed metadata
            "cameras":       cameras,
            "poses":         poses,
            "num_registered": len(poses),
            "num_points":    pc_data["num_points"],
            "centroid":      pc_data["centroid"],
            "points_xyz":    pc_data["points_xyz"],  # np.ndarray (N,3)

            # Config used — useful for provenance
            "config": {
                "feature_type": feature_type,
                "matcher_type": matcher_type,
                "mapper":       mapper,
                "camera_model": camera_model,
            },
        }

        # Also write a JSON summary (without the numpy array)
        summary = {k: v for k, v in sfm_result.items() if k != "points_xyz"}
        summary_path = output_path / "sfm_summary.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        log(f"Summary written: {summary_path}")

        status_log = "\n".join(log_lines)
        return (sfm_result, status_log)


# ── registration ────────────────────────────────────────────────────────────────

NODE_CLASS_MAPPINGS = {
    "SfM_COLMAP_ALIKED": SfM_COLMAP_ALIKED,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SfM_COLMAP_ALIKED": "SfM · COLMAP (ALIKED + LightGlue)",
}
