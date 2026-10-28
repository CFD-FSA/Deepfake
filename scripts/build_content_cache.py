from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
from pathlib import Path

import pandas as pd
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from torchvision.transforms.functional import to_tensor

from src.models import FrozenCLIPEncoder
from src.utils import load_yaml, get_device
from src.frequency import block_dct_descriptor


def crop_with_mtcnn(image: Image.Image, detector, device: torch.device, image_size: int = 224, margin: int = 20):
    boxes, probs = detector.detect(image)
    if boxes is None or len(boxes) == 0:
        return None
    best = int(probs.argmax()) if probs is not None else 0
    x1, y1, x2, y2 = boxes[best]
    w, h = image.size
    x1 = max(0, int(x1) - margin)
    y1 = max(0, int(y1) - margin)
    x2 = min(w, int(x2) + margin)
    y2 = min(h, int(y2) + margin)
    crop = image.crop((x1, y1, x2, y2)).resize((image_size, image_size))
    return crop


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--external-content-target", default="", help="Optional .pt/.pth tensor or dict aligned to generated face rows")
    args = parser.parse_args()
    cfg = load_yaml(args.config)
    device = get_device(cfg.get("device", "auto"))

    # Heavy dependencies are loaded here so manifest generation does not need
    # the vision checkpoints.
    from facenet_pytorch import MTCNN, InceptionResnetV1
    from transformers import CLIPImageProcessor

    manifest = pd.read_csv(cfg["paths"]["manifest"])
    manifest = manifest[manifest["split"].isin(["train", "val", "test"])].copy()
    face_root = Path(cfg["paths"].get("face_root", "data/faces"))
    feature_root = Path(cfg["paths"].get("feature_root", "data/features"))
    face_root.mkdir(parents=True, exist_ok=True)
    feature_root.mkdir(parents=True, exist_ok=True)

    mtcnn = MTCNN(
        image_size=cfg["data"].get("face_size", 224),
        margin=cfg["data"].get("mtcnn_margin", 20),
        keep_all=False,
        post_process=False,
        device=device,
    )
    recognizer = InceptionResnetV1(pretrained=cfg["model"].get("content_model", "vggface2")).eval().to(device)
    clip = FrozenCLIPEncoder(cfg["model"].get("clip_name", "openai/clip-vit-base-patch32"), device)
    processor: CLIPImageProcessor = clip.processor

    frame_rows = []
    grouped = manifest.groupby("split")
    for split, split_df in grouped:
        for _, row in tqdm(split_df.iterrows(), total=len(split_df), desc=f"Frames + faces [{split}]"):
            video_path = Path(row["path"])
            video_out = face_root / split / row["manipulation"] / row["source_video_key"].replace("/", "__")
            video_out.mkdir(parents=True, exist_ok=True)
            # Reuse cfd_fsa.data extraction utility for deterministic frame sampling.
            from src.data import extract_video_frames
            frame_paths = extract_video_frames(
                video_path,
                video_out,
                every_n_frames=int(cfg["data"].get("frame_every_n", 10)),
                max_frames=cfg["data"].get("max_frames_per_video"),
            )
            for frame_path in frame_paths:
                image = Image.open(frame_path).convert("RGB")
                crop = crop_with_mtcnn(image, mtcnn, device, cfg["data"].get("face_size", 224), cfg["data"].get("mtcnn_margin", 20))
                if crop is None:
                    continue
                face_path = frame_path.with_name(frame_path.stem + "_face.jpg")
                crop.save(face_path, quality=95)
                frame_rows.append({
                    "face_path": str(face_path),
                    "frame_path": str(frame_path),
                    "source_video_key": row["source_video_key"],
                    "source_group_key": row.get("source_group_key", row["source_video_key"]),
                    "video_id": row.get("video_id", ""),
                    "identity": row.get("identity", ""),
                    "label": int(row["label"]),
                    "manipulation": row["manipulation"],
                    "compression": row["compression"],
                    "split": split,
                })

    frame_df = pd.DataFrame(frame_rows)
    external_targets = None
    if args.external_content_target:
        external_targets = torch.load(args.external_content_target, map_location="cpu")
        if isinstance(external_targets, dict):
            external_targets = torch.stack([external_targets[str(p)] for p in frame_df["face_path"]])
        elif not torch.is_tensor(external_targets):
            raise ValueError("External content target must be a tensor or dict keyed by face_path.")
        if len(external_targets) != len(frame_df):
            raise ValueError("External content target length must match the generated frame manifest.")
    frame_manifest = Path(cfg["paths"]["manifest"]).with_name("frame_manifest.csv")
    frame_df.to_csv(frame_manifest, index=False)
    print(f"Face manifest: {frame_manifest} ({len(frame_df)} faces)")

    # Cache per split. Content embeddings come from a frozen face-recognition network;
    # CLIP embeddings come from frozen CLIP.
    for split in ["train", "val", "test"]:
        sub = frame_df[frame_df["split"].eq(split)].reset_index(drop=True)
        if len(sub) == 0:
            continue
        clip_feats, content_feats, frequency_feats = [], [], []
        batch_paths = []
        batch_size = 32
        for _, row in tqdm(sub.iterrows(), total=len(sub), desc=f"Feature cache [{split}]"):
            batch_paths.append(row["face_path"])
            # Flush full batches; after the loop, flush the final partial batch too.
            if len(batch_paths) == batch_size:
                images = [Image.open(p).convert("RGB") for p in batch_paths]
                inputs = processor(images=images, return_tensors="pt").pixel_values.to(device)
                with torch.no_grad():
                    clip_batch = clip(inputs).cpu()
                    # InceptionResnet expects roughly 160x160 tensors.
                    rec = torch.stack([2 * (to_tensor(im.resize((160, 160))) - 0.5) for im in images]).to(device)
                    cont_batch = torch.nn.functional.normalize(recognizer(rec), dim=-1).cpu()
                freq_batch = torch.tensor(np.stack([block_dct_descriptor(im) for im in images]), dtype=torch.float32)
                clip_feats.append(clip_batch)
                content_feats.append(cont_batch)
                frequency_feats.append(freq_batch)
                batch_paths = []

        # Process the final partial batch. The previous implementation could
        # leave up to batch_size-1 samples out of the feature tensors while
        # still keeping them in the metadata, causing indexing errors later.
        if batch_paths:
            images = [Image.open(p).convert("RGB") for p in batch_paths]
            inputs = processor(images=images, return_tensors="pt").pixel_values.to(device)
            with torch.no_grad():
                clip_batch = clip(inputs).cpu()
                rec = torch.stack([2 * (to_tensor(im.resize((160, 160))) - 0.5) for im in images]).to(device)
                cont_batch = torch.nn.functional.normalize(recognizer(rec), dim=-1).cpu()
            freq_batch = torch.tensor(np.stack([block_dct_descriptor(im) for im in images]), dtype=torch.float32)
            clip_feats.append(clip_batch)
            content_feats.append(cont_batch)
            frequency_feats.append(freq_batch)
            batch_paths = []

        content_tensor = torch.cat(content_feats).float()
        if external_targets is not None:
            # Align by the split's row positions in frame_df.
            global_positions = frame_df.index[frame_df["split"].eq(split)].tolist()
            content_tensor = external_targets[global_positions].float()
        payload = {
            "clip_features": torch.cat(clip_feats).float(),
            "content_features": content_tensor,
            "frequency_features": torch.cat(frequency_feats).float(),
            "metadata": sub.drop(columns=[]).to_dict(orient="records"),
        }
        out = feature_root / f"{split}.pt"
        torch.save(payload, out)
        print(f"Saved: {out}")


if __name__ == "__main__":
    main()
