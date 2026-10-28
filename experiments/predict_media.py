from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Always load src/ from this CFD-FSA project, not another project on PYTHONPATH.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from facenet_pytorch import MTCNN

from src.models import CFD_FSA, FrozenCLIPEncoder
from src.protonet import prototypical_logits
from src.sampling import sample_support
from src.utils import FeatureStore, get_device


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv"}


def get_face_detector(device):
    return MTCNN(
        image_size=224,
        margin=20,
        keep_all=False,
        post_process=False,
        device=device,
    )


def crop_face(image: Image.Image, detector):
    boxes, probs = detector.detect(image)

    if boxes is None or len(boxes) == 0:
        return None

    best = int(probs.argmax()) if probs is not None else 0

    x1, y1, x2, y2 = boxes[best]

    w, h = image.size
    margin = 20

    x1 = max(0, int(x1) - margin)
    y1 = max(0, int(y1) - margin)
    x2 = min(w, int(x2) + margin)
    y2 = min(h, int(y2) + margin)

    return image.crop((x1, y1, x2, y2)).resize((224, 224))


def read_image(path: Path, detector):
    image = Image.open(path).convert("RGB")
    face = crop_face(image, detector)

    if face is None:
        raise RuntimeError(
            f"No face was detected in image: {path}\n"
            "The CFD-FSA model is trained on face crops, so a full-image fallback "
            "is intentionally disabled."
        )

    return face


def sample_video_frames(path: Path, num_frames: int):
    cap = cv2.VideoCapture(str(path))

    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames <= 0:
        cap.release()
        raise RuntimeError("Could not determine video frame count.")

    indices = np.linspace(
        0,
        total_frames - 1,
        min(num_frames, total_frames),
        dtype=int,
    )

    frames = []

    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()

        if not ok:
            continue

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(Image.fromarray(frame))

    cap.release()

    if not frames:
        raise RuntimeError("Could not extract any video frames.")

    return frames


def load_model(checkpoint_path, device):
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
    )

    ckpt_cfg = checkpoint.get("config", {})
    model_cfg = ckpt_cfg.get("model", {})

    forensic_dim = int(model_cfg.get("forensic_dim", 128))
    content_dim = int(model_cfg.get("content_dim", 512))
    grl_lambda = float(model_cfg.get("grl_lambda", 1.0))
    dropout = float(model_cfg.get("adapter_dropout", 0.1))
    use_frequency = bool(model_cfg.get("use_frequency_branch", False))
    temperature = float(model_cfg.get("temperature", 0.07))
    learnable_temperature = bool(model_cfg.get("learnable_temperature", False))
    use_mspm = bool(model_cfg.get("use_mspm", False))
    mspm_slots = int(model_cfg.get("mspm_slots", 16))

    clip_encoder = FrozenCLIPEncoder(
        model_name="openai/clip-vit-base-patch32",
        device=device,
    )

    clip_dim = clip_encoder.output_dim

    model = CFD_FSA(
        clip_dim=clip_dim,
        forensic_dim=forensic_dim,
        content_dim=content_dim,
        grl_lambda=grl_lambda,
        dropout=dropout,
        use_frequency=use_frequency,
        temperature=temperature,
        learnable_temperature=learnable_temperature,
        use_mspm=use_mspm,
        mspm_slots=mspm_slots,
    ).to(device)

    state = checkpoint.get("model", checkpoint)

    try:
        model.load_state_dict(state, strict=True)
    except RuntimeError as exc:
        raise RuntimeError(
            "\nCheckpoint architecture does not match the normal CFD-FSA model.\n"
            "Use a normal Stage-D checkpoint such as k1.pt/k5.pt/k10.pt/k20.pt.\n\n"
            f"Original error:\n{exc}"
        ) from exc

    model.eval()

    return model, clip_encoder, ckpt_cfg


def build_prototypes(model, feature_store, cfg, checkpoint):
    train_df = feature_store.metadata
    k = int(checkpoint.get("k", 20))

    adaptation = (
        cfg.get("experiment", {})
        .get("adaptation_manipulations", ["DF", "F2F"])
    )

    # Use the same five support seeds used for evaluation and combine them.
    support_list = []

    for support_seed in [0, 1, 2, 3, 4]:
        support = sample_support(
            train_df,
            k=k,
            seed=support_seed,
            fake_manipulations=adaptation,
            split="train",
        )
        support_list.append(support)

    support = pd.concat(support_list, ignore_index=True)

    support_clip, _, support_freq = feature_store.get(support)

    support_labels = torch.tensor(
        support["label"].to_numpy(),
        dtype=torch.long,
    )

    return support_clip, support_freq, support_labels


@torch.no_grad()
def classify_embeddings(
    model,
    support_clip,
    support_freq,
    support_labels,
    query_clip,
    device,
    temperature,
):
    support_clip = support_clip.to(device)
    query_clip = query_clip.to(device)

    if model.use_frequency and support_freq is not None:
        support_freq = support_freq.to(device)
    else:
        support_freq = None

    support_z = model.forensic(
        support_clip,
        support_freq,
    )

    query_z = model.forensic(
        query_clip,
        None,
    )

    enricher = model.enrich_prototypes if hasattr(model, "enrich_prototypes") else None

    logits, _ = prototypical_logits(
        support_z,
        support_labels.to(device),
        query_z,
        temperature,
        prototype_enricher=enricher,
    )

    probabilities = torch.softmax(logits, dim=1)
    return probabilities[:, 1].cpu().numpy()


def predict_image(
    input_path,
    detector,
    clip_encoder,
    model,
    support_clip,
    support_freq,
    support_labels,
    device,
    temperature,
    real_threshold,
):
    face = read_image(input_path, detector)

    print("\nRunning image inference...")

    pixel_values = clip_encoder.processor(
        images=face,
        return_tensors="pt",
    ).pixel_values.to(device)

    with torch.no_grad():
        clip_features = clip_encoder.encode_pixels(pixel_values)

    fake_probs = classify_embeddings(
        model=model,
        support_clip=support_clip,
        support_freq=support_freq,
        support_labels=support_labels,
        query_clip=clip_features,
        device=device,
        temperature=temperature,
    )

    fake_probability = float(fake_probs[0])
    real_probability = 1.0 - fake_probability
    prediction = "REAL" if real_probability >= args.real_threshold else "FAKE"

    print("\n==============================")
    print("CFD-FSA IMAGE RESULT")
    print("==============================")
    print(f"Prediction       : {prediction}")
    print(f"Fake probability : {fake_probability:.4f}")
    print(f"Real probability : {real_probability:.4f}")
    print(f"Real threshold   : {args.real_threshold:.2f}")
    print("==============================")


def predict_video(
    input_path,
    num_frames,
    detector,
    clip_encoder,
    model,
    support_clip,
    support_freq,
    support_labels,
    device,
    temperature,
    real_threshold,
):
    frames = sample_video_frames(input_path, num_frames)

    cropped_frames = []

    for frame in frames:
        face = crop_face(frame, detector)
        if face is not None:
            cropped_frames.append(face)

    if not cropped_frames:
        raise RuntimeError(
            "No face was detected in any sampled video frame."
        )

    print(
        f"\nRunning video inference on "
        f"{len(cropped_frames)} face frames..."
    )

    pixel_values = clip_encoder.processor(
        images=cropped_frames,
        return_tensors="pt",
    ).pixel_values.to(device)

    with torch.no_grad():
        clip_features = clip_encoder.encode_pixels(pixel_values)

    fake_probs = classify_embeddings(
        model=model,
        support_clip=support_clip,
        support_freq=support_freq,
        support_labels=support_labels,
        query_clip=clip_features,
        device=device,
        temperature=temperature,
    )

    mean_fake = float(fake_probs.mean())
    mean_real = 1.0 - mean_fake
    fake_threshold = 1.0 - real_threshold
    fake_ratio = float((fake_probs >= fake_threshold).mean())

    # Keep the temporal score as an additional diagnostic, but the final
    # decision is based on mean REAL probability crossing 65% by default.
    final_score = 0.7 * mean_fake + 0.3 * fake_ratio
    prediction = "REAL" if mean_real >= real_threshold else "FAKE"

    print("\nFrame probabilities:")
    for i, p in enumerate(fake_probs):
        print(f"Frame {i + 1}: {p:.4f}")

    print("\n==============================")
    print("CFD-FSA VIDEO RESULT")
    print("==============================")
    print(f"Prediction              : {prediction}")
    print(f"Mean fake probability   : {mean_fake:.4f}")
    print(f"Mean real probability   : {mean_real:.4f}")
    print(f"Fake frame ratio        : {fake_ratio * 100:.1f}%")
    print(f"Final detection score   : {final_score:.4f}")
    print(f"Frames analysed         : {len(fake_probs)}")
    print(f"Real threshold          : {real_threshold:.2f}")
    print("==============================")


def main():
    parser = argparse.ArgumentParser(
        description="Run CFD-FSA inference on an image OR video."
    )

    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Path to trained CFD-FSA checkpoint.",
    )

    parser.add_argument(
        "--input",
        required=True,
        help="Path to a JPG/PNG/BMP/WEBP image or MP4/AVI/MOV/MKV video.",
    )

    parser.add_argument(
        "--config",
        default="configs/stage_d.yaml",
        help="Fallback config path; checkpoint config is used when available.",
    )

    parser.add_argument(
        "--frames",
        type=int,
        default=32,
        help="Number of video frames to sample.",
    )

    parser.add_argument(
        "--real-threshold",
        type=float,
        default=0.65,
        help="Minimum real probability required to classify the input as REAL.",
    )

    args = parser.parse_args()

    input_path = Path(args.input)

    if not input_path.exists():
        raise FileNotFoundError(
            f"Input file does not exist: {input_path}"
        )

    if input_path.suffix.lower() not in IMAGE_EXTS | VIDEO_EXTS:
        raise ValueError(
            f"Unsupported file type: {input_path.suffix}\n"
            f"Images: {sorted(IMAGE_EXTS)}\n"
            f"Videos: {sorted(VIDEO_EXTS)}"
        )

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        if checkpoint_path.with_suffix(".pt").exists():
            checkpoint_path = checkpoint_path.with_suffix(".pt")
        elif checkpoint_path.with_suffix(".pth").exists():
            checkpoint_path = checkpoint_path.with_suffix(".pth")
        else:
            raise FileNotFoundError(
                f"Checkpoint file does not exist: {checkpoint_path}"
            )
    args.checkpoint = str(checkpoint_path)

    device = get_device("auto")

    print(f"Device: {device}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Input: {input_path}")

    checkpoint = torch.load(
        args.checkpoint,
        map_location=device,
    )

    cfg = checkpoint.get("config", {})

    model, clip_encoder, _ = load_model(
        args.checkpoint,
        device,
    )

    feature_root = Path(
        cfg.get("paths", {})
        .get("feature_root", "data/features")
    )

    train_store = FeatureStore(
        feature_root / "train.pt"
    )

    support_clip, support_freq, support_labels = build_prototypes(
        model,
        train_store,
        cfg,
        checkpoint,
    )

    temperature = float(
        cfg.get("model", {})
        .get("temperature", 0.07)
    )

    detector = get_face_detector(device)

    if input_path.suffix.lower() in IMAGE_EXTS:
        predict_image(
            input_path=input_path,
            detector=detector,
            clip_encoder=clip_encoder,
            model=model,
            support_clip=support_clip,
            support_freq=support_freq,
            support_labels=support_labels,
            device=device,
            temperature=temperature,
            real_threshold=args.real_threshold,
        )

    else:
        predict_video(
            input_path=input_path,
            num_frames=args.frames,
            detector=detector,
            clip_encoder=clip_encoder,
            model=model,
            support_clip=support_clip,
            support_freq=support_freq,
            support_labels=support_labels,
            device=device,
            temperature=temperature,
            real_threshold=args.real_threshold,
        )


if __name__ == "__main__":
    main()
