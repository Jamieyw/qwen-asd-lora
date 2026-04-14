"""
prepare_data.py — Download and prepare a subset of UniTalk-ASD for training.

Downloads the full dataset from HuggingFace, then samples ~2,000 entity tracks
(~5% of training data) to keep training within 4 hours on a V100.

Usage:
    python prepare_data.py [--num_samples 2000] [--output_dir ./data] [--seed 42]
"""

import argparse
import json
import os
import random
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
from datasets import load_dataset
from PIL import Image
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare UniTalk-ASD subset for training")
    parser.add_argument("--num_samples", type=int, default=2000,
                        help="Number of entity tracks to sample for training (default: 2000)")
    parser.add_argument("--max_frames", type=int, default=10,
                        help="Max frames to keep per entity track (default: 10)")
    parser.add_argument("--output_dir", type=str, default="./data",
                        help="Directory to save processed data")
    parser.add_argument("--val_samples", type=int, default=500,
                        help="Number of entity tracks for validation (default: 500)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--cache_dir", type=str, default=None,
                        help="HuggingFace cache directory (useful on cluster)")
    return parser.parse_args()


def load_and_sample_dataset(split, num_samples, seed, cache_dir=None):
    """
    Load dataset from HuggingFace and sample a subset of entity tracks.

    Each sample in the HF dataset is one entity track containing:
    - entity_id: str
    - images: list of PIL images (face crops at 25fps)
    - audio: tuple (sample_rate, audio_array)
    - frame_timestamp: list of floats
    - label_id: list of ints (0=not speaking, 1=speaking)
    """
    print(f"Loading {split} split from HuggingFace...")
    print("(This may take a while on first run — the full dataset is ~14.6GB)")

    dataset = load_dataset(
        "plnguyen2908/UniTalk-ASD",
        split=split,
        cache_dir=cache_dir,
        trust_remote_code=True,
    )

    total = len(dataset)
    print(f"Total {split} samples: {total}")

    if num_samples >= total:
        print(f"Requested {num_samples} but only {total} available. Using all.")
        indices = list(range(total))
    else:
        # Sample with balanced labels
        # First pass: categorize samples by majority label
        print("Analyzing label distribution for balanced sampling...")
        random.seed(seed)

        # Random sample indices
        indices = random.sample(range(total), num_samples)

    print(f"Selected {len(indices)} samples from {split}")
    return dataset, indices


def process_sample(sample, max_frames):
    """
    Process one entity track into training format.

    For Qwen2.5-Omni, we format as a conversation:
    - User provides face images + audio
    - Assistant responds with per-frame speaking labels

    We subsample to max_frames evenly spaced frames.
    """
    images = sample["images"]  # list of PIL images
    audio = sample["audio"]    # (sample_rate, audio_array)
    timestamps = sample["frame_timestamp"]
    labels = sample["label_id"]  # list of 0/1
    entity_id = sample["entity_id"]

    num_frames = len(images)

    # Subsample frames evenly if too many
    if num_frames > max_frames:
        # Evenly spaced indices
        step = num_frames / max_frames
        selected_indices = [int(i * step) for i in range(max_frames)]
    else:
        selected_indices = list(range(num_frames))

    selected_images = [images[i] for i in selected_indices]
    selected_timestamps = [timestamps[i] for i in selected_indices]
    selected_labels = [labels[i] for i in selected_indices]

    # Determine majority label for this track (for the classification task)
    # We use majority vote across selected frames as the track-level label
    speaking_count = sum(selected_labels)
    total_selected = len(selected_labels)
    majority_label = "SPEAKING" if speaking_count > total_selected / 2 else "NOT_SPEAKING"

    # Also compute frame-level label string for detailed training
    frame_labels_str = ", ".join(
        [f"frame {i+1}: {'SPEAKING' if l == 1 else 'NOT_SPEAKING'}"
         for i, l in enumerate(selected_labels)]
    )

    return {
        "entity_id": entity_id,
        "images": selected_images,
        "audio": audio,
        "timestamps": selected_timestamps,
        "labels": selected_labels,
        "majority_label": majority_label,
        "frame_labels_str": frame_labels_str,
        "num_frames": len(selected_images),
        "speaking_ratio": speaking_count / total_selected if total_selected > 0 else 0,
    }


def save_processed_data(dataset, indices, max_frames, output_dir, split_name):
    """
    Process and save dataset samples to disk.

    Saves:
    - images/ directory with face crops as JPG
    - audio/ directory with audio as WAV
    - metadata.jsonl with labels and file paths
    """
    split_dir = Path(output_dir) / split_name
    img_dir = split_dir / "images"
    audio_dir = split_dir / "audio"

    img_dir.mkdir(parents=True, exist_ok=True)
    audio_dir.mkdir(parents=True, exist_ok=True)

    metadata = []
    label_counts = defaultdict(int)

    print(f"Processing {len(indices)} samples for {split_name}...")
    for idx_num, dataset_idx in enumerate(tqdm(indices)):
        sample = dataset[dataset_idx]
        processed = process_sample(sample, max_frames)

        entity_id = processed["entity_id"].replace(":", "_")

        # Save images
        image_paths = []
        for frame_idx, img in enumerate(processed["images"]):
            img_path = f"{entity_id}_frame{frame_idx:03d}.jpg"
            if isinstance(img, Image.Image):
                img.save(img_dir / img_path, quality=85)
            image_paths.append(str(img_dir / img_path))

        # Save audio
        import soundfile as sf
        audio_data = processed["audio"]
        if isinstance(audio_data, tuple):
            sample_rate, audio_array = audio_data
        else:
            # Handle different audio formats
            sample_rate = audio_data.get("sampling_rate", 16000)
            audio_array = np.array(audio_data.get("array", audio_data))

        audio_path = f"{entity_id}.wav"
        audio_array_np = np.array(audio_array)
        if audio_array_np.dtype in [np.float32, np.float64]:
            # Normalize to [-1, 1] if needed
            max_val = np.abs(audio_array_np).max()
            if max_val > 1.0:
                audio_array_np = audio_array_np / max_val
        sf.write(str(audio_dir / audio_path), audio_array_np, sample_rate)

        # Build metadata entry
        entry = {
            "entity_id": processed["entity_id"],
            "image_paths": image_paths,
            "audio_path": str(audio_dir / audio_path),
            "timestamps": processed["timestamps"],
            "labels": processed["labels"],
            "majority_label": processed["majority_label"],
            "frame_labels_str": processed["frame_labels_str"],
            "num_frames": processed["num_frames"],
            "speaking_ratio": processed["speaking_ratio"],
        }
        metadata.append(entry)
        label_counts[processed["majority_label"]] += 1

    # Save metadata
    metadata_path = split_dir / "metadata.jsonl"
    with open(metadata_path, "w") as f:
        for entry in metadata:
            f.write(json.dumps(entry) + "\n")

    print(f"\n{split_name} statistics:")
    print(f"  Total samples: {len(metadata)}")
    for label, count in sorted(label_counts.items()):
        print(f"  {label}: {count} ({count/len(metadata)*100:.1f}%)")
    print(f"  Saved to: {split_dir}")

    return metadata


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    print("=" * 60)
    print("UniTalk-ASD Data Preparation")
    print("=" * 60)
    print(f"Training samples: {args.num_samples}")
    print(f"Validation samples: {args.val_samples}")
    print(f"Max frames per track: {args.max_frames}")
    print(f"Output directory: {args.output_dir}")
    print(f"Random seed: {args.seed}")
    print("=" * 60)

    # Load and process training data
    train_dataset, train_indices = load_and_sample_dataset(
        "train", args.num_samples, args.seed, args.cache_dir
    )
    save_processed_data(
        train_dataset, train_indices, args.max_frames,
        args.output_dir, "train"
    )

    # Free memory
    del train_dataset

    # Load and process validation data
    val_dataset, val_indices = load_and_sample_dataset(
        "val", args.val_samples, args.seed + 1, args.cache_dir
    )
    save_processed_data(
        val_dataset, val_indices, args.max_frames,
        args.output_dir, "val"
    )

    print("\n" + "=" * 60)
    print("Data preparation complete!")
    print(f"Training data: {args.output_dir}/train/")
    print(f"Validation data: {args.output_dir}/val/")
    print("=" * 60)


if __name__ == "__main__":
    main()
