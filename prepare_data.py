"""
prepare_data.py — Download and prepare a subset of UniTalk-ASD for training.

Downloads raw files from HuggingFace Hub (CSVs, images, audio), then samples
~2,000 entity tracks (~5% of training data) to keep training within 4 hours.

Usage:
    python prepare_data.py [--num_samples 2000] [--output_dir ./data] [--seed 42]
"""

import argparse
import json
import os
import random
import shutil
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download, list_repo_tree
from PIL import Image
from tqdm import tqdm

DATASET_REPO = "plnguyen2908/UniTalk-ASD"

# The CSV files have no headers. These are the actual column names
# based on the dataset documentation.
CSV_COLUMNS = [
    "video_id",
    "frame_timestamp",
    "entity_box_x1",
    "entity_box_y1",
    "entity_box_x2",
    "entity_box_y2",
    "label",
    "entity_id",
    "label_id",
    "instance_id",
]


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


def download_csv_files(split, cache_dir=None):
    """
    Download all CSV files for a split and combine into one DataFrame.
    Each CSV is one video, with no header row.
    """
    print(f"Listing CSV files for {split} split...")

    # List files in the csv/{split}/ directory
    csv_prefix = f"csv/{split}/"
    all_files = list_repo_tree(DATASET_REPO, path_in_repo=csv_prefix, repo_type="dataset")
    csv_files = [f.path for f in all_files if f.path.endswith(".csv")]

    print(f"Found {len(csv_files)} CSV files for {split}")

    all_dfs = []
    for csv_path in tqdm(csv_files, desc=f"Downloading {split} CSVs"):
        local_path = hf_hub_download(
            DATASET_REPO,
            filename=csv_path,
            repo_type="dataset",
            cache_dir=cache_dir,
        )
        df = pd.read_csv(local_path, header=None, names=CSV_COLUMNS)
        all_dfs.append(df)

    combined = pd.concat(all_dfs, ignore_index=True)
    print(f"Total rows in {split}: {len(combined):,}")
    return combined


def sample_entity_tracks(df, num_samples, seed):
    """
    Sample entity tracks (unique entity_ids) from the DataFrame.
    Tries to balance SPEAKING vs NOT_SPEAKING tracks.
    """
    random.seed(seed)

    # Group by entity_id and determine majority label per track
    track_labels = df.groupby("entity_id")["label_id"].apply(
        lambda x: "SPEAKING" if x.mean() > 0.5 else "NOT_SPEAKING"
    )

    speaking_ids = track_labels[track_labels == "SPEAKING"].index.tolist()
    not_speaking_ids = track_labels[track_labels == "NOT_SPEAKING"].index.tolist()

    total_tracks = len(track_labels)
    print(f"Total entity tracks: {total_tracks}")
    print(f"  SPEAKING tracks: {len(speaking_ids)}")
    print(f"  NOT_SPEAKING tracks: {len(not_speaking_ids)}")

    if num_samples >= total_tracks:
        print(f"Requested {num_samples} but only {total_tracks} available. Using all.")
        return track_labels.index.tolist()

    # Balanced sampling: half speaking, half not speaking
    half = num_samples // 2
    n_speaking = min(half, len(speaking_ids))
    n_not_speaking = min(num_samples - n_speaking, len(not_speaking_ids))
    # If one class is short, take more from the other
    if n_not_speaking < num_samples - n_speaking:
        n_speaking = min(num_samples - n_not_speaking, len(speaking_ids))

    sampled_speaking = random.sample(speaking_ids, n_speaking)
    sampled_not_speaking = random.sample(not_speaking_ids, n_not_speaking)
    sampled_ids = sampled_speaking + sampled_not_speaking

    random.shuffle(sampled_ids)
    print(f"Sampled {len(sampled_ids)} tracks ({n_speaking} speaking, {n_not_speaking} not speaking)")
    return sampled_ids


def download_media_for_track(entity_id, split, cache_dir=None):
    """
    Download the audio WAV and face crop images for one entity track.

    File structure on HF:
    - clips_audios/{split}/{video_id}/{entity_id}.wav
    - clips_videos/{split}/{video_id}/{entity_id}/*.jpg
    """
    # entity_id format is "video_id:track_number" e.g. "-ASZexdSdWE:0"
    parts = entity_id.rsplit(":", 1)
    video_id = parts[0]

    # Download audio
    audio_hf_path = f"clips_audios/{split}/{video_id}/{entity_id}.wav"
    try:
        audio_local = hf_hub_download(
            DATASET_REPO,
            filename=audio_hf_path,
            repo_type="dataset",
            cache_dir=cache_dir,
        )
    except Exception as e:
        print(f"  Warning: Could not download audio for {entity_id}: {e}")
        return None, []

    # List and download images for this entity
    img_prefix = f"clips_videos/{split}/{video_id}/{entity_id}/"
    try:
        img_files = list_repo_tree(
            DATASET_REPO, path_in_repo=img_prefix, repo_type="dataset"
        )
        img_paths_hf = sorted([f.path for f in img_files if f.path.endswith(".jpg")])
    except Exception:
        img_paths_hf = []

    img_locals = []
    for img_hf_path in img_paths_hf:
        try:
            local = hf_hub_download(
                DATASET_REPO,
                filename=img_hf_path,
                repo_type="dataset",
                cache_dir=cache_dir,
            )
            img_locals.append(local)
        except Exception:
            continue

    return audio_local, img_locals


def process_and_save_split(df, sampled_ids, split, max_frames, output_dir, cache_dir=None):
    """
    For each sampled entity track:
    1. Download audio + face crops from HuggingFace
    2. Subsample frames
    3. Copy to output directory
    4. Write metadata
    """
    split_dir = Path(output_dir) / split
    img_dir = split_dir / "images"
    audio_dir = split_dir / "audio"
    img_dir.mkdir(parents=True, exist_ok=True)
    audio_dir.mkdir(parents=True, exist_ok=True)

    metadata = []
    label_counts = defaultdict(int)
    skipped = 0

    print(f"\nDownloading and processing {len(sampled_ids)} tracks for {split}...")
    for entity_id in tqdm(sampled_ids, desc=f"Processing {split}"):
        # Get track data from CSV
        track_df = df[df["entity_id"] == entity_id].sort_values("frame_timestamp")

        if track_df.empty:
            skipped += 1
            continue

        # Download media files
        audio_src, img_srcs = download_media_for_track(entity_id, split, cache_dir)

        if audio_src is None or len(img_srcs) == 0:
            skipped += 1
            continue

        # Subsample frames evenly
        num_frames = len(img_srcs)
        if num_frames > max_frames:
            step = num_frames / max_frames
            selected_indices = [int(i * step) for i in range(max_frames)]
        else:
            selected_indices = list(range(num_frames))

        selected_imgs = [img_srcs[i] for i in selected_indices]

        # Get labels for selected frames
        # Match by position (frames are sorted by timestamp)
        track_labels = track_df["label_id"].tolist()
        track_timestamps = track_df["frame_timestamp"].tolist()

        # Subsample labels to match selected frames
        if len(track_labels) > max_frames:
            step = len(track_labels) / max_frames
            label_indices = [int(i * step) for i in range(max_frames)]
        else:
            label_indices = list(range(len(track_labels)))

        selected_labels = [int(track_labels[i]) for i in label_indices]
        selected_timestamps = [float(track_timestamps[i]) for i in label_indices]

        # Determine majority label
        speaking_count = sum(selected_labels)
        total_selected = len(selected_labels)
        majority_label = "SPEAKING" if speaking_count > total_selected / 2 else "NOT_SPEAKING"

        # Copy files to output directory
        safe_entity_id = entity_id.replace(":", "_")

        # Copy audio
        audio_dst = audio_dir / f"{safe_entity_id}.wav"
        shutil.copy2(audio_src, audio_dst)

        # Copy selected images
        image_paths = []
        for frame_idx, img_src in enumerate(selected_imgs):
            img_dst = img_dir / f"{safe_entity_id}_frame{frame_idx:03d}.jpg"
            shutil.copy2(img_src, img_dst)
            image_paths.append(str(img_dst))

        # Build metadata
        frame_labels_str = ", ".join(
            [f"frame {i+1}: {'SPEAKING' if l == 1 else 'NOT_SPEAKING'}"
             for i, l in enumerate(selected_labels)]
        )

        entry = {
            "entity_id": entity_id,
            "image_paths": image_paths,
            "audio_path": str(audio_dst),
            "timestamps": selected_timestamps,
            "labels": selected_labels,
            "majority_label": majority_label,
            "frame_labels_str": frame_labels_str,
            "num_frames": len(selected_imgs),
            "speaking_ratio": speaking_count / total_selected if total_selected > 0 else 0,
        }
        metadata.append(entry)
        label_counts[majority_label] += 1

    # Save metadata
    metadata_path = split_dir / "metadata.jsonl"
    with open(metadata_path, "w") as f:
        for entry in metadata:
            f.write(json.dumps(entry) + "\n")

    print(f"\n{split} statistics:")
    print(f"  Total processed: {len(metadata)}")
    print(f"  Skipped (missing media): {skipped}")
    for label, count in sorted(label_counts.items()):
        pct = count / len(metadata) * 100 if metadata else 0
        print(f"  {label}: {count} ({pct:.1f}%)")
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

    # --- Training data ---
    print("\n--- Training Data ---")
    train_df = download_csv_files("train", args.cache_dir)
    train_sampled_ids = sample_entity_tracks(train_df, args.num_samples, args.seed)
    process_and_save_split(
        train_df, train_sampled_ids, "train",
        args.max_frames, args.output_dir, args.cache_dir,
    )
    del train_df

    # --- Validation data ---
    print("\n--- Validation Data ---")
    val_df = download_csv_files("val", args.cache_dir)
    val_sampled_ids = sample_entity_tracks(val_df, args.val_samples, args.seed + 1)
    process_and_save_split(
        val_df, val_sampled_ids, "val",
        args.max_frames, args.output_dir, args.cache_dir,
    )

    print("\n" + "=" * 60)
    print("Data preparation complete!")
    print(f"Training data: {args.output_dir}/train/")
    print(f"Validation data: {args.output_dir}/val/")
    print("=" * 60)


if __name__ == "__main__":
    main()
