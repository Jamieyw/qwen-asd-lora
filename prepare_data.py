"""
prepare_data.py — Download and prepare a subset of UniTalk-ASD for training.

The HuggingFace repo stores media as ZIP archives per video_id:
  - clips_audios/{split}/{video_id}.zip  → contains {entity_id}.wav files
  - clips_videos/{split}/{video_id}.zip  → contains {entity_id}/*.jpg files
  - csv/{split}/{video_id}.csv           → headerless CSVs with annotations

This script:
  1. Downloads all CSVs to identify entity tracks
  2. Samples ~2,000 tracks (~5% of training data)
  3. Downloads only the ZIP files for selected videos
  4. Extracts the needed entity files
  5. Saves processed data for training

Usage:
    python prepare_data.py [--num_samples 2000] [--output_dir ./data] [--seed 42]
"""

import argparse
import json
import os
import random
import shutil
import tempfile
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download, list_repo_tree
from PIL import Image
from tqdm import tqdm

DATASET_REPO = "plnguyen2908/UniTalk-ASD"

# CSVs have no headers — these are the actual column names
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


# -----------------------------------------------------------------------
# Step 1: Download and parse CSV metadata
# -----------------------------------------------------------------------

def download_csv_files(split, cache_dir=None):
    """Download all CSV files for a split and combine into one DataFrame."""
    print(f"\nListing CSV files for {split} split...")

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
        try:
            df = pd.read_csv(local_path, header=None, names=CSV_COLUMNS)
            all_dfs.append(df)
        except Exception as e:
            print(f"  Warning: Could not parse {csv_path}: {e}")
            continue

    combined = pd.concat(all_dfs, ignore_index=True)
    print(f"Total rows in {split}: {len(combined):,}")
    print(f"Unique entity tracks: {combined['entity_id'].nunique():,}")
    print(f"Unique videos: {combined['video_id'].nunique():,}")
    return combined


# -----------------------------------------------------------------------
# Step 2: Sample entity tracks
# -----------------------------------------------------------------------

def sample_entity_tracks(df, num_samples, seed):
    """Sample entity tracks with balanced labels."""
    random.seed(seed)

    # Majority label per track
    track_labels = df.groupby("entity_id")["label_id"].apply(
        lambda x: "SPEAKING" if x.mean() > 0.5 else "NOT_SPEAKING"
    )

    speaking_ids = track_labels[track_labels == "SPEAKING"].index.tolist()
    not_speaking_ids = track_labels[track_labels == "NOT_SPEAKING"].index.tolist()

    total_tracks = len(track_labels)
    print(f"\nTotal entity tracks: {total_tracks}")
    print(f"  SPEAKING: {len(speaking_ids)}")
    print(f"  NOT_SPEAKING: {len(not_speaking_ids)}")

    if num_samples >= total_tracks:
        print(f"Requested {num_samples} but only {total_tracks} available. Using all.")
        return track_labels.index.tolist()

    # Balanced sampling
    half = num_samples // 2
    n_speaking = min(half, len(speaking_ids))
    n_not_speaking = min(num_samples - n_speaking, len(not_speaking_ids))
    if n_not_speaking < num_samples - n_speaking:
        n_speaking = min(num_samples - n_not_speaking, len(speaking_ids))

    sampled = random.sample(speaking_ids, n_speaking) + random.sample(not_speaking_ids, n_not_speaking)
    random.shuffle(sampled)

    print(f"Sampled {len(sampled)} tracks ({n_speaking} speaking, {n_not_speaking} not speaking)")
    return sampled


# -----------------------------------------------------------------------
# Step 3: Download ZIPs and extract needed files
# -----------------------------------------------------------------------

def download_and_extract_video_zip(video_id, split, media_type, extract_dir, cache_dir=None):
    """
    Download a ZIP archive for one video and extract it.

    media_type: "clips_audios" or "clips_videos"
    Returns the path to the extracted directory, or None on failure.
    """
    zip_hf_path = f"{media_type}/{split}/{video_id}.zip"
    video_extract_dir = Path(extract_dir) / media_type / split / video_id

    # Skip if already extracted
    if video_extract_dir.exists() and any(video_extract_dir.iterdir()):
        return video_extract_dir

    try:
        zip_local = hf_hub_download(
            DATASET_REPO,
            filename=zip_hf_path,
            repo_type="dataset",
            cache_dir=cache_dir,
        )
    except Exception as e:
        print(f"  Warning: Could not download {zip_hf_path}: {e}")
        return None

    # Extract
    video_extract_dir.mkdir(parents=True, exist_ok=True)
    try:
        with zipfile.ZipFile(zip_local, "r") as zf:
            zf.extractall(video_extract_dir)
    except Exception as e:
        print(f"  Warning: Could not extract {zip_hf_path}: {e}")
        return None

    return video_extract_dir


def find_entity_audio(entity_id, audio_extract_dir):
    """Find the audio WAV file for an entity after extraction."""
    if audio_extract_dir is None:
        return None

    # The WAV might be at: {extract_dir}/{entity_id}.wav
    # Or nested: {extract_dir}/{video_id}/{entity_id}.wav
    # Try multiple patterns
    candidates = [
        audio_extract_dir / f"{entity_id}.wav",
        # Sometimes the zip extracts with a subdirectory
    ]

    # Also search recursively for any .wav matching entity_id
    for wav_file in audio_extract_dir.rglob("*.wav"):
        if entity_id in wav_file.stem or entity_id.replace(":", "_") in wav_file.stem:
            return wav_file

    for c in candidates:
        if c.exists():
            return c

    return None


def find_entity_images(entity_id, video_extract_dir):
    """Find face crop images for an entity after extraction."""
    if video_extract_dir is None:
        return []

    # Images might be at: {extract_dir}/{entity_id}/*.jpg
    # Or: {extract_dir}/{video_id}/{entity_id}/*.jpg
    # Search recursively
    images = []

    # Try direct path
    entity_dir = video_extract_dir / entity_id
    if entity_dir.exists():
        images = sorted(entity_dir.glob("*.jpg"))

    # Try with underscore instead of colon
    if not images:
        safe_id = entity_id.replace(":", "_")
        entity_dir = video_extract_dir / safe_id
        if entity_dir.exists():
            images = sorted(entity_dir.glob("*.jpg"))

    # Recursive search as fallback
    if not images:
        for d in video_extract_dir.rglob(entity_id):
            if d.is_dir():
                images = sorted(d.glob("*.jpg"))
                if images:
                    break

    return images


# -----------------------------------------------------------------------
# Step 4: Process and save
# -----------------------------------------------------------------------

def process_and_save_split(df, sampled_ids, split, max_frames, output_dir, cache_dir=None):
    """Download ZIPs for needed videos, extract, process tracks, save."""
    split_dir = Path(output_dir) / split
    img_dir = split_dir / "images"
    audio_dir = split_dir / "audio"
    img_dir.mkdir(parents=True, exist_ok=True)
    audio_dir.mkdir(parents=True, exist_ok=True)

    # Temporary directory for extracted ZIPs
    extract_dir = Path(output_dir) / "_extracted"
    extract_dir.mkdir(parents=True, exist_ok=True)

    # Figure out which video_ids we need
    sampled_df = df[df["entity_id"].isin(sampled_ids)]
    needed_videos = sampled_df["video_id"].unique().tolist()
    print(f"\nNeed to download ZIPs for {len(needed_videos)} videos")

    # Download and extract ZIPs for needed videos
    audio_dirs = {}
    video_dirs = {}

    print("Downloading audio ZIPs...")
    for vid in tqdm(needed_videos, desc="Audio ZIPs"):
        audio_dirs[vid] = download_and_extract_video_zip(
            vid, split, "clips_audios", extract_dir, cache_dir
        )

    print("Downloading video ZIPs...")
    for vid in tqdm(needed_videos, desc="Video ZIPs"):
        video_dirs[vid] = download_and_extract_video_zip(
            vid, split, "clips_videos", extract_dir, cache_dir
        )

    # Process each entity track
    metadata = []
    label_counts = defaultdict(int)
    skipped = 0

    print(f"\nProcessing {len(sampled_ids)} entity tracks...")
    for entity_id in tqdm(sampled_ids, desc=f"Processing {split}"):
        # Get video_id from entity_id (format: "video_id:track_num")
        video_id = entity_id.rsplit(":", 1)[0]

        # Get track data from CSV
        track_df = df[df["entity_id"] == entity_id].sort_values("frame_timestamp")
        if track_df.empty:
            skipped += 1
            continue

        # Find audio file
        audio_src = find_entity_audio(entity_id, audio_dirs.get(video_id))
        if audio_src is None:
            skipped += 1
            continue

        # Find image files
        img_srcs = find_entity_images(entity_id, video_dirs.get(video_id))
        if len(img_srcs) == 0:
            skipped += 1
            continue

        # Subsample frames
        num_frames = len(img_srcs)
        if num_frames > max_frames:
            step = num_frames / max_frames
            selected_indices = [int(i * step) for i in range(max_frames)]
        else:
            selected_indices = list(range(num_frames))

        selected_imgs = [img_srcs[i] for i in selected_indices]

        # Get labels
        track_labels = track_df["label_id"].tolist()
        track_timestamps = track_df["frame_timestamp"].tolist()

        if len(track_labels) > max_frames:
            step = len(track_labels) / max_frames
            label_indices = [int(i * step) for i in range(max_frames)]
        else:
            label_indices = list(range(len(track_labels)))

        selected_labels = [int(track_labels[i]) for i in label_indices]
        selected_timestamps = [float(track_timestamps[i]) for i in label_indices]

        # Majority label
        speaking_count = sum(selected_labels)
        total_selected = len(selected_labels)
        majority_label = "SPEAKING" if speaking_count > total_selected / 2 else "NOT_SPEAKING"

        # Copy files to output
        safe_entity_id = entity_id.replace(":", "_")

        audio_dst = audio_dir / f"{safe_entity_id}.wav"
        shutil.copy2(audio_src, audio_dst)

        image_paths = []
        for frame_idx, img_src in enumerate(selected_imgs):
            img_dst = img_dir / f"{safe_entity_id}_frame{frame_idx:03d}.jpg"
            shutil.copy2(img_src, img_dst)
            image_paths.append(str(img_dst))

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

    # Clean up extracted ZIPs to save space
    print("Cleaning up extracted ZIP files...")
    shutil.rmtree(extract_dir, ignore_errors=True)

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
