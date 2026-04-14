"""
prepare_data.py — Download and prepare a subset of UniTalk-ASD for training.

Strategy to minimize disk usage and download time:
  1. Download only the small CSV metadata files (~575MB)
  2. Pick a small number of videos that have enough entity tracks
  3. Sample tracks from ONLY those videos
  4. Download ZIPs for only those videos (not all 80+)
  5. Process one video at a time: extract → copy needed files → delete extraction

Usage:
    python prepare_data.py [--num_samples 2000] [--output_dir ./data] [--seed 42]
"""

import argparse
import json
import os
import random
import shutil
import zipfile
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download, list_repo_tree
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
    parser.add_argument("--max_videos", type=int, default=30,
                        help="Max videos to sample from (fewer = fewer ZIPs to download)")
    parser.add_argument("--num_workers", type=int, default=8,
                        help="Parallel download threads (default: 8)")
    return parser.parse_args()


# -----------------------------------------------------------------------
# Step 1: Download and parse CSV metadata (small files, fast)
# -----------------------------------------------------------------------

def download_csv_files(split, cache_dir=None):
    """Download all CSV files for a split and combine into one DataFrame."""
    print(f"\nListing CSV files for {split} split...")

    csv_prefix = f"csv/{split}/"
    all_files = list_repo_tree(DATASET_REPO, path_in_repo=csv_prefix, repo_type="dataset")
    csv_files = [f.path for f in all_files if f.path.endswith(".csv")]

    print(f"Found {len(csv_files)} CSV files for {split}")

    # Download CSVs in parallel
    def _download_csv(csv_path):
        local_path = hf_hub_download(
            DATASET_REPO,
            filename=csv_path,
            repo_type="dataset",
            cache_dir=cache_dir,
        )
        return local_path

    all_dfs = []
    with ThreadPoolExecutor(max_workers=16) as executor:
        futures = {executor.submit(_download_csv, p): p for p in csv_files}
        for future in tqdm(as_completed(futures), total=len(futures), desc=f"Downloading {split} CSVs"):
            csv_path = futures[future]
            try:
                local_path = future.result()
                df = pd.read_csv(local_path, header=None, names=CSV_COLUMNS)
                all_dfs.append(df)
            except Exception as e:
                print(f"  Warning: Could not process {csv_path}: {e}")
                continue

    combined = pd.concat(all_dfs, ignore_index=True)
    print(f"Total rows in {split}: {len(combined):,}")
    print(f"Unique entity tracks: {combined['entity_id'].nunique():,}")
    print(f"Unique videos: {combined['video_id'].nunique():,}")
    return combined


# -----------------------------------------------------------------------
# Step 2: Pick videos first, then sample tracks from those videos only
# -----------------------------------------------------------------------

def sample_tracks_from_few_videos(df, num_samples, max_videos, seed):
    """
    Instead of sampling tracks randomly (which spreads across many videos),
    pick a small set of videos that have enough tracks, then sample from those.
    This means we only need to download ZIPs for max_videos instead of 80+.
    """
    random.seed(seed)

    # Count tracks per video, with label info
    track_info = df.groupby(["video_id", "entity_id"])["label_id"].mean().reset_index()
    track_info["majority"] = (track_info["label_id"] > 0.5).map(
        {True: "SPEAKING", False: "NOT_SPEAKING"}
    )

    tracks_per_video = track_info.groupby("video_id").size().sort_values(ascending=False)

    print(f"\nTotal videos: {len(tracks_per_video)}")
    print(f"Tracks per video: min={tracks_per_video.min()}, "
          f"median={tracks_per_video.median():.0f}, max={tracks_per_video.max()}")

    # Pick videos with the most tracks (so we can get enough samples from fewer ZIPs)
    # But shuffle a bit to add variety
    top_videos = tracks_per_video.head(max_videos * 2).index.tolist()
    selected_videos = random.sample(top_videos, min(max_videos, len(top_videos)))

    # Get all tracks from selected videos
    selected_tracks = track_info[track_info["video_id"].isin(selected_videos)]
    speaking_ids = selected_tracks[selected_tracks["majority"] == "SPEAKING"]["entity_id"].tolist()
    not_speaking_ids = selected_tracks[selected_tracks["majority"] == "NOT_SPEAKING"]["entity_id"].tolist()

    available = len(speaking_ids) + len(not_speaking_ids)
    print(f"\nSelected {len(selected_videos)} videos with {available} total tracks")
    print(f"  SPEAKING: {len(speaking_ids)}")
    print(f"  NOT_SPEAKING: {len(not_speaking_ids)}")

    if available < num_samples:
        print(f"  Warning: only {available} tracks available, requested {num_samples}")
        num_samples = available

    # Balanced sampling
    half = num_samples // 2
    n_speaking = min(half, len(speaking_ids))
    n_not_speaking = min(num_samples - n_speaking, len(not_speaking_ids))
    if n_not_speaking < num_samples - n_speaking:
        n_speaking = min(num_samples - n_not_speaking, len(speaking_ids))

    sampled = random.sample(speaking_ids, n_speaking) + random.sample(not_speaking_ids, n_not_speaking)
    random.shuffle(sampled)

    # Which videos do we actually need?
    sampled_df = df[df["entity_id"].isin(sampled)]
    needed_videos = sampled_df["video_id"].unique().tolist()

    print(f"Sampled {len(sampled)} tracks from {len(needed_videos)} videos "
          f"({n_speaking} speaking, {n_not_speaking} not speaking)")

    return sampled, needed_videos


# -----------------------------------------------------------------------
# Step 3: Download ZIPs, extract one video at a time, process, clean up
# -----------------------------------------------------------------------

def download_zip(video_id, split, media_type, cache_dir=None):
    """Download a single ZIP file (returns local path or None)."""
    zip_hf_path = f"{media_type}/{split}/{video_id}.zip"
    try:
        return hf_hub_download(
            DATASET_REPO,
            filename=zip_hf_path,
            repo_type="dataset",
            cache_dir=cache_dir,
        )
    except Exception as e:
        return None


def find_entity_audio_in_zip(zf, entity_id):
    """Find audio WAV for an entity inside an open ZipFile."""
    # Try common patterns
    candidates = [
        f"{entity_id}.wav",
        f"{entity_id.replace(':', '_')}.wav",
    ]
    all_names = zf.namelist()

    for name in all_names:
        basename = os.path.basename(name)
        if basename in [f"{entity_id}.wav", f"{entity_id.replace(':', '_')}.wav"]:
            return name
        # Also match by entity_id appearing in the path
        if entity_id in name and name.endswith(".wav"):
            return name

    return None


def find_entity_images_in_zip(zf, entity_id):
    """Find face crop images for an entity inside an open ZipFile."""
    matches = []
    for name in zf.namelist():
        # Images are in: {entity_id}/*.jpg or similar
        if entity_id in name and name.endswith(".jpg"):
            matches.append(name)
        elif entity_id.replace(":", "_") in name and name.endswith(".jpg"):
            matches.append(name)
    return sorted(matches)


def process_video_tracks(video_id, entity_ids, df, split, max_frames,
                         output_dir, cache_dir=None):
    """
    Process all sampled tracks from one video:
    1. Download audio + video ZIPs
    2. Open ZIPs (no full extraction to disk)
    3. Extract only the needed entity files
    4. Done — ZIPs stay in HF cache, no temp extraction needed
    """
    split_dir = Path(output_dir) / split
    img_dir = split_dir / "images"
    audio_dir = split_dir / "audio"

    # Download ZIPs
    audio_zip_path = download_zip(video_id, split, "clips_audios", cache_dir)
    video_zip_path = download_zip(video_id, split, "clips_videos", cache_dir)

    if audio_zip_path is None or video_zip_path is None:
        return [], len(entity_ids)

    results = []
    skipped = 0

    try:
        audio_zf = zipfile.ZipFile(audio_zip_path, "r")
        video_zf = zipfile.ZipFile(video_zip_path, "r")
    except Exception as e:
        print(f"  Warning: Could not open ZIPs for {video_id}: {e}")
        return [], len(entity_ids)

    with audio_zf, video_zf:
        for entity_id in entity_ids:
            track_df = df[df["entity_id"] == entity_id].sort_values("frame_timestamp")
            if track_df.empty:
                skipped += 1
                continue

            # Find audio in ZIP
            audio_name = find_entity_audio_in_zip(audio_zf, entity_id)
            if audio_name is None:
                skipped += 1
                continue

            # Find images in ZIP
            image_names = find_entity_images_in_zip(video_zf, entity_id)
            if len(image_names) == 0:
                skipped += 1
                continue

            # Subsample frames
            num_frames = len(image_names)
            if num_frames > max_frames:
                step = num_frames / max_frames
                selected_indices = [int(i * step) for i in range(max_frames)]
            else:
                selected_indices = list(range(num_frames))
            selected_image_names = [image_names[i] for i in selected_indices]

            # Subsample labels
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

            safe_id = entity_id.replace(":", "_")

            # Extract audio directly from ZIP to output
            audio_dst = audio_dir / f"{safe_id}.wav"
            with audio_zf.open(audio_name) as src, open(audio_dst, "wb") as dst:
                dst.write(src.read())

            # Extract selected images directly from ZIP to output
            image_paths = []
            for frame_idx, img_name in enumerate(selected_image_names):
                img_dst = img_dir / f"{safe_id}_frame{frame_idx:03d}.jpg"
                with video_zf.open(img_name) as src, open(img_dst, "wb") as dst:
                    dst.write(src.read())
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
                "num_frames": len(selected_image_names),
                "speaking_ratio": speaking_count / total_selected if total_selected > 0 else 0,
            }
            results.append(entry)

    return results, skipped


def process_split(df, sampled_ids, needed_videos, split, max_frames,
                  output_dir, num_workers, cache_dir=None):
    """Process all tracks for a split, one video at a time."""
    split_dir = Path(output_dir) / split
    img_dir = split_dir / "images"
    audio_dir = split_dir / "audio"
    img_dir.mkdir(parents=True, exist_ok=True)
    audio_dir.mkdir(parents=True, exist_ok=True)

    # Group sampled entity_ids by video
    sampled_set = set(sampled_ids)
    video_to_entities = defaultdict(list)
    for _, row in df[df["entity_id"].isin(sampled_set)].groupby("video_id"):
        vid = row["video_id"].iloc[0]
        entities = row["entity_id"].unique().tolist()
        video_to_entities[vid] = [e for e in entities if e in sampled_set]

    metadata = []
    label_counts = defaultdict(int)
    total_skipped = 0

    print(f"\nProcessing {len(sampled_ids)} tracks from {len(needed_videos)} videos...")

    for vid in tqdm(needed_videos, desc=f"Processing {split} videos"):
        entity_ids = video_to_entities.get(vid, [])
        if not entity_ids:
            continue

        results, skipped = process_video_tracks(
            vid, entity_ids, df, split, max_frames, output_dir, cache_dir
        )
        total_skipped += skipped

        for entry in results:
            metadata.append(entry)
            label_counts[entry["majority_label"]] += 1

    # Save metadata
    metadata_path = split_dir / "metadata.jsonl"
    with open(metadata_path, "w") as f:
        for entry in metadata:
            f.write(json.dumps(entry) + "\n")

    print(f"\n{split} statistics:")
    print(f"  Total processed: {len(metadata)}")
    print(f"  Skipped (missing media): {total_skipped}")
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
    print(f"Max videos to sample from: {args.max_videos}")
    print(f"Output directory: {args.output_dir}")
    print(f"Random seed: {args.seed}")
    print("=" * 60)

    # --- Training data ---
    print("\n--- Training Data ---")
    train_df = download_csv_files("train", args.cache_dir)
    train_sampled_ids, train_videos = sample_tracks_from_few_videos(
        train_df, args.num_samples, args.max_videos, args.seed
    )
    process_split(
        train_df, train_sampled_ids, train_videos, "train",
        args.max_frames, args.output_dir, args.num_workers, args.cache_dir,
    )
    del train_df

    # --- Validation data ---
    print("\n--- Validation Data ---")
    val_df = download_csv_files("val", args.cache_dir)
    val_sampled_ids, val_videos = sample_tracks_from_few_videos(
        val_df, args.val_samples, max(args.max_videos // 3, 10), args.seed + 1
    )
    process_split(
        val_df, val_sampled_ids, val_videos, "val",
        args.max_frames, args.output_dir, args.num_workers, args.cache_dir,
    )

    print("\n" + "=" * 60)
    print("Data preparation complete!")
    print(f"Training data: {args.output_dir}/train/")
    print(f"Validation data: {args.output_dir}/val/")
    print("=" * 60)


if __name__ == "__main__":
    main()
