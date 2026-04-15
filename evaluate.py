"""
evaluate.py — Evaluate the fine-tuned Qwen2.5-Omni-3B LoRA model on ASD.

Loads the base model + LoRA adapter, runs inference on validation data,
and computes classification metrics.

Usage:
    python evaluate.py [--adapter_path ./output/best_model] [--data_dir ./data]
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from peft import PeftModel
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from tqdm import tqdm
from transformers import (
    Qwen2_5OmniForConditionalGeneration,
    Qwen2_5OmniProcessor,
)


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate fine-tuned ASD model")
    p.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-Omni-3B")
    p.add_argument("--adapter_path", type=str, default="./output/best_model",
                    help="Path to LoRA adapter weights")
    p.add_argument("--data_dir", type=str, default="./data")
    p.add_argument("--output_dir", type=str, default="./output")
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--max_samples", type=int, default=None,
                    help="Limit evaluation to N samples (for quick testing)")
    p.add_argument("--fp16", action="store_true",
                    help="Use fp16 (for V100)")
    p.add_argument("--no_adapter", action="store_true",
                    help="Run evaluation WITHOUT LoRA adapter (baseline comparison)")
    return p.parse_args()


def load_model(args):
    """Load base model and optionally apply LoRA adapter."""
    dtype = torch.float16 if args.fp16 else torch.bfloat16

    print(f"Loading base model: {args.model_name}")
    model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
        args.model_name,
        torch_dtype=dtype,
        device_map="auto",
    )
    processor = Qwen2_5OmniProcessor.from_pretrained(args.model_name)

    if not args.no_adapter:
        print(f"Loading LoRA adapter from: {args.adapter_path}")
        model.thinker = PeftModel.from_pretrained(
            model.thinker,
            args.adapter_path,
        )
        print("LoRA adapter loaded successfully")
    else:
        print("Running WITHOUT LoRA adapter (baseline)")

    model.eval()
    return model, processor


def load_val_data(data_dir):
    """Load validation metadata."""
    val_dir = Path(data_dir) / "val"
    metadata_path = val_dir / "metadata.jsonl"

    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Validation metadata not found at {metadata_path}. Run prepare_data.py first."
        )

    samples = []
    with open(metadata_path) as f:
        for line in f:
            samples.append(json.loads(line.strip()))

    print(f"Loaded {len(samples)} validation samples")
    return samples


def build_conversation(sample):
    """Build inference conversation (without assistant response)."""
    user_content = []

    # Use video input if available (enables TMRoPE temporal alignment)
    video_path = sample.get("video_path")
    if video_path and Path(video_path).exists():
        user_content.append({"type": "video", "video": video_path})
    else:
        # Fallback to separate images + audio
        for img_path in sample["image_paths"]:
            user_content.append({"type": "image", "image": img_path})
        user_content.append({"type": "audio", "audio": sample["audio_path"]})

    num_frames = sample["num_frames"]
    user_content.append({
        "type": "text",
        "text": (
            f"This video shows {num_frames} sequential frames of a person's face "
            f"with corresponding audio. "
            f"For each frame, determine whether this person is actively speaking "
            f"at that moment by analyzing their lip movements and the audio. "
            f"Output one line per frame in the format: Frame N: SPEAKING or NOT_SPEAKING"
        ),
    })

    conversation = [
        {
            "role": "system",
            "content": [{"type": "text", "text": (
                "You are an active speaker detection system. "
                "Given sequential face images and audio from a video, "
                "analyze each frame to determine whether the person is "
                "speaking at that moment. Look at lip movements across frames "
                "and match them with the audio signal."
            )}],
        },
        {
            "role": "user",
            "content": user_content,
        },
    ]

    return conversation


def extract_per_frame_predictions(generated_text, num_frames):
    """
    Extract per-frame SPEAKING/NOT_SPEAKING labels from generated text.

    Expected format:
    Frame 1: SPEAKING
    Frame 2: NOT_SPEAKING
    ...
    Overall: SPEAKING (6/10 frames)
    """
    predictions = []
    text = generated_text.strip().upper()

    for i in range(1, num_frames + 1):
        # Look for "Frame N: SPEAKING" or "Frame N: NOT_SPEAKING"
        if f"FRAME {i}: NOT_SPEAKING" in text or f"FRAME {i}: NOT SPEAKING" in text:
            predictions.append(0)
        elif f"FRAME {i}: SPEAKING" in text:
            predictions.append(1)
        else:
            predictions.append(-1)  # couldn't parse

    return predictions


def extract_overall_prediction(generated_text):
    """Extract overall SPEAKING/NOT_SPEAKING from the Overall line or majority vote."""
    text = generated_text.strip().upper()

    # Try to find "Overall: SPEAKING" or "Overall: NOT_SPEAKING"
    if "OVERALL: NOT_SPEAKING" in text or "OVERALL: NOT SPEAKING" in text:
        return "NOT_SPEAKING"
    elif "OVERALL: SPEAKING" in text:
        return "SPEAKING"

    # Fallback: count per-frame predictions
    speaking = text.count("SPEAKING") - text.count("NOT_SPEAKING")
    if speaking > 0:
        return "SPEAKING"
    elif speaking < 0:
        return "NOT_SPEAKING"
    else:
        return "UNKNOWN"


def evaluate(args):
    # Load model
    model, processor = load_model(args)
    device = next(model.parameters()).device
    dtype = torch.float16 if args.fp16 else torch.bfloat16

    # Load data
    samples = load_val_data(args.data_dir)
    if args.max_samples:
        samples = samples[:args.max_samples]
        print(f"Limited to {len(samples)} samples")

    # Run inference
    from qwen_omni_utils import process_mm_info

    predictions = []       # track-level overall predictions
    ground_truths = []     # track-level overall ground truths
    all_frame_preds = []   # per-frame predictions (across all samples)
    all_frame_gts = []     # per-frame ground truths
    per_track_results = [] # list of (track_preds, track_gts) for mAP
    raw_outputs = []

    print(f"\nRunning inference on {len(samples)} samples...")
    start_time = time.time()

    for i, sample in enumerate(tqdm(samples, desc="Evaluating")):
        conversation = build_conversation(sample)

        try:
            # Prepare inputs
            text = processor.apply_chat_template(
                conversation,
                add_generation_prompt=True,
                tokenize=False,
            )
            audios, images, videos = process_mm_info(
                conversation, use_audio_in_video=True
            )
            inputs = processor(
                text=text,
                audio=audios,
                images=images,
                videos=videos,
                return_tensors="pt",
                padding=True,
                use_audio_in_video=True,
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}

            # Generate — need more tokens now for per-frame output
            num_frames = sample["num_frames"]
            max_tokens = num_frames * 15 + 30  # ~15 tokens per "Frame N: SPEAKING\n" + overall

            with torch.no_grad():
                with torch.amp.autocast("cuda", dtype=dtype):
                    output_ids = model.generate(
                        **inputs,
                        return_audio=False,
                        use_audio_in_video=True,
                        max_new_tokens=max_tokens,
                        do_sample=False,
                    )

            # Handle output format - generate may return tuple or tensor
            if isinstance(output_ids, tuple):
                output_ids = output_ids[0]

            # Decode only the new tokens
            input_len = inputs["input_ids"].shape[-1]
            generated_ids = output_ids[:, input_len:]
            generated_text = processor.batch_decode(
                generated_ids, skip_special_tokens=True
            )[0]

            # Extract per-frame predictions
            frame_preds = extract_per_frame_predictions(generated_text, num_frames)
            overall_pred = extract_overall_prediction(generated_text)

            # Per-frame ground truth
            frame_gts = sample["labels"]

            # Collect per-frame results
            track_fp = []
            track_fg = []
            for fp, fg in zip(frame_preds, frame_gts):
                if fp != -1:  # only count parseable predictions
                    all_frame_preds.append(fp)
                    all_frame_gts.append(fg)
                    track_fp.append(fp)
                    track_fg.append(fg)
            if track_fp:
                per_track_results.append((track_fp, track_fg))

            # Track overall (track-level) predictions
            predictions.append(overall_pred)
            ground_truths.append(sample["majority_label"])
            raw_outputs.append({
                "entity_id": sample["entity_id"],
                "generated_text": generated_text,
                "frame_predictions": frame_preds,
                "frame_ground_truth": frame_gts,
                "overall_prediction": overall_pred,
                "overall_ground_truth": sample["majority_label"],
                "overall_correct": overall_pred == sample["majority_label"],
            })

        except Exception as e:
            print(f"\nError on sample {i} ({sample['entity_id']}): {e}")
            predictions.append("UNKNOWN")
            ground_truths.append(sample["majority_label"])
            raw_outputs.append({
                "entity_id": sample["entity_id"],
                "error": str(e),
                "overall_prediction": "UNKNOWN",
                "overall_ground_truth": sample["majority_label"],
                "overall_correct": False,
            })

    total_time = time.time() - start_time

    # -----------------------------------------------------------------------
    # Compute metrics
    # -----------------------------------------------------------------------
    print(f"\n{'='*60}")
    print("EVALUATION RESULTS")
    print(f"{'='*60}")
    print(f"Total track samples: {len(predictions)}")
    print(f"Inference time: {total_time:.1f}s ({total_time/len(predictions):.2f}s/sample)")

    results = {}

    # --- Per-frame metrics (primary metric for ASD) ---
    print(f"\n--- Per-Frame Metrics (Primary) ---")
    print(f"Total frame predictions: {len(all_frame_preds)}")

    if all_frame_preds:
        frame_acc = accuracy_score(all_frame_gts, all_frame_preds)
        frame_prec = precision_score(all_frame_gts, all_frame_preds, zero_division=0)
        frame_rec = recall_score(all_frame_gts, all_frame_preds, zero_division=0)
        frame_f1 = f1_score(all_frame_gts, all_frame_preds, zero_division=0)
        frame_cm = confusion_matrix(all_frame_gts, all_frame_preds)

        print(f"Frame Accuracy: {frame_acc:.4f} ({frame_acc*100:.1f}%)")
        print(f"Frame Precision: {frame_prec:.4f}")
        print(f"Frame Recall: {frame_rec:.4f}")
        print(f"Frame F1 Score: {frame_f1:.4f}")

        if frame_cm.shape == (2, 2):
            print(f"\nFrame Confusion Matrix:")
            print(f"                  Predicted")
            print(f"                  NOT_SPEAK  SPEAKING")
            print(f"  Actual NOT_SPEAK  {frame_cm[0][0]:>6}    {frame_cm[0][1]:>6}")
            print(f"  Actual SPEAKING   {frame_cm[1][0]:>6}    {frame_cm[1][1]:>6}")

        print(f"\nFrame Classification Report:")
        print(classification_report(
            all_frame_gts, all_frame_preds,
            target_names=["NOT_SPEAKING", "SPEAKING"],
            digits=4,
        ))

        # Frame-level mAP (AP across all frames globally)
        try:
            frame_mAP = average_precision_score(all_frame_gts, all_frame_preds)
            print(f"Frame mAP: {frame_mAP:.4f}")
        except ValueError:
            frame_mAP = None
            print(f"Frame mAP: N/A (need both classes in ground truth)")

        # Track-level mAP (AP per track, then average — standard ASD metric)
        per_track_aps = []
        for t_preds, t_gts in per_track_results:
            if len(set(t_gts)) > 1:  # need both classes to compute AP
                ap = average_precision_score(t_gts, t_preds)
                per_track_aps.append(ap)
        if per_track_aps:
            track_mAP = np.mean(per_track_aps)
            print(f"Track mAP: {track_mAP:.4f} (computed over {len(per_track_aps)}/{len(per_track_results)} tracks)")
        else:
            track_mAP = None
            print(f"Track mAP: N/A (no tracks had both speaking and not-speaking frames)")

        print(f"\nNote: mAP is computed with binary predictions (0/1), not confidence")
        print(f"scores, since we use a generative model. This is a lower bound on")
        print(f"what mAP would be with continuous probability scores.")

        results["frame_accuracy"] = frame_acc
        results["frame_precision"] = frame_prec
        results["frame_recall"] = frame_rec
        results["frame_f1"] = frame_f1
        results["frame_mAP"] = frame_mAP
        results["track_mAP"] = track_mAP
        results["track_mAP_num_tracks"] = len(per_track_aps)
        results["frame_confusion_matrix"] = frame_cm.tolist()
        results["total_frames_evaluated"] = len(all_frame_preds)

    # --- Track-level metrics (overall per entity) ---
    print(f"\n--- Track-Level Metrics (Overall) ---")

    valid_mask = [p != "UNKNOWN" for p in predictions]
    valid_preds = [p for p, v in zip(predictions, valid_mask) if v]
    valid_gts = [g for g, v in zip(ground_truths, valid_mask) if v]
    unknown_count = len(predictions) - len(valid_preds)

    if unknown_count > 0:
        print(f"WARNING: {unknown_count} samples had UNKNOWN overall predictions")

    if valid_preds:
        label_map = {"SPEAKING": 1, "NOT_SPEAKING": 0}
        y_pred = [label_map.get(p, 0) for p in valid_preds]
        y_true = [label_map.get(g, 0) for g in valid_gts]

        track_acc = accuracy_score(y_true, y_pred)
        track_f1 = f1_score(y_true, y_pred, zero_division=0)

        print(f"Track Accuracy: {track_acc:.4f} ({track_acc*100:.1f}%)")
        print(f"Track F1 Score: {track_f1:.4f}")

        results["track_accuracy"] = track_acc
        results["track_f1"] = track_f1

    # Random baseline comparison
    print(f"\nRandom baseline: 50.0%")
    if all_frame_preds:
        print(f"Frame improvement over random: {(frame_acc - 0.5) * 100:+.1f}%")

    results["total_samples"] = len(predictions)
    results["unknown_count"] = unknown_count
    results["inference_time_seconds"] = total_time
    results["adapter_path"] = args.adapter_path
    results["is_baseline"] = args.no_adapter

    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    suffix = "_baseline" if args.no_adapter else ""
    with open(output_dir / f"eval_results{suffix}.json", "w") as f:
        json.dump(results, f, indent=2)

    with open(output_dir / f"eval_predictions{suffix}.jsonl", "w") as f:
        for entry in raw_outputs:
            f.write(json.dumps(entry) + "\n")

    print(f"\nResults saved to {output_dir}/eval_results{suffix}.json")
    print(f"Predictions saved to {output_dir}/eval_predictions{suffix}.jsonl")


if __name__ == "__main__":
    args = parse_args()
    evaluate(args)
