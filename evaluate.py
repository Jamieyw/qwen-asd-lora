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
    """Build inference conversation (without assistant response).

    Uses simple prompt format — see experiment_logs/prompt_investigation.md
    for why verbose per-frame prompts cause NOT_SPEAKING collapse.
    """
    user_content = []

    # All face crop images (10 frames, matches training)
    for img_path in sample["image_paths"]:
        user_content.append({"type": "image", "image": img_path})

    user_content.append({"type": "audio", "audio": sample["audio_path"]})
    user_content.append({
        "type": "text",
        "text": "Is this person currently speaking? Answer with only SPEAKING or NOT_SPEAKING.",
    })

    conversation = [
        {
            "role": "system",
            "content": [{"type": "text", "text": "You are an active speaker detection system."}],
        },
        {
            "role": "user",
            "content": user_content,
        },
    ]

    return conversation


def extract_prediction(generated_text):
    """Extract SPEAKING/NOT_SPEAKING from generated text."""
    text = generated_text.strip().upper()

    if "NOT_SPEAKING" in text or "NOT SPEAKING" in text:
        return "NOT_SPEAKING"
    elif "SPEAKING" in text:
        return "SPEAKING"
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

    predictions = []
    ground_truths = []
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
                conversation, use_audio_in_video=False
            )
            inputs = processor(
                text=text,
                audio=audios,
                images=images,
                videos=videos,
                return_tensors="pt",
                padding=True,
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}

            # Generate
            with torch.no_grad():
                with torch.amp.autocast("cuda", dtype=dtype):
                    output_ids = model.generate(
                        **inputs,
                        return_audio=False,
                        max_new_tokens=20,
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

            prediction = extract_prediction(generated_text)
            predictions.append(prediction)
            ground_truths.append(sample["majority_label"])
            raw_outputs.append({
                "entity_id": sample["entity_id"],
                "generated_text": generated_text,
                "prediction": prediction,
                "ground_truth": sample["majority_label"],
                "correct": prediction == sample["majority_label"],
            })

        except Exception as e:
            print(f"\nError on sample {i} ({sample['entity_id']}): {e}")
            predictions.append("UNKNOWN")
            ground_truths.append(sample["majority_label"])
            raw_outputs.append({
                "entity_id": sample["entity_id"],
                "error": str(e),
                "prediction": "UNKNOWN",
                "ground_truth": sample["majority_label"],
                "correct": False,
            })

    total_time = time.time() - start_time

    # -----------------------------------------------------------------------
    # Compute metrics
    # -----------------------------------------------------------------------
    print(f"\n{'='*60}")
    print("EVALUATION RESULTS")
    print(f"{'='*60}")
    print(f"Total samples: {len(predictions)}")
    print(f"Inference time: {total_time:.1f}s ({total_time/len(predictions):.2f}s/sample)")

    results = {}

    # Filter out UNKNOWN predictions
    valid_mask = [p != "UNKNOWN" for p in predictions]
    valid_preds = [p for p, v in zip(predictions, valid_mask) if v]
    valid_gts = [g for g, v in zip(ground_truths, valid_mask) if v]
    unknown_count = len(predictions) - len(valid_preds)

    if unknown_count > 0:
        print(f"\nWARNING: {unknown_count} samples had UNKNOWN predictions")

    if valid_preds:
        label_map = {"SPEAKING": 1, "NOT_SPEAKING": 0}
        y_pred = [label_map.get(p, 0) for p in valid_preds]
        y_true = [label_map.get(g, 0) for g in valid_gts]

        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        cm = confusion_matrix(y_true, y_pred)

        # mAP
        try:
            mAP = average_precision_score(y_true, y_pred)
        except ValueError:
            mAP = None

        print(f"\nAccuracy: {accuracy:.4f} ({accuracy*100:.1f}%)")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
        if mAP is not None:
            print(f"mAP: {mAP:.4f}")

        if cm.shape == (2, 2):
            print(f"\nConfusion Matrix:")
            print(f"                  Predicted")
            print(f"                  NOT_SPEAK  SPEAKING")
            print(f"  Actual NOT_SPEAK  {cm[0][0]:>6}    {cm[0][1]:>6}")
            print(f"  Actual SPEAKING   {cm[1][0]:>6}    {cm[1][1]:>6}")

        print(f"\nClassification Report:")
        print(classification_report(
            y_true, y_pred,
            target_names=["NOT_SPEAKING", "SPEAKING"],
            digits=4,
        ))

        print(f"Random baseline: 50.0%")
        print(f"Improvement over random: {(accuracy - 0.5) * 100:+.1f}%")

        results["accuracy"] = accuracy
        results["precision"] = precision
        results["recall"] = recall
        results["f1_score"] = f1
        results["mAP"] = mAP
        results["confusion_matrix"] = cm.tolist()
    else:
        print("\nNo valid predictions — all UNKNOWN")

    results["total_samples"] = len(predictions)
    results["valid_samples"] = len(valid_preds)
    results["unknown_count"] = unknown_count
    results["inference_time_seconds"] = total_time
    results["per_sample_time"] = total_time / max(len(predictions), 1)
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
