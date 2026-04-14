"""
train.py — LoRA fine-tuning of Qwen2.5-Omni-3B for Active Speaker Detection.

Loads the pre-trained model, attaches LoRA adapters to the thinker component,
and trains on the prepared UniTalk-ASD subset.

Usage:
    python train.py [--data_dir ./data] [--output_dir ./output] [--epochs 3]

The model learns to answer "Is this person speaking?" given face images + audio.
"""

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from PIL import Image
from peft import LoraConfig, get_peft_model, TaskType
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import (
    Qwen2_5OmniForConditionalGeneration,
    Qwen2_5OmniProcessor,
    get_linear_schedule_with_warmup,
)


# ---------------------------------------------------------------------------
# Arguments
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="LoRA fine-tuning for ASD")
    p.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-Omni-3B")
    p.add_argument("--data_dir", type=str, default="./data")
    p.add_argument("--output_dir", type=str, default="./output")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch_size", type=int, default=1,
                   help="Batch size per GPU (1 for multimodal, use grad accum for effective batch)")
    p.add_argument("--gradient_accumulation_steps", type=int, default=8)
    p.add_argument("--learning_rate", type=float, default=2e-4)
    p.add_argument("--warmup_steps", type=int, default=100)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--lora_r", type=int, default=8)
    p.add_argument("--lora_alpha", type=int, default=16)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--logging_steps", type=int, default=10)
    p.add_argument("--save_steps", type=int, default=500)
    p.add_argument("--timing_test_steps", type=int, default=50,
                   help="Run N steps first to estimate total time (0 to skip)")
    p.add_argument("--gradient_checkpointing", action="store_true",
                   help="Enable gradient checkpointing to save memory")
    p.add_argument("--fp16", action="store_true",
                   help="Use fp16 instead of bf16 (for V100 which lacks bf16)")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class ASDDataset(Dataset):
    """
    Active Speaker Detection dataset with per-frame labels.

    Each sample is formatted as a conversation for Qwen2.5-Omni:
    - User: [face images] + [audio] + "For each frame, is this person speaking?"
    - Assistant: "Frame 1: SPEAKING\nFrame 2: NOT_SPEAKING\n..." (per-frame labels)

    This forces the model to actually analyze each frame rather than
    taking shortcuts with a single-token answer.
    """

    def __init__(self, data_dir, split="train"):
        self.data_dir = Path(data_dir) / split
        self.metadata = []

        metadata_path = self.data_dir / "metadata.jsonl"
        if not metadata_path.exists():
            raise FileNotFoundError(
                f"Metadata not found at {metadata_path}. Run prepare_data.py first."
            )

        with open(metadata_path) as f:
            for line in f:
                self.metadata.append(json.loads(line.strip()))

        print(f"Loaded {len(self.metadata)} samples from {split}")

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        entry = self.metadata[idx]

        # Build conversation in Qwen2.5-Omni format
        user_content = []

        # Add face crop images (already saved as JPG)
        for i, img_path in enumerate(entry["image_paths"]):
            user_content.append({
                "type": "image",
                "image": img_path,
            })

        # Add audio
        user_content.append({
            "type": "audio",
            "audio": entry["audio_path"],
        })

        # Add the question — ask for per-frame analysis
        num_frames = entry["num_frames"]
        user_content.append({
            "type": "text",
            "text": (
                f"These are {num_frames} sequential frames of a person's face "
                f"extracted from a video, along with the corresponding audio. "
                f"For each frame, determine whether this person is actively speaking "
                f"at that moment by analyzing their lip movements and the audio. "
                f"Output one line per frame in the format: Frame N: SPEAKING or NOT_SPEAKING"
            ),
        })

        # Build per-frame answer string
        # e.g. "Frame 1: SPEAKING\nFrame 2: NOT_SPEAKING\n..."
        labels = entry["labels"]
        answer_lines = []
        for i, label in enumerate(labels):
            status = "SPEAKING" if label == 1 else "NOT_SPEAKING"
            answer_lines.append(f"Frame {i+1}: {status}")
        answer_text = "\n".join(answer_lines)

        # Also add overall summary
        speaking_count = sum(labels)
        total = len(labels)
        overall = "SPEAKING" if speaking_count > total / 2 else "NOT_SPEAKING"
        answer_text += f"\nOverall: {overall} ({speaking_count}/{total} frames)"

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
            {
                "role": "assistant",
                "content": [{"type": "text", "text": answer_text}],
            },
        ]

        return {
            "conversation": conversation,
            "label": answer_text,
            "labels": labels,
            "majority_label": entry["majority_label"],
            "entity_id": entry["entity_id"],
        }


def collate_fn(batch, processor):
    """
    Collate function that processes conversations into model inputs.

    Since multimodal inputs have variable sizes, we process one at a time
    and let the processor handle padding.
    """
    from qwen_omni_utils import process_mm_info

    all_input_ids = []
    all_attention_masks = []
    all_labels_tensors = []

    for sample in batch:
        conversation = sample["conversation"]

        # Apply chat template to get the full text
        text = processor.apply_chat_template(
            conversation,
            add_generation_prompt=False,
            tokenize=False,
        )

        # Process multimodal info (images, audio)
        audios, images, videos = process_mm_info(conversation, use_audio_in_video=False)

        # Tokenize and process
        inputs = processor(
            text=text,
            audio=audios,
            images=images,
            videos=videos,
            return_tensors="pt",
            padding=True,
        )

        input_ids = inputs["input_ids"].squeeze(0)
        attention_mask = inputs["attention_mask"].squeeze(0)

        # Create labels: mask everything except the assistant's response
        # Labels = -100 for tokens we don't want to compute loss on
        labels = input_ids.clone()

        # Find where the assistant response starts
        # The assistant response is the label text (SPEAKING/NOT_SPEAKING)
        # We mask everything before it
        label_text = sample["label"]
        label_tokens = processor.tokenizer.encode(label_text, add_special_tokens=False)
        label_len = len(label_tokens)

        # Mask all tokens except the last label_len tokens (+ EOS)
        # This makes the model only learn to predict the answer
        if label_len < len(labels):
            labels[:-label_len - 1] = -100  # -1 for EOS token

        all_input_ids.append(input_ids)
        all_attention_masks.append(attention_mask)
        all_labels_tensors.append(labels)

    # Pad sequences to same length
    max_len = max(ids.size(0) for ids in all_input_ids)

    padded_input_ids = []
    padded_attention_masks = []
    padded_labels = []

    pad_token_id = processor.tokenizer.pad_token_id or 0

    for input_ids, attn_mask, labels in zip(
        all_input_ids, all_attention_masks, all_labels_tensors
    ):
        pad_len = max_len - input_ids.size(0)
        if pad_len > 0:
            input_ids = torch.cat([input_ids, torch.full((pad_len,), pad_token_id)])
            attn_mask = torch.cat([attn_mask, torch.zeros(pad_len, dtype=attn_mask.dtype)])
            labels = torch.cat([labels, torch.full((pad_len,), -100)])
        padded_input_ids.append(input_ids)
        padded_attention_masks.append(attn_mask)
        padded_labels.append(labels)

    return {
        "input_ids": torch.stack(padded_input_ids),
        "attention_mask": torch.stack(padded_attention_masks),
        "labels": torch.stack(padded_labels),
    }


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def run_timing_test(model, dataloader, device, num_steps, collate_fn_with_proc):
    """Run a few steps and estimate total training time."""
    print(f"\n{'='*60}")
    print(f"Running timing test ({num_steps} steps)...")
    print(f"{'='*60}")

    model.train()
    start_time = time.time()

    for i, batch in enumerate(dataloader):
        if i >= num_steps:
            break

        batch = {k: v.to(device) for k, v in batch.items()}

        with torch.amp.autocast("cuda", dtype=torch.float16):
            outputs = model.thinker(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
            )
            loss = outputs.loss

        loss.backward()
        model.zero_grad()

        if (i + 1) % 10 == 0:
            elapsed = time.time() - start_time
            print(f"  Step {i+1}/{num_steps}: {elapsed:.1f}s elapsed, "
                  f"~{elapsed/(i+1):.2f}s/step")

    elapsed = time.time() - start_time
    per_step = elapsed / min(num_steps, len(dataloader))

    return per_step


def train(args):
    # Set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # -----------------------------------------------------------------------
    # Load model and processor
    # -----------------------------------------------------------------------
    print(f"\nLoading model: {args.model_name}")

    # Use fp16 for V100 (no native bf16 support), bf16 for newer GPUs
    dtype = torch.float16 if args.fp16 else torch.bfloat16

    model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
        args.model_name,
        torch_dtype=dtype,
        device_map="auto",
    )
    processor = Qwen2_5OmniProcessor.from_pretrained(args.model_name)

    # Ensure pad token is set
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token

    print(f"Model loaded. Total parameters: {sum(p.numel() for p in model.parameters()):,}")

    # -----------------------------------------------------------------------
    # Apply LoRA to the thinker component
    # -----------------------------------------------------------------------
    print("\nApplying LoRA configuration...")

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    # The thinker's config is missing vocab_size, which the loss function needs.
    # Get it from the embedding layer (always reliable).
    if not hasattr(model.thinker.config, "vocab_size"):
        vocab_size = model.thinker.lm_head.out_features
        model.thinker.config.vocab_size = vocab_size
        print(f"Set thinker vocab_size = {vocab_size}")

    # Apply LoRA specifically to the thinker (text generation) component
    model.thinker = get_peft_model(model.thinker, lora_config)
    model.thinker.print_trainable_parameters()

    if args.gradient_checkpointing:
        model.thinker.enable_input_require_grads()
        model.thinker.gradient_checkpointing_enable()
        print("Gradient checkpointing enabled")

    # -----------------------------------------------------------------------
    # Load data
    # -----------------------------------------------------------------------
    print("\nLoading dataset...")
    train_dataset = ASDDataset(args.data_dir, split="train")

    # Create collate function with processor bound
    def collate_fn_bound(batch):
        return collate_fn(batch, processor)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn_bound,
        num_workers=2,
        pin_memory=True,
    )

    # -----------------------------------------------------------------------
    # Timing test
    # -----------------------------------------------------------------------
    if args.timing_test_steps > 0:
        per_step = run_timing_test(
            model, train_dataloader, device,
            args.timing_test_steps, collate_fn_bound,
        )

        total_steps = len(train_dataloader) * args.epochs
        optimizer_steps = total_steps // args.gradient_accumulation_steps
        estimated_time = per_step * total_steps

        print(f"\nTiming results:")
        print(f"  Per step: {per_step:.2f}s")
        print(f"  Total training steps: {total_steps}")
        print(f"  Optimizer steps: {optimizer_steps}")
        print(f"  Estimated total time: {estimated_time/3600:.1f} hours")

        if estimated_time > 4 * 3600:
            print(f"\n  WARNING: Estimated time ({estimated_time/3600:.1f}h) exceeds 4 hour limit!")
            print(f"  Consider reducing --epochs or data subset size.")
        print()

    # -----------------------------------------------------------------------
    # Optimizer and scheduler
    # -----------------------------------------------------------------------
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    total_steps = len(train_dataloader) * args.epochs
    optimizer_steps = total_steps // args.gradient_accumulation_steps

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=optimizer_steps,
    )

    # -----------------------------------------------------------------------
    # Training loop
    # -----------------------------------------------------------------------
    print("=" * 60)
    print("Starting training")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Gradient accumulation: {args.gradient_accumulation_steps}")
    print(f"  Effective batch size: {args.batch_size * args.gradient_accumulation_steps}")
    print(f"  Learning rate: {args.learning_rate}")
    print(f"  LoRA rank: {args.lora_r}, alpha: {args.lora_alpha}")
    print(f"  Total optimizer steps: {optimizer_steps}")
    print("=" * 60)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save training config
    config = vars(args)
    config["total_training_samples"] = len(train_dataset)
    config["total_optimizer_steps"] = optimizer_steps
    with open(output_dir / "training_config.json", "w") as f:
        json.dump(config, f, indent=2)

    model.train()
    global_step = 0
    total_loss = 0
    best_loss = float("inf")
    training_start = time.time()
    log_history = []

    for epoch in range(args.epochs):
        epoch_start = time.time()
        epoch_loss = 0
        num_batches = 0

        progress_bar = tqdm(
            train_dataloader,
            desc=f"Epoch {epoch+1}/{args.epochs}",
            leave=True,
        )

        for step, batch in enumerate(progress_bar):
            # Move batch to device
            batch = {k: v.to(device) for k, v in batch.items()}

            # Forward pass through thinker (text generation component)
            with torch.amp.autocast("cuda", dtype=dtype):
                outputs = model.thinker(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"],
                )
                loss = outputs.loss / args.gradient_accumulation_steps

            # Backward pass
            loss.backward()

            total_loss += loss.item()
            epoch_loss += loss.item() * args.gradient_accumulation_steps
            num_batches += 1

            # Optimizer step (with gradient accumulation)
            if (step + 1) % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad],
                    args.max_grad_norm,
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                # Logging
                if global_step % args.logging_steps == 0:
                    avg_loss = total_loss / args.logging_steps
                    elapsed = time.time() - training_start
                    lr = scheduler.get_last_lr()[0]

                    log_entry = {
                        "step": global_step,
                        "epoch": epoch + 1,
                        "loss": round(avg_loss, 4),
                        "lr": lr,
                        "elapsed_min": round(elapsed / 60, 1),
                    }
                    log_history.append(log_entry)

                    progress_bar.set_postfix({
                        "loss": f"{avg_loss:.4f}",
                        "lr": f"{lr:.2e}",
                        "time": f"{elapsed/60:.0f}m",
                    })

                    total_loss = 0

                # Save checkpoint
                if args.save_steps > 0 and global_step % args.save_steps == 0:
                    ckpt_dir = output_dir / f"checkpoint-{global_step}"
                    model.thinker.save_pretrained(str(ckpt_dir))
                    print(f"\n  Checkpoint saved to {ckpt_dir}")

        # End of epoch
        epoch_time = time.time() - epoch_start
        avg_epoch_loss = epoch_loss / max(num_batches, 1)

        print(f"\nEpoch {epoch+1}/{args.epochs} complete:")
        print(f"  Average loss: {avg_epoch_loss:.4f}")
        print(f"  Time: {epoch_time/60:.1f} minutes")
        print(f"  Total elapsed: {(time.time() - training_start)/60:.1f} minutes")

        # Save best model
        if avg_epoch_loss < best_loss:
            best_loss = avg_epoch_loss
            best_dir = output_dir / "best_model"
            model.thinker.save_pretrained(str(best_dir))
            processor.save_pretrained(str(best_dir))
            print(f"  New best model saved (loss: {best_loss:.4f})")

    # -----------------------------------------------------------------------
    # Save final model
    # -----------------------------------------------------------------------
    total_time = time.time() - training_start
    print(f"\n{'='*60}")
    print("Training complete!")
    print(f"  Total time: {total_time/60:.1f} minutes ({total_time/3600:.2f} hours)")
    print(f"  Best loss: {best_loss:.4f}")
    print(f"{'='*60}")

    # Save final adapter
    final_dir = output_dir / "final_model"
    model.thinker.save_pretrained(str(final_dir))
    processor.save_pretrained(str(final_dir))
    print(f"Final model saved to {final_dir}")

    # Save training log
    with open(output_dir / "training_log.json", "w") as f:
        json.dump({
            "log_history": log_history,
            "total_time_seconds": total_time,
            "best_loss": best_loss,
            "final_epoch_loss": avg_epoch_loss,
        }, f, indent=2)

    print(f"Training log saved to {output_dir / 'training_log.json'}")


if __name__ == "__main__":
    args = parse_args()
    train(args)
