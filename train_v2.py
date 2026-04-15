"""
train_v2.py — Improved LoRA fine-tuning of Qwen2.5-Omni-3B for Active Speaker Detection.

Changes from train.py:
  - Label smoothing (default 0.1) to prevent overconfident predictions
  - LoRA on audio encoder last N layers (default 8) with lower LR
  - Updated hyperparameters: higher rank (16), lower LR (5e-5), more dropout (0.1)
  - LoRA targets feed-forward layers too (gate_proj, up_proj, down_proj)
  - Separate optimizer parameter groups for thinker vs audio encoder

Usage:
    python train_v2.py [--data_dir ./data] [--output_dir ./output] [--epochs 3]
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
    p = argparse.ArgumentParser(description="LoRA fine-tuning for ASD (v2)")
    p.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-Omni-3B")
    p.add_argument("--data_dir", type=str, default="./data")
    p.add_argument("--output_dir", type=str, default="./output")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch_size", type=int, default=1,
                   help="Batch size per GPU (1 for multimodal, use grad accum for effective batch)")
    p.add_argument("--gradient_accumulation_steps", type=int, default=8)
    p.add_argument("--learning_rate", type=float, default=1e-4)
    p.add_argument("--warmup_steps", type=int, default=50)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--lora_r", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--lora_dropout", type=float, default=0.1)
    p.add_argument("--label_smoothing", type=float, default=0.1,
                   help="Label smoothing factor (0.0 = hard labels, 0.1 = recommended)")
    p.add_argument("--unfreeze_audio_layers", type=int, default=8,
                   help="Number of audio encoder layers to apply LoRA to (from the end, 0 to disable)")
    p.add_argument("--audio_lr_scale", type=float, default=0.1,
                   help="Audio encoder LR = learning_rate * this scale (default: 0.1)")
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
    Active Speaker Detection dataset.

    Each sample is formatted as a conversation for Qwen2.5-Omni:
    - User: [face images] + [audio] + simple ASD question
    - Assistant: "SPEAKING" or "NOT_SPEAKING" (majority label)
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

        # Build conversation — simple prompt that works (see experiment_logs/prompt_investigation.md)
        user_content = []

        # Add all face crop images (10 frames)
        for img_path in entry["image_paths"]:
            user_content.append({
                "type": "image",
                "image": img_path,
            })

        # Add audio
        user_content.append({
            "type": "audio",
            "audio": entry["audio_path"],
        })

        # Prompt that hints the audio may not belong to this person
        user_content.append({
            "type": "text",
            "text": (
                "These are 10 frames of a person's face with audio from the scene. "
                "The audio may or may not belong to this person — someone else in the "
                "scene could be the one speaking. Is this person speaking? "
                "Answer with only SPEAKING or NOT_SPEAKING."
            ),
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
            {
                "role": "assistant",
                "content": [{"type": "text", "text": entry["majority_label"]}],
            },
        ]

        return {
            "conversation": conversation,
            "label": entry["majority_label"],
            "is_not_speaking": 1 if entry["majority_label"] == "NOT_SPEAKING" else 0,
            "labels": entry["labels"],
            "majority_label": entry["majority_label"],
            "entity_id": entry["entity_id"],
        }


def collate_fn(batch, processor, speaking_token_id, not_token_id):
    """
    Collate function for classification-based training.

    The loss is computed on the logit at the last position, comparing
    the probability of SPEAKING vs NOT_SPEAKING token IDs.
    """
    from qwen_omni_utils import process_mm_info

    all_input_ids = []
    all_attention_masks = []
    all_class_labels = []  # 1 = SPEAKING, 0 = NOT_SPEAKING

    for sample in batch:
        # Build conversation WITHOUT the assistant answer
        conversation = sample["conversation"][:-1]  # remove assistant message

        text = processor.apply_chat_template(
            conversation,
            add_generation_prompt=True,
            tokenize=False,
        )

        audios, images, videos = process_mm_info(conversation, use_audio_in_video=False)

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

        all_input_ids.append(input_ids)
        all_attention_masks.append(attention_mask)
        all_class_labels.append(1 if sample["majority_label"] == "SPEAKING" else 0)

    # Pad sequences to same length
    max_len = max(ids.size(0) for ids in all_input_ids)
    pad_token_id = processor.tokenizer.pad_token_id or 0

    padded_input_ids = []
    padded_attention_masks = []

    for input_ids, attn_mask in zip(all_input_ids, all_attention_masks):
        pad_len = max_len - input_ids.size(0)
        if pad_len > 0:
            input_ids = torch.cat([input_ids, torch.full((pad_len,), pad_token_id)])
            attn_mask = torch.cat([attn_mask, torch.zeros(pad_len, dtype=attn_mask.dtype)])
        padded_input_ids.append(input_ids)
        padded_attention_masks.append(attn_mask)

    return {
        "input_ids": torch.stack(padded_input_ids),
        "attention_mask": torch.stack(padded_attention_masks),
        "class_labels": torch.tensor(all_class_labels, dtype=torch.long),
    }


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def run_timing_test(model, dataloader, device, num_steps, speaking_token_id, not_token_id, dtype, label_smoothing):
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

        with torch.amp.autocast("cuda", dtype=dtype):
            outputs = model.thinker(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
            )
            logits = outputs.logits
            seq_lens = batch["attention_mask"].sum(dim=1) - 1
            last_logits = logits[torch.arange(logits.size(0)), seq_lens]
            class_logits = torch.stack([last_logits[:, not_token_id], last_logits[:, speaking_token_id]], dim=1)
            loss = torch.nn.functional.cross_entropy(
                class_logits, batch["class_labels"], label_smoothing=label_smoothing
            )

        loss.backward()
        model.zero_grad()

        if (i + 1) % 10 == 0:
            elapsed = time.time() - start_time
            print(f"  Step {i+1}/{num_steps}: {elapsed:.1f}s elapsed, "
                  f"~{elapsed/(i+1):.2f}s/step")

    elapsed = time.time() - start_time
    per_step = elapsed / min(num_steps, len(dataloader))

    return per_step


def setup_audio_encoder_lora(model, args):
    """
    Apply LoRA to the last N layers of the audio encoder.

    The audio encoder (model.thinker.audio_tower) has 32 transformer layers
    with standard q_proj/k_proj/v_proj attention projections.
    We apply a small-rank LoRA to the last few layers so the encoder
    can learn to extract ASD-relevant audio features while preserving
    pretrained representations in early layers.
    """
    if args.unfreeze_audio_layers <= 0:
        print("Audio encoder LoRA: disabled")
        return

    total_audio_layers = 32
    start_layer = total_audio_layers - args.unfreeze_audio_layers

    # Build target module list for last N layers
    audio_target_modules = []
    for i in range(start_layer, total_audio_layers):
        audio_target_modules.extend([
            f"layers.{i}.self_attn.q_proj",
            f"layers.{i}.self_attn.k_proj",
            f"layers.{i}.self_attn.v_proj",
        ])

    audio_lora_config = LoraConfig(
        r=4,
        lora_alpha=8,
        lora_dropout=0.05,
        target_modules=audio_target_modules,
        bias="none",
    )

    print(f"\nApplying LoRA to audio encoder (last {args.unfreeze_audio_layers} of {total_audio_layers} layers)...")
    print(f"  Audio LoRA targets: layers {start_layer}-{total_audio_layers - 1}, q/k/v_proj")
    print(f"  Audio LoRA rank: 4, alpha: 8")

    model.thinker.audio_tower = get_peft_model(model.thinker.audio_tower, audio_lora_config)
    model.thinker.audio_tower.print_trainable_parameters()


def build_optimizer(model, args):
    """
    Build optimizer with separate parameter groups for thinker and audio encoder.

    The audio encoder uses a lower learning rate (default 10x lower) since
    we want to make small adjustments to pretrained audio features, not
    overwrite them.
    """
    thinker_params = []
    audio_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "audio_tower" in name:
            audio_params.append(param)
        else:
            thinker_params.append(param)

    param_groups = [
        {"params": thinker_params, "lr": args.learning_rate},
    ]

    if audio_params:
        audio_lr = args.learning_rate * args.audio_lr_scale
        param_groups.append({"params": audio_params, "lr": audio_lr})
        print(f"\nOptimizer parameter groups:")
        print(f"  Thinker: {len(thinker_params)} params, lr={args.learning_rate}")
        print(f"  Audio encoder: {len(audio_params)} params, lr={audio_lr}")
    else:
        print(f"\nOptimizer: {len(thinker_params)} trainable params, lr={args.learning_rate}")

    optimizer = torch.optim.AdamW(
        param_groups,
        weight_decay=args.weight_decay,
    )
    return optimizer


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

    dtype = torch.float16 if args.fp16 else torch.bfloat16

    model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
        args.model_name,
        torch_dtype=dtype,
        device_map="auto",
    )
    processor = Qwen2_5OmniProcessor.from_pretrained(args.model_name)

    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token

    print(f"Model loaded. Total parameters: {sum(p.numel() for p in model.parameters()):,}")

    # -----------------------------------------------------------------------
    # Apply LoRA to the thinker component (text LLM)
    # -----------------------------------------------------------------------
    print("\nApplying thinker LoRA configuration...")

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",  # attention
            "gate_proj", "up_proj", "down_proj",       # feed-forward
        ],
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    if not hasattr(model.thinker.config, "vocab_size"):
        vocab_size = model.thinker.lm_head.out_features
        model.thinker.config.vocab_size = vocab_size
        print(f"Set thinker vocab_size = {vocab_size}")

    model.thinker = get_peft_model(model.thinker, lora_config)
    print("Thinker LoRA:")
    model.thinker.print_trainable_parameters()

    # -----------------------------------------------------------------------
    # Apply LoRA to audio encoder last layers
    # -----------------------------------------------------------------------
    setup_audio_encoder_lora(model, args)

    if args.gradient_checkpointing:
        model.thinker.enable_input_require_grads()
        model.thinker.gradient_checkpointing_enable()
        print("Gradient checkpointing enabled")

    # -----------------------------------------------------------------------
    # Load data
    # -----------------------------------------------------------------------
    print("\nLoading dataset...")
    train_dataset = ASDDataset(args.data_dir, split="train")

    speaking_token_id = processor.tokenizer.encode("SPEAKING", add_special_tokens=False)[0]
    not_token_id = processor.tokenizer.encode("NOT", add_special_tokens=False)[0]
    print(f"Token IDs — SPEAKING: {speaking_token_id}, NOT: {not_token_id}")

    def collate_fn_bound(batch):
        return collate_fn(batch, processor, speaking_token_id, not_token_id)

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
            args.timing_test_steps, speaking_token_id, not_token_id, dtype,
            args.label_smoothing,
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
    optimizer = build_optimizer(model, args)

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
    print("Starting training (v2)")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Gradient accumulation: {args.gradient_accumulation_steps}")
    print(f"  Effective batch size: {args.batch_size * args.gradient_accumulation_steps}")
    print(f"  Learning rate: {args.learning_rate}")
    print(f"  Audio encoder LR: {args.learning_rate * args.audio_lr_scale}")
    print(f"  Label smoothing: {args.label_smoothing}")
    print(f"  LoRA rank: {args.lora_r}, alpha: {args.lora_alpha}, dropout: {args.lora_dropout}")
    print(f"  Audio encoder LoRA layers: {args.unfreeze_audio_layers}")
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
            batch = {k: v.to(device) for k, v in batch.items()}

            with torch.amp.autocast("cuda", dtype=dtype):
                outputs = model.thinker(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                )

                logits = outputs.logits
                seq_lens = batch["attention_mask"].sum(dim=1) - 1
                last_logits = logits[torch.arange(logits.size(0)), seq_lens]

                speaking_logit = last_logits[:, speaking_token_id]
                not_speaking_logit = last_logits[:, not_token_id]

                class_logits = torch.stack([not_speaking_logit, speaking_logit], dim=1)
                class_labels = batch["class_labels"]

                loss = torch.nn.functional.cross_entropy(
                    class_logits, class_labels, label_smoothing=args.label_smoothing
                )
                loss = loss / args.gradient_accumulation_steps

            loss.backward()

            total_loss += loss.item()
            epoch_loss += loss.item() * args.gradient_accumulation_steps
            num_batches += 1

            if (step + 1) % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad],
                    args.max_grad_norm,
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

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

    final_dir = output_dir / "final_model"
    model.thinker.save_pretrained(str(final_dir))
    processor.save_pretrained(str(final_dir))
    print(f"Final model saved to {final_dir}")

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
