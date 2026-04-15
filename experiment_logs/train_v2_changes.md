# Train V2: Label Smoothing + Audio Encoder LoRA + Hyperparameter Tuning

Date: 2026-04-15

## Motivation

The original train.py produced a suspiciously low training loss (0.0011 after epoch 1) and the fine-tuned model performed worse than the base model (46.8% vs 51.4% accuracy). Root cause analysis (see training_loss_analysis.md) identified that the LoRA was learning text-level shortcuts rather than attending to visual/audio content, because the vision and audio encoders were completely frozen.

## Changes in train_v2.py

### 1. Label Smoothing (--label_smoothing 0.1)

Instead of hard 0/1 targets, we use 0.1/0.9 soft targets via `cross_entropy(label_smoothing=0.1)`.

- Prevents the model from becoming 100% confident on every prediction
- Forces it to keep attending to inputs rather than collapsing to a shortcut
- Expected loss floor moves from ~0.001 to ~0.05-0.1
- Applied in both the training loop and the timing test

### 2. Audio Encoder LoRA (--unfreeze_audio_layers 8)

Applied a separate LoRA (rank 4, alpha 8) to the last 8 of 32 audio encoder layers (`model.thinker.audio_tower`).

- The audio encoder uses standard naming: `layers.N.self_attn.q_proj/k_proj/v_proj`
- Only the last 8 layers are unfrozen to preserve pretrained features in early layers
- Uses a separate optimizer parameter group with 10x lower LR (`--audio_lr_scale 0.1`)
- Rationale: the audio encoder is more important than vision for ASD because the key signal is whether audio matches lip movements — the encoder needs to learn ASD-relevant features

### 3. LoRA Hyperparameter Changes

| Parameter | train.py (v1) | train_v2.py | Rationale |
|-----------|--------------|-------------|-----------|
| lora_r | 8 | 16 | More capacity (ASR paper used 32) |
| lora_alpha | 16 | 32 | Keep alpha = 2*r ratio |
| learning_rate | 2e-4 | 5e-5 | Previous LR too aggressive, caused collapse |
| lora_dropout | 0.05 | 0.1 | More regularization for 2000 samples |
| warmup_steps | 100 | 50 | 100 was too many for ~250 optimizer steps |
| target_modules | q,k,v,o_proj | q,k,v,o_proj + gate,up,down_proj | Adding feed-forward layers (ASR paper showed QVKOFC >> QV) |

### 4. Separate Optimizer Parameter Groups

The optimizer now uses two parameter groups:
- Thinker (text LLM) params: lr = 5e-5
- Audio encoder params: lr = 5e-6 (10x lower)

This prevents the audio encoder from being overwritten too aggressively while still allowing it to adapt.

## What We Expect

- Training loss should start at ~0.5-0.7 (binary cross-entropy with label smoothing on a balanced dataset)
- Loss should gradually decrease to ~0.05-0.15 over 3 epochs
- The model should show genuine discrimination between SPEAKING and NOT_SPEAKING
- Evaluation accuracy should meaningfully exceed the 51.4% baseline

## Files

- `train_v2.py` — New training script
- `train_v2.sh` — SLURM submission script with all v2 flags

## How to Run

```bash
sbatch train_v2.sh
```

## Verification Needed

The audio encoder layer naming (`layers.N.self_attn.q_proj`) is based on the HuggingFace config. If the names differ on the actual model, run this on the cluster to check:

```python
python -c "
from transformers import Qwen2_5OmniForConditionalGeneration
m = Qwen2_5OmniForConditionalGeneration.from_pretrained('Qwen/Qwen2.5-Omni-3B')
for n, _ in m.thinker.audio_tower.named_modules():
    if 'proj' in n: print(n)
" 2>/dev/null | head -20
```

Adjust `setup_audio_encoder_lora()` in train_v2.py if the module names differ.
