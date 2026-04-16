# LoRA Fine-Tuning: A Complete Guide

A step-by-step explanation of LoRA fine-tuning.

---

## Table of Contents

1. [What Is a Language Model?](#1-what-is-a-language-model)
2. [What Is Fine-Tuning?](#2-what-is-fine-tuning)
3. [The Problem: Fine-Tuning Is Expensive](#3-the-problem-fine-tuning-is-expensive)
4. [What Is LoRA?](#4-what-is-lora)
5. [How LoRA Works (Technical Details)](#5-how-lora-works-technical-details)
6. [Our Specific Setup](#6-our-specific-setup)
7. [The Training Process Step by Step](#7-the-training-process-step-by-step)
8. [Key Hyperparameters Explained](#8-key-hyperparameters-explained)
9. [What Happens on the GPU](#9-what-happens-on-the-gpu)
10. [After Training](#10-after-training)
11. [Common Issues and Solutions](#11-common-issues-and-solutions)

---

## 1. What Is a Language Model?

A language model (LM) is a program that has learned patterns from massive amounts of data. Think of it as a very sophisticated pattern-matching machine.

**Qwen2.5-Omni-3B** is our model. Let's break down the name:
- **Qwen** — the model family, made by Alibaba
- **2.5** — version 2.5
- **Omni** — it can understand multiple types of input: text, images, audio, and video
- **3B** — it has 3 billion **parameters**

### What are parameters?

Parameters are numbers (called "weights") that the model has learned during its original training. They are stored in matrices (2D grids of numbers). Every decision the model makes flows through these matrices.

A 3B model has 3,000,000,000 individual numbers that collectively encode everything it knows. These are organized into layers, and each layer contains weight matrices.

---

## 2. What Is Fine-Tuning?

The model was originally trained on general internet data — it knows a lot about everything, but nothing specific about Active Speaker Detection (ASD).

**Fine-tuning** means: take the pre-trained model and continue training it on our specific dataset (face images + audio -> speaking or not speaking), so it gets good at our task.

Analogy: The model already graduated from "general university." Fine-tuning is like sending it to a specialized certification program.

### How training works (simplified):

1. Show the model an input (face images + audio)
2. The model makes a prediction ("SPEAKING" or "NOT_SPEAKING")
3. Compare prediction to the true answer (the label)
4. Calculate how wrong it was (the **loss**)
5. Adjust parameters slightly to be less wrong next time (**backpropagation**)
6. Repeat millions of times

Each full pass through the entire dataset is called an **epoch**.

---

## 3. The Problem: Fine-Tuning Is Expensive

With 3 billion parameters, updating ALL of them requires:

- **Memory:** Each parameter needs ~16 bytes during training (the parameter itself + gradient + optimizer state). That's 3B x 16 = ~48GB of GPU memory just for parameters.
- **Time:** Computing gradients for 3 billion parameters is slow.
- **Storage:** Saving a full copy of 3B modified parameters = ~6GB per checkpoint.

Our V100 GPU has 32GB of memory. We literally cannot fit full fine-tuning in memory.

---

## 4. What Is LoRA?

**LoRA** = **Lo**w-**R**ank **A**daptation

The core insight: when you fine-tune a model for a specific task, you don't need to change all 3 billion parameters. The changes are actually "low-rank" — meaning the important changes can be compressed into much smaller matrices.

**LoRA freezes the original model weights** (they don't change at all) and adds tiny trainable matrices alongside the existing ones. Only these small matrices get updated during training.

Result:
- Original model: 3,000,000,000 parameters (frozen, not trained)
- LoRA adapters: ~10,000,000 parameters (trained) — only **~0.3%** of the total
- GPU memory: drops from ~48GB to ~20-27GB
- Training speed: much faster
- Storage: LoRA adapter saves are ~10-50MB instead of ~6GB

---

## 5. How LoRA Works (Technical Details)

### The Math

In a neural network, key operations happen in **linear layers** (matrix multiplications). A linear layer takes input `x` and computes:

```
output = W * x
```

Where `W` is a weight matrix of shape `(d_out, d_in)`. For example, a matrix might be `(4096, 4096)` — that's 16 million numbers in just one layer.

### LoRA's Trick: Low-Rank Decomposition

Instead of modifying `W` directly, LoRA adds a **parallel path**:

```
output = W * x + (B * A) * x
```

Where:
- `W` = original weight matrix (4096 x 4096) — **frozen**
- `A` = small matrix (r x 4096) — **trainable**
- `B` = small matrix (4096 x r) — **trainable**
- `r` = rank (we use 16 for the thinker, 4 for the vision encoder)

So instead of training a 4096x4096 matrix (16.7M params), we train:
- A: 16 x 4096 = 65,536 params
- B: 4096 x 16 = 65,536 params
- Total: 131,072 params (128x fewer!)

### Why "Low Rank"?

The matrix `B * A` has shape (4096, 4096) — same as `W` — but because it's the product of two thin matrices (through a bottleneck of size `r=16`), it can only represent changes in a 16-dimensional subspace. Research shows this is enough for task-specific adaptation.

### Which Layers Get LoRA?

We apply LoRA to two components of the model:

**Thinker (text LLM) — attention + feed-forward:**

| Module | What It Does |
|--------|-------------|
| `q_proj` | **Query** projection — "what am I looking for?" |
| `k_proj` | **Key** projection — "what information do I have?" |
| `v_proj` | **Value** projection — "what's the actual content?" |
| `o_proj` | **Output** projection — combines attention results |
| `gate_proj` | Feed-forward gate (SwiGLU activation) |
| `up_proj` | Feed-forward up projection |
| `down_proj` | Feed-forward down projection |

Including feed-forward layers gives the model more capacity to learn task-specific transformations beyond just attention patterns.

**Vision encoder (last 8 of 32 layers) — attention only:**

| Module | What It Does |
|--------|-------------|
| `q`, `k`, `v` | Attention projections in the visual transformer |

The vision encoder needs to learn that **lip state** (open vs closed, movement across frames) is the key signal for ASD — not just whether a face is present. Applying LoRA to only the last 8 layers preserves early-layer visual features (edges, textures) while fine-tuning high-level representations.

---

## 6. Our Specific Setup

### Model Architecture

```
Qwen2.5-Omni-3B
├── Vision Encoder    — processes face crop images (JPG)
│   └── 32 transformer layers; LoRA on last 8 (layers 24–31)
├── Audio Encoder     — processes audio segments (WAV)
│   └── Converts audio waveforms to embeddings (frozen)
├── TMRoPE           — time-aligns visual and audio embeddings
│   └── Ensures frame timestamps match audio timestamps
├── Thinker (LLM)    — the main "brain"
│   ├── Self-Attention (q/k/v/o_proj) ← LoRA here (r=16)
│   └── Feed-Forward (gate/up/down_proj) ← LoRA here (r=16)
└── Output Head       — produces final prediction
```

### Thinker LoRA Configuration

```python
LoraConfig(
    r=16,                   # rank — size of the bottleneck (doubled from v1)
    lora_alpha=32,          # scaling factor (alpha/r = 2, same ratio as v1)
    lora_dropout=0.1,       # randomly zeroes 10% of LoRA outputs during training
    target_modules=[        # attention + feed-forward layers
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    task_type="CAUSAL_LM"
)
```

### Vision Encoder LoRA Configuration

```python
LoraConfig(
    r=4,                    # small rank — preserve pretrained visual features
    lora_alpha=8,
    lora_dropout=0.05,
    target_modules=[        # layer-specific attention modules (last 8 layers)
        "blocks.24.attn.q", "blocks.24.attn.k", "blocks.24.attn.v",
        # ... through blocks.31
    ],
)
```

### Label Smoothing

Instead of hard labels (1 for SPEAKING, 0 for NOT_SPEAKING), we use **label smoothing = 0.1**:
- SPEAKING → target probability = 0.9
- NOT_SPEAKING → target probability = 0.1

This prevents the model from driving its output logits to extreme values, acts as regularization, and improves calibration. It is especially useful for ASD because the boundary between speaking and not-speaking is sometimes genuinely ambiguous (e.g. trailing off, background noise).

### Data Flow (What Happens to One Training Sample)

```
Input:
  ├── Face crops: [img_t1.jpg, ..., img_t10.jpg]  (10 frames)
  └── Audio: entity_audio.wav

      ↓ Vision Encoder (last 8 layers have LoRA)
      ↓   → visual embeddings (lip state, face features)
      ↓ Audio Encoder (frozen)
      ↓   → audio embeddings
      ↓ TMRoPE adds positional + temporal information
      ↓ Thinker transformer layers (LoRA on attention + FFN)
      ↓   → processes all modalities together
      ↓ Logits at last position
      ↓   → extract SPEAKING logit vs NOT_SPEAKING logit
      ↓ Binary cross-entropy loss (with label_smoothing=0.1)
      ↓ Update ONLY LoRA parameters (vision + thinker)
```

---

## 7. The Training Process Step by Step

### Step 1: Load the Pre-trained Model

```python
model = Qwen2_5OmniForConditionalGeneration.from_pretrained("Qwen/Qwen2.5-Omni-3B")
```

Downloads ~6GB of weights into GPU memory. All 3B parameters are frozen.

### Step 2: Attach LoRA Adapters

```python
model.thinker = get_peft_model(model.thinker, lora_config)
model.thinker.visual = get_peft_model(model.thinker.visual, vision_lora_config)
```

Adds A and B matrices alongside targeted layers in both the thinker and vision encoder. Only these new matrices are trainable.

### Step 3: Prepare a Batch

```
Batch = [
    (images_1, audio_1, label_1),   # SPEAKING
    (images_2, audio_2, label_2),   # NOT_SPEAKING
]
```

### Step 4: Forward Pass

The model processes the batch:
1. Images go through the vision encoder (last 8 layers modified by LoRA)
2. Audio goes through the audio encoder (frozen)
3. Both pass through the thinker transformer layers (LoRA on attention + FFN)
4. We extract the logit at the last sequence position for `SPEAKING` and `NOT_SPEAKING` tokens
5. Binary cross-entropy loss is computed between these two logits and the ground truth label

### Step 5: Compute Loss (with Label Smoothing)

```
loss = cross_entropy(class_logits, labels, label_smoothing=0.1)
```

Label smoothing modifies the target distribution:
- Hard label: [0, 1] (NOT_SPEAKING=0, SPEAKING=1)
- Smoothed label: [0.05, 0.95] (the smoothing mass is spread uniformly)

### Step 6: Backward Pass

Loss flows back through the network. Gradients are computed only for LoRA parameters (vision encoder + thinker). The 3B frozen parameters are skipped entirely.

### Step 7: Separate Optimizer Step

Two parameter groups with different learning rates:
- **Thinker LoRA:** lr = 5e-5
- **Vision encoder LoRA:** lr = 5e-5 × 0.2 = 1e-5

The vision encoder gets a lower LR because we want small adjustments to pretrained visual features, not a full overwrite.

### Step 8: Gradient Accumulation

With batch_size=1, we accumulate gradients over 8 steps before updating, giving an effective batch size of 8.

### Step 9: Repeat

8 epochs × ~2000 samples = ~16,000 forward passes. The model learns gradually; both precision and recall improve as it learns to detect lip movement patterns.

---

## 8. Key Hyperparameters Explained

### LoRA Hyperparameters (Thinker)

| Parameter | Value | What It Controls |
|-----------|-------|-----------------|
| `r` (rank) | 16 | Bottleneck size. Higher = more capacity. Doubled from v1 to handle feed-forward targets. |
| `lora_alpha` | 32 | Scaling factor. LoRA output multiplied by `alpha/r = 2`. Same ratio as v1. |
| `lora_dropout` | 0.1 | Zeroes 10% of LoRA outputs during training. Increased from 0.05 to compensate for larger adapter. |

### LoRA Hyperparameters (Vision Encoder)

| Parameter | Value | What It Controls |
|-----------|-------|-----------------|
| `r` (rank) | 4 | Small rank to preserve pretrained visual features. |
| `lora_alpha` | 8 | Scaling factor (alpha/r = 2). |
| `lora_dropout` | 0.05 | Light regularization for visual adapter. |

### Training Hyperparameters

| Parameter | Value | What It Controls |
|-----------|-------|-----------------|
| `learning_rate` | 5e-5 | Thinker LR. Lowered from 1e-4 for more stable convergence. |
| `vision_lr_scale` | 0.2 | Vision encoder LR = 5e-5 × 0.2 = 1e-5. Protects pretrained visual features. |
| `label_smoothing` | 0.1 | Soft labels instead of hard 0/1. Reduces overconfidence. |
| `batch_size` | 1 | Limited by GPU memory (multimodal data is large). |
| `gradient_accumulation_steps` | 8 | Effective batch = 1 × 8 = 8. |
| `epochs` | 8 | More epochs than v1 to accommodate the lower LR. |
| `warmup_steps` | 100 | LR ramps from 0 to 5e-5 over 100 steps. |
| `weight_decay` | 0.01 | Regularization. |

---

## 9. What Happens on the GPU

### Memory Breakdown (V100, 32GB)

```
Model weights (frozen, bf16):          ~6 GB
Thinker LoRA parameters:               ~25 MB
Vision encoder LoRA parameters:        ~2 MB
Optimizer states (LoRA only):          ~50 MB
Gradients (LoRA only):                 ~25 MB
Activations (intermediate values):     ~10-15 GB
Input data (images + audio batch):     ~2-4 GB
CUDA overhead:                         ~2 GB
─────────────────────────────────────────────
Total:                                 ~20-27 GB  (fits in 32GB)
```

If memory is tight, enable `--gradient_checkpointing`: discards activations during the forward pass and recomputes during backward. Trades ~30% more compute for ~40% less memory.

---

## 10. After Training

### What Gets Saved

Only the LoRA adapter weights are saved (~20-60MB total):

```
output/
├── best_model/
│   ├── adapter_model.safetensors    # thinker LoRA A and B matrices
│   ├── adapter_config.json          # LoRA configuration
│   └── ...                          # processor/tokenizer files
└── training_config.json             # all hyperparameters used
```

The vision encoder LoRA is saved as part of the thinker's PEFT model since `model.thinker.visual` is wrapped inside `model.thinker`.

### How to Use the Fine-Tuned Model

```python
# Load original model
model = Qwen2_5OmniForConditionalGeneration.from_pretrained("Qwen/Qwen2.5-Omni-3B")
# Load thinker LoRA adapter (includes vision encoder LoRA)
model.thinker = PeftModel.from_pretrained(model.thinker, "output/best_model")
```

### Evaluation Metrics

- **Accuracy:** % of correct predictions
- **Precision:** Of all predicted SPEAKING, how many were actually speaking?
- **Recall:** Of all actually speaking, how many were caught?
- **F1 Score:** Harmonic mean of precision and recall
- **mAP:** Area under the Precision-Recall curve
- **Confusion Matrix:** 2×2 grid of TP, FP, TN, FN

---

## 11. Common Issues and Solutions

| Issue | Symptom | Solution |
|-------|---------|----------|
| Out of memory (OOM) | CUDA out of memory error | Enable `--gradient_checkpointing`, reduce `--unfreeze_vision_layers` |
| Loss not decreasing | Loss stays flat | Lower LR, check that label smoothing isn't too high |
| Loss goes to NaN | Loss becomes NaN or inf | Lower LR, switch to fp32, reduce `--max_grad_norm` |
| Overfitting | Training loss drops but val loss rises | Increase dropout, reduce epochs |
| Slow training | Each step takes very long | Reduce `--unfreeze_vision_layers`, check GPU utilization |
| Model predicts same class | Always SPEAKING or NOT_SPEAKING | Check label balance, verify label smoothing is active |
| Vision LoRA not loading | Module name mismatch | Check exact layer names: `for n, _ in model.thinker.visual.named_modules(): print(n)` |

---

## Glossary

| Term | Definition |
|------|-----------|
| **Backpropagation** | Algorithm that computes how each parameter contributed to the error, flowing backward through the network |
| **Batch** | A group of training samples processed together in one step |
| **Embedding** | A numerical representation (vector) of input data that the model can work with |
| **Epoch** | One complete pass through the entire training dataset |
| **Gradient** | The direction and magnitude to adjust a parameter to reduce loss |
| **Label smoothing** | Replacing hard 0/1 targets with soft values (e.g. 0.1/0.9) to prevent overconfidence |
| **Loss** | A number measuring how wrong the model's prediction was |
| **Optimizer** | Algorithm that uses gradients to update parameters (AdamW in our case) |
| **Overfitting** | When the model memorizes training data instead of learning general patterns |
| **Parameter** | A learnable number (weight) in the model |
| **Self-attention** | Mechanism that lets the model decide which parts of the input to focus on |
| **Tensor** | A multi-dimensional array of numbers (generalization of matrices) |
| **Token** | A unit of text/input that the model processes |
| **Transformer** | The neural network architecture used by Qwen and most modern LLMs |
