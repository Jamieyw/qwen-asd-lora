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
- LoRA adapters: ~3,000,000 parameters (trained) — only **0.1%** of the total
- GPU memory: drops from ~48GB to ~6-8GB for trainable params
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
- `r` = rank (we use 8)

So instead of training a 4096x4096 matrix (16.7M params), we train:
- A: 8 x 4096 = 32,768 params
- B: 4096 x 8 = 32,768 params
- Total: 65,536 params (250x fewer!)

### Why "Low Rank"?

The matrix `B * A` has shape (4096, 4096) — same as `W` — but because it's the product of two thin matrices (through a bottleneck of size `r=8`), it can only represent changes in an 8-dimensional subspace. Research shows this is enough for task-specific adaptation.

### Which Layers Get LoRA?

Not every layer in the model needs LoRA. We apply it to the **attention layers** — specifically:

| Module | What It Does |
|--------|-------------|
| `q_proj` | **Query** projection — "what am I looking for?" |
| `k_proj` | **Key** projection — "what information do I have?" |
| `v_proj` | **Value** projection — "what's the actual content?" |
| `o_proj` | **Output** projection — combines attention results |

These are part of the **self-attention mechanism**, which is how the model decides which parts of the input (which face, which audio segment) to focus on. This is crucial for ASD because the model needs to learn to attend to lip movements synchronized with audio.

---

## 6. Our Specific Setup

### Model Architecture

```
Qwen2.5-Omni-3B
├── Vision Encoder    — processes face crop images (JPG)
│   └── Converts images to numerical representations (embeddings)
├── Audio Encoder     — processes audio segments (WAV)
│   └── Converts audio waveforms to embeddings
├── TMRoPE           — time-aligns visual and audio embeddings
│   └── Ensures frame timestamps match audio timestamps
├── Transformer Layers (x N)  — the main "brain"
│   ├── Self-Attention (q_proj, k_proj, v_proj, o_proj) ← LoRA goes here
│   └── Feed-Forward Network
└── Output Head       — produces final prediction
```

### LoRA Configuration

```python
LoraConfig(
    r=8,                    # rank — size of the bottleneck (see section 5)
    lora_alpha=16,          # scaling factor (explained below)
    lora_dropout=0.05,      # randomly zeroes 5% of LoRA outputs during training
    target_modules=[        # which weight matrices to adapt
        "q_proj", "k_proj", "v_proj", "o_proj"
    ],
    task_type="CAUSAL_LM"   # type of task
)
```

### Data Flow (What Happens to One Training Sample)

```
Input:
  ├── Face crops: [img_t1.jpg, img_t2.jpg, ..., img_t25.jpg]  (25 frames = 1 second)
  └── Audio: entity_audio.wav (corresponding 1-second audio clip)
      
      ↓ Vision Encoder converts each face to a 1024-dim vector
      ↓ Audio Encoder converts audio to sequence of 1024-dim vectors
      ↓ TMRoPE adds positional + temporal information
      ↓ Transformer layers process everything together
      ↓   (LoRA adapters modify attention behavior here)
      ↓ Output head predicts: "SPEAKING" or "NOT_SPEAKING"
      
      ↓ Compare with true label
      ↓ Compute loss
      ↓ Update ONLY LoRA parameters (A and B matrices)
```

---

## 7. The Training Process Step by Step

### Step 1: Load the Pre-trained Model

```python
model = Qwen2_5OmniModel.from_pretrained("Qwen/Qwen2.5-Omni-3B")
```

This downloads ~6GB of weights and loads them into GPU memory. All 3B parameters are loaded but marked as **frozen** (no gradients computed).

### Step 2: Attach LoRA Adapters

```python
model = get_peft_model(model, lora_config)
```

This adds small A and B matrices alongside every q/k/v/o_proj layer. Only these new matrices are marked as **trainable**.

The model now has two types of parameters:
- Frozen: 3,000,000,000 (original, not updated)
- Trainable: ~3,000,000 (LoRA, updated every step)

### Step 3: Prepare a Batch

A **batch** is a small group of training samples processed together. We use batch_size=2 (limited by GPU memory since multimodal data is large).

```
Batch = [
    (images_1, audio_1, label_1),   # Sample 1: face crops + audio -> SPEAKING
    (images_2, audio_2, label_2),   # Sample 2: face crops + audio -> NOT_SPEAKING
]
```

### Step 4: Forward Pass

The model processes the batch:
1. Images go through the vision encoder -> visual embeddings
2. Audio goes through the audio encoder -> audio embeddings
3. Both are concatenated and passed through transformer layers
4. Each transformer layer applies self-attention:
   - Original: `attention = softmax(Q * K^T) * V` (using frozen W_q, W_k, W_v)
   - With LoRA: `attention = softmax((Q + delta_Q) * (K + delta_K)^T) * (V + delta_V)`
   - Where `delta_Q = B_q * A_q * x` (the LoRA modification)
5. The model outputs a probability distribution over tokens
6. We extract the prediction: "SPEAKING" or "NOT_SPEAKING"

### Step 5: Compute Loss

**Loss** = how wrong the model was. We use **cross-entropy loss**:

```
loss = -log(probability the model assigned to the correct answer)
```

- If the correct answer is "SPEAKING" and the model said 90% chance -> loss = -log(0.9) = 0.105 (low, good)
- If the correct answer is "SPEAKING" and the model said 10% chance -> loss = -log(0.1) = 2.303 (high, bad)

### Step 6: Backward Pass (Backpropagation)

The loss flows backward through the network. For each LoRA parameter, we compute its **gradient** — how much and in which direction to adjust it to reduce the loss.

Key: gradients are ONLY computed for LoRA parameters (the frozen original parameters are skipped), which saves massive amounts of computation.

### Step 7: Optimizer Step

The **optimizer** (AdamW) uses the gradients to update LoRA parameters:

```
new_param = old_param - learning_rate * gradient
```

AdamW is smarter than this simple formula — it keeps running averages of gradients and their squares to make more informed updates. But the basic idea is: nudge each parameter in the direction that reduces loss.

### Step 8: Gradient Accumulation

Because our batch size is only 2 (GPU memory limit), we use **gradient accumulation = 4**:

- Process 4 batches of 2 samples each
- Accumulate (add up) the gradients from all 4 batches
- Only then do one optimizer update

This effectively gives us a batch size of 2 x 4 = 8, which produces more stable training, without needing more GPU memory.

### Step 9: Repeat

One **epoch** = processing every sample in the training set once. We train for 3 epochs, meaning each sample is seen 3 times. The model improves gradually:

```
Epoch 1: loss ~2.0 → ~0.8  (learning the basics)
Epoch 2: loss ~0.8 → ~0.5  (refining)
Epoch 3: loss ~0.5 → ~0.3  (fine-tuning details)
```

(Actual numbers will vary)

---

## 8. Key Hyperparameters Explained

### LoRA Hyperparameters

| Parameter | Value | What It Controls |
|-----------|-------|-----------------|
| `r` (rank) | 8 | Bottleneck size. Higher = more capacity but more memory. 4-16 is typical. |
| `lora_alpha` | 16 | Scaling factor. The LoRA output is multiplied by `alpha/r = 16/8 = 2`. Controls how much influence the LoRA adapters have vs. the original weights. |
| `lora_dropout` | 0.05 | During training, randomly zeros 5% of LoRA outputs. Prevents **overfitting** (memorizing training data instead of learning patterns). |

### Training Hyperparameters

| Parameter | Value | What It Controls |
|-----------|-------|-----------------|
| `learning_rate` | 2e-4 (0.0002) | How big each parameter update step is. Too high = unstable, too low = too slow. |
| `batch_size` | 2 | Samples processed at once. Limited by GPU memory. |
| `gradient_accumulation_steps` | 4 | How many batches to accumulate before updating. Effective batch = 2 x 4 = 8. |
| `num_epochs` | 3 | How many times to iterate over the full training set. |
| `warmup_steps` | 100 | For the first 100 steps, learning rate gradually increases from 0 to 2e-4. Prevents early instability. |
| `weight_decay` | 0.01 | Slightly penalizes large parameter values. Regularization to prevent overfitting. |
| `bf16` | True | Use bfloat16 precision (16-bit instead of 32-bit). Halves memory usage with minimal accuracy loss. V100 may use fp16 instead. |

---

## 9. What Happens on the GPU

### Memory Breakdown (V100, 32GB)

```
Model weights (frozen, bf16):     ~6 GB
LoRA parameters:                  ~12 MB  (tiny!)
Optimizer states (for LoRA only): ~24 MB
Gradients (for LoRA only):        ~12 MB
Activations (intermediate values): ~10-15 GB  (depends on sequence length)
Input data (images + audio batch): ~2-4 GB
CUDA overhead:                     ~2 GB
─────────────────────────────────────────
Total:                             ~20-27 GB  (fits in 32GB!)
```

### Why Activations Are So Large

During the forward pass, the model saves intermediate results at every layer (called "activations"). These are needed for the backward pass to compute gradients. With multimodal input (25 images + audio), these activations are substantial.

If memory is tight, **gradient checkpointing** can be enabled: it discards activations during forward pass and recomputes them during backward pass. Trades computation time for memory.

---

## 10. After Training

### What Gets Saved

Only the LoRA adapter weights are saved (~10-50MB), not the full model. The saved files:

```
output/
├── adapter_model.safetensors    # LoRA A and B matrices
├── adapter_config.json          # LoRA configuration
└── training_args.json           # hyperparameters used
```

### How to Use the Fine-Tuned Model

```python
# Load original model
model = Qwen2_5OmniModel.from_pretrained("Qwen/Qwen2.5-Omni-3B")
# Load LoRA adapter on top
model = PeftModel.from_pretrained(model, "output/")
# Now model has the original weights + your LoRA adjustments
```

### Evaluation Metrics

- **Accuracy:** % of correct predictions (SPEAKING vs NOT_SPEAKING)
- **Precision:** Of all predicted "SPEAKING", how many were actually speaking?
- **Recall:** Of all actually speaking, how many did we catch?
- **F1 Score:** Harmonic mean of precision and recall (balances both)
- **Confusion Matrix:** 2x2 grid showing true positives, false positives, etc.

---

## 11. Common Issues and Solutions

| Issue | Symptom | Solution |
|-------|---------|----------|
| Out of memory (OOM) | CUDA out of memory error | Reduce batch_size to 1, enable gradient checkpointing, reduce max image resolution |
| Loss not decreasing | Loss stays flat or oscillates | Lower learning rate (try 5e-5), check data formatting, ensure labels are correct |
| Loss goes to NaN | Loss becomes NaN or inf | Lower learning rate, check for data corruption, switch to fp32 |
| Overfitting | Training loss drops but validation loss increases | Increase dropout, reduce epochs, add more training data |
| Slow training | Each step takes very long | Reduce number of frames per sample, lower image resolution, check GPU utilization with `nvidia-smi` |
| Model predicts same class | Always says SPEAKING or NOT_SPEAKING | Check label balance in dataset, adjust class weights in loss function |

---

## Glossary

| Term | Definition |
|------|-----------|
| **Backpropagation** | Algorithm that computes how each parameter contributed to the error, flowing backward through the network |
| **Batch** | A group of training samples processed together in one step |
| **Embedding** | A numerical representation (vector) of input data that the model can work with |
| **Epoch** | One complete pass through the entire training dataset |
| **Gradient** | The direction and magnitude to adjust a parameter to reduce loss |
| **Loss** | A number measuring how wrong the model's prediction was |
| **Optimizer** | Algorithm that uses gradients to update parameters (AdamW in our case) |
| **Overfitting** | When the model memorizes training data instead of learning general patterns |
| **Parameter** | A learnable number (weight) in the model |
| **Self-attention** | Mechanism that lets the model decide which parts of the input to focus on |
| **Tensor** | A multi-dimensional array of numbers (generalization of matrices) |
| **Token** | A unit of text/input that the model processes (a word, subword, or special symbol) |
| **Transformer** | The neural network architecture used by Qwen and most modern LLMs |
