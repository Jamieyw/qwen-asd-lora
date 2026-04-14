# train.py — Full Code Walkthrough

A line-by-line explanation of every step in the training script.

---

## Big Picture

The script does 7 things in order:

1. Parse command-line arguments
2. Load the pre-trained Qwen2.5-Omni-3B model
3. Attach LoRA adapters (the small trainable matrices)
4. Load the prepared dataset
5. Run a timing test to estimate total training time
6. Run the training loop (the core learning process)
7. Save the trained LoRA weights

---

## 1. Imports (Lines 13–30)

```python
import torch                    # PyTorch — the deep learning framework
from peft import LoraConfig, get_peft_model, TaskType  # LoRA library
from torch.utils.data import Dataset, DataLoader       # Data loading utilities
from transformers import (
    Qwen2_5OmniForConditionalGeneration,   # The model class
    Qwen2_5OmniProcessor,                   # Handles tokenization + image/audio processing
    get_linear_schedule_with_warmup,         # Learning rate scheduler
)
```

**Why these?**
- `torch` is what actually runs the math on the GPU
- `peft` (Parameter-Efficient Fine-Tuning) provides LoRA
- `transformers` is HuggingFace's library that has the Qwen model
- `Dataset`/`DataLoader` handle feeding data to the model in batches

---

## 2. Arguments (Lines 37–62)

```python
def parse_args():
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--gradient_accumulation_steps", type=int, default=8)
    p.add_argument("--learning_rate", type=float, default=2e-4)
    p.add_argument("--lora_r", type=int, default=8)
    # ... etc
```

These let you tweak training from the command line without editing code. The key ones:

| Argument | Default | What it controls |
|----------|---------|-----------------|
| `--batch_size` | 1 | Samples processed at once (1 because multimodal data is huge) |
| `--gradient_accumulation_steps` | 8 | Accumulate gradients over 8 mini-batches before updating weights. Effective batch = 1 x 8 = 8 |
| `--learning_rate` | 2e-4 | How big each weight update step is. Too high = unstable, too low = too slow |
| `--lora_r` | 8 | LoRA rank — the bottleneck size (see LORA_GUIDE.md) |
| `--fp16` | off | Use 16-bit floats (needed for V100, not needed for A100) |
| `--gradient_checkpointing` | off | Trade compute time for memory savings |

---

## 3. ASDDataset Class (Lines 69–151)

This is how the model sees each training sample.

### `__init__` — Load metadata

```python
def __init__(self, data_dir, split="train"):
    metadata_path = self.data_dir / "metadata.jsonl"
    with open(metadata_path) as f:
        for line in f:
            self.metadata.append(json.loads(line.strip()))
```

Reads the `metadata.jsonl` file that `prepare_data.py` created. Each line has paths to images, audio, and the label.

### `__getitem__` — Build one training conversation

```python
def __getitem__(self, idx):
    entry = self.metadata[idx]
```

When the DataLoader asks for sample #42, this function builds a **conversation** that Qwen2.5-Omni understands:

```python
conversation = [
    {
        "role": "system",
        "content": [{"type": "text", "text": "You are an active speaker detection system..."}]
    },
    {
        "role": "user",
        "content": [
            {"type": "image", "image": "path/to/face_frame000.jpg"},
            {"type": "image", "image": "path/to/face_frame001.jpg"},
            # ... up to 10 face images
            {"type": "audio", "audio": "path/to/audio.wav"},
            {"type": "text", "text": "Is this person currently speaking?"}
        ]
    },
    {
        "role": "assistant",
        "content": [{"type": "text", "text": "SPEAKING"}]  # ← the answer we want the model to learn
    }
]
```

**Why a conversation format?** Qwen2.5-Omni is a chat model. It was trained to follow conversation patterns. By formatting our task as "user asks question, assistant answers," we leverage what the model already knows about following instructions.

---

## 4. Collate Function (Lines 154–238)

This is the trickiest part. The DataLoader calls this to combine multiple samples into one batch.

### Step 4a: Turn conversation into tokens

```python
text = processor.apply_chat_template(conversation, add_generation_prompt=False, tokenize=False)
audios, images, videos = process_mm_info(conversation, use_audio_in_video=False)
inputs = processor(text=text, audio=audios, images=images, videos=videos, ...)
```

What happens here:
1. `apply_chat_template` converts the conversation into a text string with special tokens like `<|im_start|>system`, `<|im_start|>user`, etc.
2. `process_mm_info` extracts the actual image/audio data from the file paths
3. `processor()` tokenizes the text and converts images/audio into numerical tensors

The result is `input_ids` — a sequence of numbers where each number represents a token (word piece, image patch, or audio chunk).

### Step 4b: Create labels with masking

```python
labels = input_ids.clone()
labels[:-label_len - 1] = -100
```

This is crucial. The `labels` tensor tells the model **which tokens to learn to predict**.

- `-100` means "ignore this token, don't compute loss on it"
- Only the last few tokens (the answer "SPEAKING" or "NOT_SPEAKING") have real labels

**Why?** We don't want the model to learn to generate the system prompt or user question. We only want it to learn: given this input, the correct answer is "SPEAKING."

**Visual example:**

```
input_ids:  [system_tokens... user_tokens... image_tokens... audio_tokens... "SPEAKING" EOS]
labels:     [-100 -100 -100   -100 -100       -100            -100           "SPEAKING" EOS]
                                                                              ↑ only these count
```

### Step 4c: Pad to same length

```python
max_len = max(ids.size(0) for ids in all_input_ids)
# ... pad shorter sequences with zeros
```

Different samples have different lengths (different number of images, different audio lengths). GPUs need all inputs in a batch to be the same length, so we pad shorter ones with zeros. The `attention_mask` tells the model which positions are real (1) vs padding (0).

---

## 5. Load Model (Lines 282–312)

```python
dtype = torch.float16 if args.fp16 else torch.bfloat16

model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
    args.model_name,
    torch_dtype=dtype,
    device_map="auto",
)
processor = Qwen2_5OmniProcessor.from_pretrained(args.model_name)
```

- `from_pretrained` downloads the 3B parameter model from HuggingFace (or uses cache)
- `torch_dtype=dtype` loads weights in 16-bit precision (halves memory vs 32-bit)
- `device_map="auto"` automatically places the model on the GPU
- The processor handles converting text/images/audio into the format the model expects

---

## 6. Apply LoRA (Lines 314–335)

```python
lora_config = LoraConfig(
    r=8,                     # rank — bottleneck size of A and B matrices
    lora_alpha=16,           # scaling factor (output multiplied by alpha/r = 2)
    lora_dropout=0.05,       # randomly zero 5% of LoRA outputs during training
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],  # which layers
    bias="none",             # don't train bias terms
    task_type=TaskType.CAUSAL_LM,   # we're doing text generation
)

model.thinker = get_peft_model(model.thinker, lora_config)
```

**Why `model.thinker`?** Qwen2.5-Omni has two components:
- **Thinker** — processes inputs and generates text (this is what we fine-tune)
- **Talker** — generates speech audio output (we don't need this for ASD)

`get_peft_model` inserts small LoRA matrices (A and B) alongside every q/k/v/o_proj layer in the thinker. All original weights are frozen.

```python
model.thinker.print_trainable_parameters()
# Output: "trainable params: 3,145,728 || all params: 3,000,000,000 || trainable%: 0.1049%"
```

### Gradient checkpointing

```python
if args.gradient_checkpointing:
    model.thinker.enable_input_require_grads()
    model.thinker.gradient_checkpointing_enable()
```

Normally, the forward pass saves all intermediate values (activations) for the backward pass. With gradient checkpointing, it discards them and recomputes during backward. Trades ~30% more compute time for ~40% less GPU memory.

---

## 7. DataLoader (Lines 337–354)

```python
train_dataloader = DataLoader(
    train_dataset,
    batch_size=args.batch_size,   # 1 sample per batch
    shuffle=True,                  # randomize order each epoch
    collate_fn=collate_fn_bound,   # our custom function from step 4
    num_workers=2,                 # 2 CPU threads load data while GPU trains
    pin_memory=True,               # pre-load data into GPU-friendly memory
)
```

The DataLoader is a conveyor belt:
1. Picks random indices from the dataset
2. Calls `__getitem__` for each
3. Calls `collate_fn` to batch them together
4. Delivers the batch to the training loop

---

## 8. Timing Test (Lines 356–378)

```python
if args.timing_test_steps > 0:
    per_step = run_timing_test(model, train_dataloader, device, 50, ...)

    estimated_time = per_step * total_steps
    if estimated_time > 4 * 3600:
        print("WARNING: exceeds 4 hour limit!")
```

Runs 50 training steps, measures how long each takes, then extrapolates. If the estimate exceeds 4 hours, it warns you so you can cancel (`scancel`) and reduce data instead of wasting a full 4-hour GPU slot.

---

## 9. Optimizer and Scheduler (Lines 380–396)

```python
optimizer = torch.optim.AdamW(
    [p for p in model.parameters() if p.requires_grad],  # ONLY LoRA params
    lr=args.learning_rate,     # 0.0002
    weight_decay=args.weight_decay,  # 0.01
)
```

**AdamW** is the optimizer — the algorithm that actually updates LoRA weights. It's smarter than basic gradient descent:
- Keeps a running average of gradients (momentum)
- Keeps a running average of squared gradients (adaptive learning rate per parameter)
- Weight decay penalizes large parameter values (regularization)

`if p.requires_grad` filters to only LoRA parameters (the frozen 3B original params have `requires_grad=False`).

```python
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=100,          # gradually increase LR from 0 to 0.0002 over 100 steps
    num_training_steps=optimizer_steps,  # then linearly decay to 0
)
```

**Learning rate schedule:**

```
LR
0.0002 |       /‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾\
       |      /                      \
       |     /                        \
       |    /                          \
0      |___/                            \___
       0  100                         total steps
       ↑ warmup                      ↑ decay
```

Warmup prevents early instability (random initial gradients are large, high LR would overshoot). Linear decay helps fine-tune in the final steps.

---

## 10. Training Loop (Lines 398–516)

This is the core learning process. Let's trace one complete iteration:

### 10a: Start epoch

```python
for epoch in range(args.epochs):  # 3 epochs = 3 passes through all data
```

### 10b: Get a batch

```python
for step, batch in enumerate(progress_bar):
    batch = {k: v.to(device) for k, v in batch.items()}
```

Move the batch (input_ids, attention_mask, labels) from CPU to GPU.

### 10c: Forward pass

```python
with torch.amp.autocast("cuda", dtype=dtype):
    outputs = model.thinker(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        labels=batch["labels"],
    )
    loss = outputs.loss / args.gradient_accumulation_steps
```

**What happens inside `model.thinker(...)`:**
1. `input_ids` are looked up in the embedding table → vectors
2. Image tokens are replaced with vision encoder outputs
3. Audio tokens are replaced with audio encoder outputs
4. All vectors pass through ~30 transformer layers
5. At each layer, self-attention runs with LoRA modifications:
   - `Q = W_q(x) + B_q(A_q(x))` — original + LoRA delta
   - Same for K, V, O
6. The model predicts the next token at every position
7. Cross-entropy loss is computed between predictions and labels
8. Loss is only computed where labels != -100 (just the "SPEAKING"/"NOT_SPEAKING" tokens)

**`autocast`** automatically uses mixed precision — some operations run in 16-bit, others in 32-bit, for speed without losing accuracy.

**Dividing loss by `gradient_accumulation_steps`**: Since we accumulate gradients from 8 batches before updating, we average the loss so the update magnitude stays the same regardless of accumulation steps.

### 10d: Backward pass

```python
loss.backward()
```

PyTorch traces back through every operation in the forward pass and computes: "how much did each LoRA parameter contribute to the loss?" These are the **gradients** — stored in `param.grad` for each trainable parameter.

Frozen parameters (the original 3B weights) are skipped entirely.

### 10e: Gradient accumulation check

```python
if (step + 1) % args.gradient_accumulation_steps == 0:
```

We only update weights every 8 steps. Steps 1-7 just accumulate gradients (add them up). On step 8, we do the actual update:

### 10f: Gradient clipping

```python
torch.nn.utils.clip_grad_norm_(
    [p for p in model.parameters() if p.requires_grad],
    args.max_grad_norm,   # 1.0
)
```

If gradients are very large (exploding gradients), scale them down so the maximum norm is 1.0. Prevents catastrophic weight updates.

### 10g: Optimizer step

```python
optimizer.step()      # update LoRA weights using accumulated gradients
scheduler.step()      # adjust learning rate
optimizer.zero_grad()  # clear gradients for next accumulation cycle
```

This is where learning actually happens. For each LoRA parameter:
```
new_weight = old_weight - learning_rate * gradient
```
(AdamW is more sophisticated than this, but that's the basic idea.)

### 10h: Logging

```python
if global_step % args.logging_steps == 0:
    avg_loss = total_loss / args.logging_steps
    progress_bar.set_postfix({"loss": f"{avg_loss:.4f}", ...})
```

Every 10 optimizer steps, print the average loss. You want to see this number going **down** over time — that means the model is learning.

### 10i: Save checkpoint

```python
if global_step % args.save_steps == 0:
    model.thinker.save_pretrained(str(ckpt_dir))
```

Every 500 steps, save the LoRA weights. If training crashes at step 499, you lose everything. Checkpoints are insurance.

### 10j: End of epoch — save best model

```python
if avg_epoch_loss < best_loss:
    best_loss = avg_epoch_loss
    model.thinker.save_pretrained(str(best_dir))
```

After each full pass through the data, if this epoch's loss is the lowest so far, save it as `best_model`. This is the one you'll use for evaluation.

---

## 11. Save Final Model (Lines 517–542)

```python
model.thinker.save_pretrained(str(final_dir))
processor.save_pretrained(str(final_dir))
```

Saves:
- `adapter_model.safetensors` — the LoRA A and B matrices (~10-50MB)
- `adapter_config.json` — LoRA configuration (r, alpha, target modules)
- Processor files — tokenizer and config needed for inference

**Only LoRA weights are saved**, not the full 3B model. To use later:
```python
# Load original 3B model
model = Qwen2_5OmniForConditionalGeneration.from_pretrained("Qwen/Qwen2.5-Omni-3B")
# Add your LoRA adapter on top
model.thinker = PeftModel.from_pretrained(model.thinker, "output/best_model")
# Now model has original weights + your fine-tuned adjustments
```

```python
with open(output_dir / "training_log.json", "w") as f:
    json.dump({"log_history": log_history, ...}, f)
```

Also saves the full training log (loss at every logging step, total time, etc.) for analysis.

---

## Complete Data Flow Summary

```
metadata.jsonl
    ↓  ASDDataset.__getitem__()
conversation (system + user + assistant messages)
    ↓  collate_fn()
    ↓  processor.apply_chat_template() → text with special tokens
    ↓  process_mm_info() → extracts raw images and audio
    ↓  processor() → tokenizes text, encodes images/audio
input_ids + attention_mask + labels (tensors of numbers)
    ↓  move to GPU
    ↓  model.thinker() forward pass
    ↓  embeddings → transformer layers (with LoRA) → predictions
loss (single number: how wrong was the prediction?)
    ↓  loss.backward()
gradients (how to adjust each LoRA parameter)
    ↓  optimizer.step()
updated LoRA weights (slightly better at ASD now)
    ↓  repeat 2000 samples × 3 epochs
    ↓  save_pretrained()
adapter_model.safetensors (the trained LoRA weights, ~10-50MB)
```
