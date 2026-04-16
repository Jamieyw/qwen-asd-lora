# train_v2.py — Full Code Walkthrough

A line-by-line explanation of every step in the training script.

> This walkthrough covers `train_v2.py`, the current training script. It differs from the original `train.py` in three ways: LoRA is applied to the vision encoder's last 8 layers, label smoothing (0.1) is used on the classification loss, and LoRA hyperparameters have been updated (r=16, alpha=32, dropout=0.1, feed-forward targets added).

---

## Big Picture

The script does 8 things in order:

1. Parse command-line arguments
2. Load the pre-trained Qwen2.5-Omni-3B model
3. Attach LoRA adapters to the thinker (text LLM) — attention + feed-forward
4. Attach LoRA adapters to the vision encoder's last N layers
5. Load the prepared dataset
6. Run a timing test to estimate total training time
7. Run the training loop
8. Save the trained LoRA weights

### The Workflow in Plain English

**Step 1: Read the recipe card.** We start with `metadata.jsonl` — each line describes one entity track: paths to 10 face images, the audio clip, and a majority label (SPEAKING or NOT_SPEAKING based on how many frames the person was speaking).

**Step 2: Build a question for the model.** We format this into a conversation: "Here are 10 face photos and an audio clip. Is this person speaking? Answer with only SPEAKING or NOT_SPEAKING." We do NOT include the answer in the input — instead, we let the model generate a prediction and read the logits directly.

**Step 3: Translate everything into numbers.** Face photos → grids of RGB numbers. Audio → amplitude values. Text → token IDs. All of these get stitched into one unified sequence and passed through the model.

**Step 4: The model thinks (forward pass).** The vision encoder (with LoRA on the last 8 layers) converts face images into embeddings. The audio encoder (frozen) converts audio. The thinker transformer layers process everything together using self-attention. LoRA modifies both the attention layers and feed-forward layers of the thinker.

**Step 5: Read the logit (classification head).** Instead of generating tokens and computing language modeling loss, we extract the logit values at the last sequence position for two specific tokens: `SPEAKING` and `NOT_SPEAKING`. We stack these into a 2-class classification problem and apply cross-entropy loss with label smoothing.

**Step 6: Figure out who's responsible (backward pass).** Gradients flow back through the network to both the thinker LoRA parameters and the vision encoder LoRA parameters. The frozen 3B original weights are skipped.

**Step 7: Nudge the weights (optimizer step).** Two optimizer parameter groups: thinker LoRA at lr=5e-5, vision encoder LoRA at lr=1e-5. Lower LR on the vision encoder protects pretrained visual feature representations.

**Step 8: Repeat and save.** 8 epochs × ~2000 samples. Best model saved after each epoch that achieves lowest average loss.

---

## 1. Imports (Lines 16–33)

```python
import torch
from peft import LoraConfig, get_peft_model, TaskType
from torch.utils.data import Dataset, DataLoader
from transformers import (
    Qwen2_5OmniForConditionalGeneration,
    Qwen2_5OmniProcessor,
    get_linear_schedule_with_warmup,
)
```

- `torch` — runs math on the GPU
- `peft` — LoRA library from HuggingFace
- `transformers` — provides the Qwen model and processor
- `Dataset`/`DataLoader` — handle feeding data in batches

---

## 2. Arguments (Lines 40–71)

Key arguments and their defaults:

| Argument | Default | What it controls |
|----------|---------|-----------------|
| `--epochs` | 8 | Training passes through the full dataset |
| `--learning_rate` | 5e-5 | Thinker LoRA learning rate |
| `--lora_r` | 16 | LoRA rank for the thinker |
| `--lora_alpha` | 32 | LoRA scaling (alpha/r = 2) |
| `--lora_dropout` | 0.1 | LoRA dropout rate |
| `--label_smoothing` | 0.1 | Soft label factor (0 = hard labels) |
| `--unfreeze_vision_layers` | 8 | How many vision encoder layers get LoRA |
| `--vision_lr_scale` | 0.2 | Vision encoder LR = learning_rate × this |
| `--gradient_accumulation_steps` | 8 | Effective batch = 1 × 8 = 8 |
| `--fp16` | off | Use fp16 (needed for V100, not A100) |
| `--gradient_checkpointing` | off | Trade compute time for memory |

---

## 3. ASDDataset Class (Lines 78–158)

### `__init__` — Load metadata

```python
def __init__(self, data_dir, split="train"):
    metadata_path = self.data_dir / "metadata.jsonl"
    with open(metadata_path) as f:
        for line in f:
            self.metadata.append(json.loads(line.strip()))
```

Reads the `metadata.jsonl` file produced by `prepare_data.py`. Each line is a dict with image paths, audio path, per-frame labels, and majority label.

### `__getitem__` — Build one training conversation

Builds a conversation in Qwen2.5-Omni's format. The key design choice is a **simple, direct prompt** — earlier experiments showed that verbose per-frame prompts caused the model to default to NOT_SPEAKING by paying too little attention to the actual media.

```python
conversation = [
    {
        "role": "system",
        "content": [{"type": "text", "text": "You are an active speaker detection system."}],
    },
    {
        "role": "user",
        "content": [
            {"type": "image", "image": "path/to/face_frame000.jpg"},
            # ... 10 images total
            {"type": "audio", "audio": "path/to/audio.wav"},
            {"type": "text", "text": "These are 10 frames of a person's face with audio from the scene. The audio may or may not belong to this person — someone else in the scene could be the one speaking. Is this person speaking? Answer with only SPEAKING or NOT_SPEAKING."},
        ],
    },
    {
        "role": "assistant",
        "content": [{"type": "text", "text": "SPEAKING"}],  # majority label
    },
]
```

The assistant message is included in the returned dict but **removed** inside `collate_fn` before passing to the model. This is intentional — we use logit-based classification, not teacher-forced generation.

---

## 4. Collate Function (Lines 161–221)

This is called by the DataLoader to combine samples into a batch.

### What it does

```python
# Remove the assistant answer — we classify via logits, not generation
conversation = sample["conversation"][:-1]

# Apply chat template WITH generation prompt
text = processor.apply_chat_template(
    conversation,
    add_generation_prompt=True,  # adds <|im_start|>assistant\n at the end
    tokenize=False,
)

# Process multimodal inputs
audios, images, videos = process_mm_info(conversation, use_audio_in_video=False)
inputs = processor(text=text, audio=audios, images=images, ...)
```

`add_generation_prompt=True` is critical: it puts the model in "about to answer" state, so the logit at the last position is the model's best prediction of what token comes first in the answer — exactly where we want to compare SPEAKING vs NOT_SPEAKING logits.

**No labels tensor** — unlike the original `train.py` which passed `labels` to the model for language modeling loss, `train_v2.py` only produces `input_ids`, `attention_mask`, and `class_labels` (integer 0 or 1). The loss is computed manually in the training loop.

---

## 5. Vision Encoder LoRA Setup (Lines 270–318)

```python
def setup_vision_encoder_lora(model, args):
    total_vision_layers = 32
    start_layer = total_vision_layers - args.unfreeze_vision_layers  # default: 24

    vision_target_modules = []
    for i in range(start_layer, total_vision_layers):
        vision_target_modules.extend([
            f"blocks.{i}.attn.q",
            f"blocks.{i}.attn.k",
            f"blocks.{i}.attn.v",
        ])

    vision_lora_config = LoraConfig(r=4, lora_alpha=8, lora_dropout=0.05,
                                    target_modules=vision_target_modules)
    model.thinker.visual = get_peft_model(model.thinker.visual, vision_lora_config)
```

**Why the vision encoder?** The model's main failure mode was: *it hears audio + sees a face → defaults to SPEAKING*. The audio encoder already provides audio information. What the model needs to learn is to use the face images more selectively — looking at whether lips are moving rather than just whether a face is present. Fine-tuning the last 8 vision layers (where high-level semantic features form) teaches this.

**Why only the last 8 layers?** Early vision encoder layers detect low-level features (edges, textures) that should stay pretrained. The last layers produce high-level semantic representations where "lip movement vs. static face" is encoded. Touching only those avoids disrupting general visual understanding.

**Why different attention names?** The vision encoder uses short names (`q`, `k`, `v`) instead of the thinker's `q_proj`, `k_proj`, `v_proj`. If this causes issues, check with:
```python
for n, _ in model.thinker.visual.named_modules():
    if any(x in n for x in ['.q', '.k', '.v']): print(n)
```

---

## 6. Optimizer Setup (Lines 321–357)

```python
def build_optimizer(model, args):
    thinker_params = []
    vision_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "visual" in name:
            vision_params.append(param)
        else:
            thinker_params.append(param)

    param_groups = [
        {"params": thinker_params, "lr": args.learning_rate},         # 5e-5
        {"params": vision_params,  "lr": args.learning_rate * 0.2},   # 1e-5
    ]
    optimizer = torch.optim.AdamW(param_groups, weight_decay=0.01)
```

**Why two parameter groups?** We want the thinker to adapt significantly to the ASD classification task. We want the vision encoder to make only small adjustments — enough to become sensitive to lip movement, but not so much that it forgets what faces look like. A 5× lower LR achieves this balance.

---

## 7. Load Model and Apply LoRA (Lines 374–419)

```python
# Load base model
model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
    args.model_name, torch_dtype=dtype, device_map="auto"
)

# Apply thinker LoRA (attention + feed-forward)
lora_config = LoraConfig(
    r=16, lora_alpha=32, lora_dropout=0.1,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    task_type=TaskType.CAUSAL_LM,
)
model.thinker = get_peft_model(model.thinker, lora_config)

# Apply vision encoder LoRA (last 8 layers, q/k/v only)
setup_vision_encoder_lora(model, args)
```

**Why include feed-forward layers in the thinker?** The feed-forward layers (`gate_proj`, `up_proj`, `down_proj`) are responsible for transforming representations after attention. For classification tasks, they often encode the "meaning" that gets mapped to the output. Including them doubles the adapter's capacity at modest memory cost.

**Why `model.thinker`?** Qwen2.5-Omni has two parts: the **Thinker** (processes input and generates text) and the **Talker** (generates speech). We don't need speech output for ASD, so only the thinker is fine-tuned.

---

## 8. Training Loop (Lines 521–613)

### Forward pass + logit classification

```python
with torch.amp.autocast("cuda", dtype=dtype):
    outputs = model.thinker(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
    )

    logits = outputs.logits  # (batch, seq_len, vocab_size)

    # Get logit at the last real token (where model predicts first answer token)
    seq_lens = batch["attention_mask"].sum(dim=1) - 1
    last_logits = logits[torch.arange(logits.size(0)), seq_lens]  # (batch, vocab_size)

    # Extract the two class logits
    speaking_logit = last_logits[:, speaking_token_id]
    not_speaking_logit = last_logits[:, not_token_id]

    # Binary classification loss with label smoothing
    class_logits = torch.stack([not_speaking_logit, speaking_logit], dim=1)
    loss = torch.nn.functional.cross_entropy(
        class_logits, batch["class_labels"], label_smoothing=args.label_smoothing
    )
    loss = loss / args.gradient_accumulation_steps
```

**Why logit-based classification?** The standard approach would be to include the answer in the input and compute language modeling loss on the answer tokens. This works but gives weak gradient signal for a binary task — the model is graded on predicting the exact token, not on understanding the task. Logit comparison directly trains the model to make the SPEAKING logit higher than NOT_SPEAKING (or vice versa), which is exactly what we want.

**What is `label_smoothing=0.1`?** Instead of telling the model "be 100% sure this is SPEAKING," it tells the model "be 90% confident." This prevents overconfident predictions, reduces the gradient magnitude for near-correct predictions, and acts as regularization — all especially valuable when training on a small dataset (~2000 samples).

### Gradient accumulation and clipping

```python
if (step + 1) % args.gradient_accumulation_steps == 0:
    torch.nn.utils.clip_grad_norm_(
        [p for p in model.parameters() if p.requires_grad],
        args.max_grad_norm,   # 1.0
    )
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad()
    global_step += 1
```

Gradient clipping (max norm = 1.0) prevents catastrophic updates when gradients are large (common in early training or after batch outliers).

### Best model saving

```python
if avg_epoch_loss < best_loss:
    best_loss = avg_epoch_loss
    model.thinker.save_pretrained(str(best_dir))
    processor.save_pretrained(str(best_dir))
```

Saves at the end of each epoch if this epoch's average loss is the lowest so far. The `best_model/` directory is what `evaluate.py` loads by default.

---

## 9. Save Final Model (Lines 618–638)

```python
model.thinker.save_pretrained(str(final_dir))
processor.save_pretrained(str(final_dir))
```

Saves:
- `adapter_model.safetensors` — thinker LoRA A and B matrices (~20-50MB)
- `adapter_config.json` — LoRA configuration
- Processor/tokenizer files

The vision encoder LoRA is saved as part of the thinker's PEFT model (since `model.thinker.visual` is a submodule of `model.thinker`).

To load and use:
```python
model = Qwen2_5OmniForConditionalGeneration.from_pretrained("Qwen/Qwen2.5-Omni-3B")
model.thinker = PeftModel.from_pretrained(model.thinker, "output/best_model")
```

---

## Complete Data Flow Summary

```
metadata.jsonl (entity_id, image paths, audio path, majority_label)
    ↓  ASDDataset.__getitem__()
conversation (system + user with 10 images + audio + question)
    ↓  collate_fn()  [removes assistant message, adds generation prompt]
    ↓  processor.apply_chat_template() → text with special tokens
    ↓  process_mm_info() → raw images and audio
    ↓  processor() → input_ids, attention_mask
    ↓  class_labels (0 or 1)
    ↓  move to GPU
    ↓  model.thinker() forward pass
    ↓    vision encoder (LoRA on last 8 layers) → visual embeddings
    ↓    audio encoder (frozen) → audio embeddings
    ↓    thinker transformer (LoRA on attention + FFN) → logits
    ↓  extract logit at last position for SPEAKING and NOT_SPEAKING
    ↓  cross_entropy(class_logits, labels, label_smoothing=0.1)
    ↓  loss.backward()
    ↓  gradients for thinker LoRA (lr=5e-5) + vision LoRA (lr=1e-5)
    ↓  optimizer.step()
updated LoRA weights (better at lip-movement-based ASD)
    ↓  repeat ~2000 samples × 8 epochs
    ↓  save_pretrained()
adapter_model.safetensors (trained LoRA weights, ~20-50MB)
```
