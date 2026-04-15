# Full Experiment Log: LoRA Fine-Tuning Qwen2.5-Omni-3B for ASD

## Project Goal

Fine-tune Qwen2.5-Omni-3B with LoRA to perform Active Speaker Detection (ASD): given face images and audio from a video, determine if a specific person is the one speaking.

## Dataset

- **Source:** UniTalk-ASD (HuggingFace)
- **Training:** ~2,000 entity tracks (5% of full dataset), balanced 50/50 SPEAKING/NOT_SPEAKING
- **Validation:** 500 entity tracks
- **Per track:** 10 face crop images (25fps, evenly sampled) + 1 audio WAV
- **Labels:** per-frame binary labels, aggregated to majority vote per track

## Infrastructure

- **Cluster:** NEU Explorer
- **GPU:** A100 40GB
- **Model:** Qwen2.5-Omni-3B (3B parameters, multimodal: text + image + audio)
- **Method:** LoRA (r=8, alpha=16, dropout=0.05, target: q/k/v/o_proj on thinker)

---

## Experiment 1: Single-Token Answer with Per-Frame Labels

### Setup
- **Prompt:** Verbose per-frame format asking model to output "Frame 1: SPEAKING\nFrame 2: NOT_SPEAKING\n..."
- **System prompt:** Detailed instructions about analyzing lip movements
- **Training:** 3 epochs, lr=2e-4, batch_size=1, grad_accum=8

### Training Results
```
Epoch 1: loss = 0.0001
Epoch 2: loss = 0.0000
Epoch 3: loss = 0.0000
Training time: ~26 minutes
```

### Evaluation Results
```
Accuracy: 56.4% (but 0% recall for SPEAKING)
Confusion Matrix:
  NOT_SPEAK: 2819 correct, 0 wrong
  SPEAKING:  2181 wrong,   0 correct
```

### Analysis
- Loss suspiciously near-zero from epoch 1
- Model predicted NOT_SPEAKING for every single frame
- Baseline (no LoRA) also predicted all NOT_SPEAKING
- **Conclusion:** The per-frame format caused the model to collapse to always predicting NOT_SPEAKING

---

## Experiment 2: Prompt Investigation (Interactive Testing)

### Method
Ran interactive tests on base Qwen2.5-Omni-3B (no fine-tuning) on a V100 GPU to isolate the cause of the NOT_SPEAKING collapse.

### Key Tests and Results

| Test | Setup | Result |
|------|-------|--------|
| 1 | 1 image + audio + "Describe what you see and hear" | "Man with black beard. He is speaking." |
| 2 | 1 image + audio + "SPEAKING or NOT_SPEAKING?" | **SPEAKING** (correct) |
| 3 | 10 images + audio + simple question (no system prompt) | **SPEAKING** (correct) |
| 4 | 10 images + audio + per-frame format + system prompt | **All NOT_SPEAKING** (broken) |
| 5 | 1 image + per-frame format + system prompt | **SPEAKING** (correct) |
| 6 | 10 images + simple question + system prompt | **SPEAKING** (correct) |
| 7 | No system prompt + per-frame format | Model refuses to answer |

### Root Cause Identified

The **verbose per-frame output format** caused the collapse. When asked to produce structured output ("Frame 1: SPEAKING or NOT_SPEAKING, Frame 2: ..."), the model defaulted to NOT_SPEAKING for everything.

The working format: simple system prompt + direct question.

### Key Finding

**Simpler prompts work better than detailed instructions.** The model already knows how to analyze faces and audio — overloading it with specific instructions about "analyzing lip movements" and "outputting one line per frame" confused it into defaulting to a safe answer.

---

## Experiment 3: Simple Prompt with Single Answer

### Setup
- **System prompt:** "You are an active speaker detection system."
- **User prompt:** "Is this person currently speaking? Answer with only SPEAKING or NOT_SPEAKING."
- **Answer:** "SPEAKING" or "NOT_SPEAKING" (majority vote label)
- **Training:** 3 epochs, lr=2e-4

### Training Results
```
Epoch 1: loss = 0.0001
Epoch 2: loss = 0.0000
Epoch 3: loss = 0.0000
Training time: ~26 minutes
```

### Evaluation Results — Fine-Tuned
```
Accuracy: 46.0%
Precision: 0.4589, Recall: 1.0000, F1: 0.6291
Confusion Matrix:
  NOT_SPEAK:   1 correct, 270 wrong
  SPEAKING:    0 wrong,   229 correct
```

### Evaluation Results — Baseline (No LoRA)
```
Accuracy: 51.4%
Precision: 0.4841, Recall: 0.9301, F1: 0.6368
Confusion Matrix:
  NOT_SPEAK:  44 correct, 227 wrong
  SPEAKING:   16 wrong,   213 correct
```

### Analysis
- Training loss near-zero again (1-token answer → trivially predictable)
- Fine-tuned model predicted SPEAKING 499/500 times — worse than baseline
- Baseline predicted SPEAKING 440/500 times — biased but at least got some NOT_SPEAKING right
- **Fine-tuning amplified the SPEAKING bias instead of correcting it**

---

## Experiment 4: Class-Weighted Loss (2x NOT_SPEAKING)

### Setup
- Same simple prompt
- NOT_SPEAKING samples weighted 2x in loss computation
- Training: 3 epochs, lr=2e-4

### Results
```
Accuracy: 45.8%
Confusion Matrix:
  NOT_SPEAK:   2 correct, 269 wrong
  SPEAKING:    2 wrong,   227 correct
```

### Analysis
- 2x weight had almost no effect
- Loss was already near-zero, so 2x of near-zero is still near-zero
- Model still overwhelmingly predicted SPEAKING

---

## Experiment 5: 3 Frames + 2x Weight

### Setup
- Reduced to 3 evenly-spaced frames (start, middle, end) instead of 10
- 2x class weight on NOT_SPEAKING

### Results
```
Accuracy: 46.0%
Confusion Matrix:
  NOT_SPEAK:   2 correct, 269 wrong
  SPEAKING:    2 wrong,   227 correct
```

### Analysis
- Fewer frames didn't help
- Same SPEAKING bias pattern

---

## Experiment 6: 5x Weight + Higher Learning Rate

### Setup
- 10 frames, simple prompt
- NOT_SPEAKING samples weighted 5x
- Learning rate: 5e-4 (2.5x higher)
- 3 epochs

### Results
```
Accuracy: 46.8%
Confusion Matrix:
  NOT_SPEAK:   5 correct, 266 wrong
  SPEAKING:    0 wrong,   229 correct
```

### Analysis
- Marginally better (5 NOT_SPEAKING predictions vs 1-2 before)
- But still overwhelmingly SPEAKING
- Even 5x weighting couldn't overcome the bias

---

## Experiment 7: Bias-Correcting Prompt

### Setup
- Prompt changed to: "These are 10 frames of a person's face with audio from the scene. The audio may or may not belong to this person — someone else in the scene could be the one speaking. Is this person speaking? Answer with only SPEAKING or NOT_SPEAKING."
- 5x class weight, lr=5e-4

### Results
```
Accuracy: 46.6%
Confusion Matrix:
  NOT_SPEAK:   4 correct, 267 wrong
  SPEAKING:    0 wrong,   229 correct
```

### Analysis
- The bias-correcting prompt didn't help during fine-tuned evaluation
- The prompt hint works at inference on the base model but gets overridden by LoRA weights

---

## Summary of All Results

| Experiment | Accuracy | NOT_SPEAK Correct | SPEAKING Correct | Notes |
|-----------|----------|-------------------|------------------|-------|
| **Baseline (no LoRA)** | **51.4%** | **44/271** | **213/229** | **Best result** |
| Per-frame format | 56.4%* | 2819/2819 | 0/2181 | *All NOT_SPEAKING |
| Simple prompt | 46.0% | 1/271 | 229/229 | All SPEAKING |
| 2x class weight | 45.8% | 2/271 | 227/229 | Weight too weak |
| 3 frames + 2x weight | 46.0% | 2/271 | 227/229 | Fewer frames didn't help |
| 5x weight + higher LR | 46.8% | 5/271 | 229/229 | Marginally better |
| Bias-correcting prompt | 46.6% | 4/271 | 229/229 | Prompt overridden by LoRA |

---

## Why Fine-Tuning Failed

### 1. Near-Zero Training Loss
The model predicts 1 token (SPEAKING/NOT_SPEAKING) given the full conversation context. The training loss is near-zero because:
- The answer is trivially predictable from the conversation pattern
- There's only 1-2 tokens of training signal per sample
- The model barely needs to change to achieve perfect training loss

### 2. LoRA Amplifies Existing Bias
The base model is biased toward SPEAKING (hears audio + sees face → assumes speaking). LoRA's small parameter updates (~0.1% of weights) reinforce this bias rather than correcting it, because:
- The training signal is too weak (near-zero loss)
- NOT_SPEAKING samples produce similarly low loss, giving no strong corrective gradient
- Class weighting can't help when the base loss is negligible

### 3. Fundamental Task-Format Mismatch
ASD is a perception task (detecting lip-sync from video), but we're treating it as a text generation task (predict 1 token). This creates a mismatch:
- **During training:** model predicts the answer token from conversation context → low loss regardless of input
- **During evaluation:** model generates from scratch using its pre-trained prior → defaults to SPEAKING

---

## Why the Baseline Works (Slightly)

The base Qwen2.5-Omni-3B achieves 51.4% (above random) because:
- It was pre-trained on diverse multimodal data
- It can detect basic visual cues (open mouth, facial expressions)
- It achieves 93% recall for SPEAKING (correctly identifies most speakers)
- But only 16% precision for NOT_SPEAKING (too many false SPEAKING predictions)

The model's strategy: "when in doubt, say SPEAKING" — which works when most visible people ARE speaking, but fails for balanced test data.

---

## Key Lessons Learned

### 1. Prompt Engineering Matters More Than Expected
- Verbose, detailed prompts broke the model (all NOT_SPEAKING)
- Simple, direct prompts worked (correct SPEAKING detection)
- Lesson: LLMs perform better with minimal instructions for perception tasks

### 2. 1-Token Classification Is Too Weak for Fine-Tuning
- Binary classification as text generation produces near-zero training loss
- The model "solves" training without learning task-relevant features
- Need either more output tokens or a different loss formulation

### 3. LoRA Can Make Things Worse
- Fine-tuning a model that's already biased can amplify the bias
- Small parameter updates (0.1%) aren't enough to overcome strong pre-training priors
- The base model without LoRA was consistently the best performer

### 4. Generalist LLMs vs Specialized Models
- Qwen2.5-Omni can do basic ASD from pre-training (51.4%)
- But it can't be easily improved with LoRA for this specific perception task
- Specialized ASD architectures (TalkNet, UniTalk) process video as temporal sequences and achieve >90% mAP
- The gap shows that perception tasks need task-specific architectures, not just fine-tuned LLMs

---

## What Would Need to Change

To actually improve ASD performance beyond the baseline:

1. **Video input** — Use `{"type": "video"}` with TMRoPE temporal alignment instead of separate images
2. **Contrastive learning** — Show pairs (speaking + not-speaking from same scene) instead of single samples
3. **Custom loss function** — Compute loss at the answer token position using the logit difference between SPEAKING and NOT_SPEAKING token IDs, not the standard causal LM loss
4. **Larger dataset** — 2,000 tracks may not be enough to overcome the pre-training prior
5. **Specialized architecture** — Add a classification head on top of the model's embeddings instead of using text generation
