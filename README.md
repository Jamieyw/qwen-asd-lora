# LoRA Fine-Tuning Qwen2.5-Omni-3B for Active Speaker Detection

CS6140 Machine Learning — Final Project

Fine-tune [Qwen2.5-Omni-3B](https://huggingface.co/Qwen/Qwen2.5-Omni-3B) using LoRA on the [UniTalk-ASD](https://huggingface.co/datasets/plnguyen2908/UniTalk-ASD) dataset to detect which person in a video is currently speaking, based on face images and audio.

## Overview

**Active Speaker Detection (ASD)** is the task of determining whether a visible person in a video is the one producing the speech. This project fine-tunes a multimodal language model to solve this as a binary classification task (SPEAKING vs NOT_SPEAKING).

**Why Qwen2.5-Omni-3B?** It is the only Qwen model that processes both video/images and audio simultaneously, using TMRoPE (Time-aligned Multimodal RoPE) to synchronize visual and audio inputs — exactly what ASD requires.

**Why LoRA?** Full fine-tuning of a 3B parameter model requires ~48GB of GPU memory. LoRA freezes the original weights and trains small adapter matrices (~0.1% of parameters), reducing memory to ~20-27GB and fitting on a single V100.

## Project Structure

```
.
├── README.md              # This file
├── PROJECT_PLAN.md        # Detailed project plan
├── LORA_GUIDE.md          # Step-by-step LoRA explanation
├── TRAIN_WALKTHROUGH.md   # Line-by-line walkthrough of train_v2.py
├── requirements.txt       # Python dependencies
├── setup_env.sh           # Conda environment setup for cluster
├── prepare_data.py        # Download and subset the dataset
├── train_v2.py            # LoRA fine-tuning script (current — use this)
├── train.py               # Original training script (v1 — superseded)
├── train.sh               # SLURM sbatch job for training
├── evaluate.py            # Evaluation with metrics
└── evaluate.sh            # SLURM sbatch job for evaluation
```

> **Use `train_v2.py`** for all training. It adds LoRA on the vision encoder, label smoothing, and improved LoRA hyperparameters over the original `train.py`. See [Changes in v2](#changes-in-v2) for details.

## Dataset

[UniTalk-ASD](https://huggingface.co/datasets/plnguyen2908/UniTalk-ASD) contains ~48,693 face tracks from videos, each with:
- **Face crops** (JPG at 25fps) — sequential images of one person's face
- **Audio** (WAV) — the corresponding audio segment
- **Labels** — per-frame SPEAKING_AUDIBLE (1) or NOT_SPEAKING (0)

We use **~2,000 training tracks (~5%)** with 10 frames each, plus 500 validation tracks, to keep training under 4 hours.

### Prepared Data Structure

After running `prepare_data.py`, the `asd-data/` folder looks like this:

```
asd-data/
├── train/
│   ├── metadata.jsonl              # one JSON line per entity track
│   ├── images/                     # face crop JPGs
│   │   ├── DGyuHzlne6A_4287_frame000.jpg
│   │   ├── DGyuHzlne6A_4287_frame001.jpg
│   │   └── ...                     # up to 10 frames per entity
│   └── audio/                      # audio WAVs
│       ├── DGyuHzlne6A_4287.wav
│       └── ...
└── val/
    ├── metadata.jsonl
    ├── images/
    └── audio/
```

- **`images/`** — Face crops from video frames. Each JPG is one person's face at one timestamp. Naming: `{videoID}_{trackNum}_frame{N}.jpg`. Up to 10 evenly-spaced frames per track.
- **`audio/`** — The audio segment for each entity track. One WAV per person, covering the time window of their face track.
- **`metadata.jsonl`** — Ties everything together. Each line is one training sample:

```json
{
  "entity_id": "DGyuHzlne6A:4287",
  "image_paths": ["asd-data/train/images/DGyuHzlne6A_4287_frame000.jpg", "..."],
  "audio_path": "asd-data/train/audio/DGyuHzlne6A_4287.wav",
  "timestamps": [0.16, 0.32, 0.48],
  "labels": [1, 1, 0, 0, 1, 1, 0, 0, 1, 1],
  "majority_label": "SPEAKING",
  "num_frames": 10,
  "speaking_ratio": 0.6
}
```

### How Data Becomes a Training Sample

The training script reads each metadata line and builds a multimodal conversation for Qwen2.5-Omni. The model receives 10 face crop images and an audio clip, then predicts a single majority label:

| Role | Content |
|------|---------|
| **System** | "You are an active speaker detection system." |
| **User** | [frame1.jpg] ... [frame10.jpg] [audio.wav] + "These are 10 frames of a person's face with audio from the scene. The audio may or may not belong to this person... Is this person speaking? Answer with only SPEAKING or NOT_SPEAKING." |
| **Assistant** | "SPEAKING" or "NOT_SPEAKING" (majority label) |

Rather than generating the assistant's token via standard language modeling loss, training uses a **logit-based classification loss**: the model's logit for the `SPEAKING` token is compared to the `NOT_SPEAKING` logit at the last position, and binary cross-entropy is applied. This gives clean, direct gradient signal for the classification task.

## LoRA Configuration

### Thinker (Text LLM)

| Parameter | Value |
|-----------|-------|
| Rank (r) | 16 |
| Alpha | 32 |
| Dropout | 0.1 |
| Target modules | q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj |
| Trainable params | ~10M / 3B (~0.3%) |

### Vision Encoder (last 8 layers)

| Parameter | Value |
|-----------|-------|
| Rank (r) | 4 |
| Alpha | 8 |
| Dropout | 0.05 |
| Target modules | q, k, v (attention) in layers 24–31 |
| LR scale | 0.2× thinker LR |

## Changes in v2

`train_v2.py` improves on `train.py` in three areas:

**1. Vision encoder LoRA**
The original script only applied LoRA to the thinker (text LLM). v2 also applies a small-rank LoRA to the last 8 layers of the vision encoder. The model's main failure mode was: *hear audio + see a face → predict SPEAKING*. Fine-tuning the vision encoder helps it learn to distinguish open/moving lips from closed/still ones across the 10 frames.

**2. Label smoothing**
v2 trains with `label_smoothing=0.1` — soft labels (0.9 / 0.1) instead of hard (1 / 0). This prevents the model from becoming overconfident, acts as regularization, and produces better-calibrated predictions. Especially useful since ASD boundaries can be genuinely ambiguous.

**3. LoRA hyperparameter changes**
- Rank doubled (8 → 16) for more capacity in the thinker
- Feed-forward layers (`gate_proj`, `up_proj`, `down_proj`) added to LoRA targets
- Dropout increased (0.05 → 0.1) to compensate for larger adapter
- Learning rate lowered (1e-4 → 5e-5) for more stable training
- Epochs increased (5 → 8) to allow for slower, steadier convergence
- Separate optimizer param groups: vision encoder uses LR × 0.2 to protect pretrained visual features

## Running on NEU Explorer Cluster (Step-by-Step)

Complete walkthrough from SSH login to finished evaluation.

### Step 1: SSH into the cluster

```bash
ssh your_username@login.explorer.northeastern.edu
```

### Step 2: Check available GPUs

```bash
sinfo -p gpu --Format=gres:25,gresused:25,nodes:7,cpus:7,nodelist:50
```

This shows all GPU types and how many are free. Speed ranking:

| GPU | VRAM | Speed | Notes |
|-----|------|-------|-------|
| H200 | 141GB | Fastest | Often fully used, long queue |
| **A100** | **40GB** | **Fast** | **Best balance of speed and availability** |
| V100-SXM2 | 32GB | Good | Decent fallback |
| V100-PCIe | 32GB | OK | Slower V100 variant |
| T4 | 16GB | Slowest | Too small for this model |

A100 and H200 support **bf16 natively** (no `--fp16` flag needed). If using V100, add `--fp16` to the python command in `train.sh`.

### Step 3: Create project directory on cluster

```bash
mkdir -p ~/qwen-asd/logs
```

### Step 4: Transfer files from your local machine

Open a **new terminal on your Mac** (not the SSH session). `cd` into the project folder first:

```bash
cd /path/to/fine-tuning-qwen
scp *.py *.sh *.txt *.md your_username@login.explorer.northeastern.edu:~/qwen-asd/
```

### Step 5: Set up conda environment

Back in your **SSH session on the cluster**:

```bash
# Request an interactive compute node (don't install on the login node)
srun --partition=short --mem=8G --time=01:00:00 --pty bash

cd ~/qwen-asd

module load anaconda3
conda init
source ~/.bashrc

bash setup_env.sh
conda activate qwen-asd
```

Verify everything works:

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
python -c "from transformers import Qwen2_5OmniForConditionalGeneration; print('Model class: OK')"
```

### Step 6: Download and prepare data subset

```bash
python prepare_data.py
```

### Step 7: Submit the training job

```bash
sbatch train.sh
# Output: "Submitted batch job 1234567"
```

### Step 8: Monitor the job

```bash
squeue -u $USER
```

| Code | Meaning |
|------|---------|
| `PD` | Pending — waiting in queue for a GPU |
| `R` | Running — training is in progress |
| `CG` | Completing — job is finishing up |

```bash
tail -f logs/1234567.out   # watch output live
cat logs/1234567.err       # check for errors
scancel 1234567            # cancel if needed
```

The first thing the training script does is a **50-step timing test** that estimates total training time. If it says >4 hours, cancel and reduce data.

### Step 9: Submit evaluation job (after training completes)

```bash
sbatch evaluate.sh
```

```bash
cat output/eval_results.json
```

### Quick Reference: Useful Commands

```bash
squeue -u $USER              # list your jobs
sinfo -p gpu                 # see GPU partition info
scancel <job_id>             # cancel a job
cat logs/<job_id>.out        # view job output
tail -f logs/<job_id>.out    # follow output live (Ctrl+C to stop)
nvidia-smi                   # check GPU usage (on compute node)
```

---

## Training Details

- **Model:** Qwen2.5-Omni-3B (3 billion parameters, multimodal)
- **Method:** LoRA via PEFT on thinker + vision encoder last 8 layers
- **GPU:** A100 40GB (bf16) — also works on V100 32GB (fp16) or H200
- **Effective batch size:** 8 (batch_size=1 × gradient_accumulation=8)
- **Epochs:** 8
- **Learning rate:** 5e-5 (thinker), 1e-5 (vision encoder) with linear warmup
- **Label smoothing:** 0.1
- **Estimated training time:** ~2-4 hours on A100

## Evaluation Metrics

- **Accuracy** — % of correct predictions
- **Precision** — of all predicted SPEAKING, how many were actually speaking
- **Recall** — of all actually speaking, how many were caught
- **F1 Score** — harmonic mean of precision and recall
- **mAP** — mean Average Precision (area under the PR curve)
- **Confusion Matrix** — 2×2 grid of true/false positives and negatives

**Baseline comparison:** the evaluation script runs both with and without the LoRA adapter to measure improvement over the base model.

## References

- [Qwen2.5-Omni Technical Report](https://arxiv.org/abs/2503.20215)
- [UniTalk-ASD Paper](https://arxiv.org/html/2505.21954v1)
- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)
- [PEFT Library](https://github.com/huggingface/peft)
