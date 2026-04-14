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
├── requirements.txt       # Python dependencies
├── setup_env.sh           # Conda environment setup for cluster
├── prepare_data.py        # Download and subset the dataset
├── train.py               # LoRA fine-tuning script
├── train.sh               # SLURM sbatch job for training
├── evaluate.py            # Evaluation with metrics
└── evaluate.sh            # SLURM sbatch job for evaluation
```

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

The training script reads each metadata line and builds a multimodal conversation for Qwen2.5-Omni. The model must analyze **each frame individually** and output per-frame labels:

| Role | Content |
|------|---------|
| **System** | "You are an active speaker detection system. Analyze each frame..." |
| **User** | [frame1.jpg] [frame2.jpg] ... [frame10.jpg] [audio.wav] + "For each frame, determine if this person is speaking" |
| **Assistant** | "Frame 1: SPEAKING\nFrame 2: SPEAKING\nFrame 3: NOT_SPEAKING\n...\nOverall: SPEAKING (6/10 frames)" |

This per-frame approach forces the model to actually analyze lip movements and audio at each timestamp, rather than taking shortcuts with a single-token answer.

## LoRA Configuration

| Parameter | Value |
|-----------|-------|
| Rank (r) | 8 |
| Alpha | 16 |
| Dropout | 0.05 |
| Target modules | q_proj, k_proj, v_proj, o_proj |
| Trainable params | ~3M / 3B (~0.1%) |

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

This shows all GPU types and how many are free. Look for GPUs with available slots (where `GRES_USED` is less than `GRES`). Speed ranking:

| GPU | VRAM | Speed | Notes |
|-----|------|-------|-------|
| H200 | 141GB | Fastest | Often fully used, long queue |
| **A100** | **40GB** | **Fast** | **Best balance of speed and availability** |
| V100-SXM2 | 32GB | Good | Decent fallback |
| V100-PCIe | 32GB | OK | Slower V100 variant |
| T4 | 16GB | Slowest | Too small for this model |

Pick the fastest GPU that has free slots, then update `train.sh` and `evaluate.sh` to match. For example, if you pick A100:

```bash
# In train.sh, the line should be:
#SBATCH --gres=gpu:a100:1

# In evaluate.sh, same:
#SBATCH --gres=gpu:a100:1
```

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

This sends only the code files (~60KB), not data. Data gets downloaded directly on the cluster later.

### Step 5: Set up conda environment

Back in your **SSH session on the cluster**:

```bash
# Request an interactive compute node (don't install on the login node)
srun --partition=short --mem=8G --time=01:00:00 --pty bash

# Go to your project
cd ~/qwen-asd

# Load conda and initialize it (first time only)
module load anaconda3
conda init
source ~/.bashrc

# Run the setup script (creates conda env + installs all dependencies)
bash setup_env.sh

# Activate the environment
conda activate qwen-asd
```

Verify everything works:

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
python -c "from transformers import Qwen2_5OmniForConditionalGeneration; print('Model class: OK')"
```

### Step 6: Download and prepare data subset

Still in the interactive session:

```bash
python prepare_data.py
```

This downloads CSVs + audio + images from HuggingFace and samples ~2,000 training tracks (~5% of data). Takes ~10-30 minutes depending on network speed.

### Step 7: Submit the training job

```bash
sbatch train.sh
# Output: "Submitted batch job 1234567"
```

This puts your job in the SLURM queue. It will run on a GPU node when one becomes available.

### Step 8: Monitor the job

```bash
# Check job status
squeue -u $USER
```

Status codes:
| Code | Meaning |
|------|---------|
| `PD` | Pending — waiting in queue for a GPU |
| `R` | Running — training is in progress |
| `CG` | Completing — job is finishing up |
| (gone) | Job finished (check logs for results) |

The `REASON` column explains why a job is pending:
- `(Priority)` — other higher-priority jobs are ahead of you
- `(Resources)` — waiting for the requested GPU type to free up

```bash
# Once the job is running (status = R), watch the output live:
tail -f logs/1234567.out

# Check for errors:
cat logs/1234567.err

# Cancel a job if needed:
scancel 1234567
```

The first thing the training script does is a **50-step timing test** that estimates total training time. If it says >4 hours, cancel and reduce data.

### Step 9: Submit evaluation job (after training completes)

```bash
sbatch evaluate.sh
```

This runs evaluation twice: once with the LoRA adapter (fine-tuned model) and once without (baseline), so you can compare the improvement.

```bash
# Check eval results when done:
cat logs/<eval_job_id>.out

# Or read the JSON results:
cat output/eval_results.json
```

### Quick Reference: Useful Commands

```bash
squeue -u $USER              # list your jobs
sinfo -p gpu                 # see GPU partition info
scancel <job_id>             # cancel a job
cat logs/<job_id>.out        # view job output
cat logs/<job_id>.err        # view job errors
tail -f logs/<job_id>.out    # follow output live (Ctrl+C to stop)
nvidia-smi                   # check GPU usage (on compute node)
du -sh ~/qwen-asd/data/      # check data size
```

---

## Training Details

- **Model:** Qwen2.5-Omni-3B (3 billion parameters, multimodal)
- **Method:** LoRA via PEFT library applied to the thinker component
- **GPU:** A100 40GB (bf16) — also works on V100 32GB (fp16) or H200
- **Effective batch size:** 8 (batch_size=1 x gradient_accumulation=8)
- **Epochs:** 3
- **Learning rate:** 2e-4 with linear warmup
- **Estimated training time:** ~2-4 hours on A100

The training script includes a **50-step timing test** at startup that estimates total training time before committing to the full run.

## Evaluation Metrics

**Per-frame metrics (primary):**
- Frame Accuracy, Precision, Recall, F1 Score
- Frame mAP (mean Average Precision)
- Frame Confusion Matrix

**Track-level metrics:**
- Track mAP — AP per entity track, averaged (standard ASD metric)
- Track Accuracy, F1

**Baseline comparison:** evaluates the model both with and without LoRA adapter to measure improvement over the base model.

## References

- [Qwen2.5-Omni Technical Report](https://arxiv.org/abs/2503.20215)
- [UniTalk-ASD Paper](https://arxiv.org/html/2505.21954v1)
- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)
- [PEFT Library](https://github.com/huggingface/peft)
