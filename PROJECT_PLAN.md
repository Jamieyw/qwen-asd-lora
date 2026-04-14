# Fine-Tuning Qwen2.5-Omni-3B for Active Speaker Detection (ASD)

## Overview

**Goal:** Fine-tune Qwen2.5-Omni-3B using LoRA on the UniTalk-ASD dataset to detect which person in a video is currently speaking.

**Constraints:** Course project (CS6140), training within 4 hours on NEU Explorer Cluster, using sbatch.

---

## Model

**Qwen2.5-Omni-3B** (`Qwen/Qwen2.5-Omni-3B`)

- Only Qwen model that handles both video/images AND audio simultaneously
- 3 billion parameters (smallest available multimodal Qwen)
- Uses TMRoPE (Time-aligned Multimodal RoPE) to synchronize audio and video timestamps
- Open source, LoRA-compatible
- Fits on a V100 (32GB) with LoRA fine-tuning

**Fallback:** Quantized `Qwen2.5-Omni-7B-GPTQ-Int4` if 3B quality is insufficient.

---

## Dataset

**UniTalk-ASD** — https://huggingface.co/datasets/plnguyen2908/UniTalk-ASD

Full dataset: ~48,693 entity tracks (~36,500 training), 14.6 GB, ~4M face crops.
We use **~2,000 training tracks (~5%)** + 500 validation tracks, with 10 frames per track.

### Structure

```
root/
├── csv/
│   ├── train_orig.csv          # metadata + labels
│   └── val_orig.csv
├── clips_audios/
│   └── train|val/<video_id>/<entity_id>.wav    # audio segments
└── clips_videos/
    └── train|val/<video_id>/<entity_id>/<timestamp>.jpg  # face crops
```

### How It Works

- Each `entity_id` = one person's face track across multiple frames
- JPG files = face crops at 25fps
- WAV files = corresponding audio segment
- CSV labels each frame as `SPEAKING_AUDIBLE` (1) or `NOT_SPEAKING` (0)

---

## Fine-Tuning Method: LoRA

- **Rank (r):** 8
- **Alpha:** 16
- **Dropout:** 0.05
- **Target modules:** q_proj, k_proj, v_proj, o_proj
- Only trains ~0.1% of total parameters
- Training time: ~2-4 hours on 1x V100

---

## Project Files

| File | Purpose |
|------|---------|
| `requirements.txt` | Python dependencies |
| `prepare_data.py` | Download and format dataset subset |
| `train.py` | LoRA fine-tuning script |
| `evaluate.py` | Run inference, compute metrics |
| `train.sh` | SLURM sbatch job script |
| `evaluate.sh` | SLURM sbatch script for evaluation |
| `setup_env.sh` | Conda environment setup |

---

## Cluster Workflow (Explorer Cluster)

### A. First-time setup (do once)

```bash
# SSH into the cluster
ssh your_username@explorer.northeastern.edu

# Check available resources
sinfo                    # see partitions and GPU types
module avail             # see available modules

# Create project directory
mkdir -p ~/qwen-asd/logs
```

### B. Transfer files from local machine

```bash
# Run from YOUR MAC (not the cluster)
scp -r /path/to/fine-tuning-qwen/* your_username@explorer.northeastern.edu:~/qwen-asd/
```

### C. Set up environment on cluster

```bash
# Request an interactive session (don't install on login node)
srun --partition=short --mem=8G --time=01:00:00 --pty bash

# Set up conda environment
bash setup_env.sh
conda activate qwen-asd
```

### D. Download data subset

```bash
python prepare_data.py
```

### E. Submit training job

```bash
sbatch train.sh
# Output: "Submitted batch job 12345"
```

### F. Monitor

```bash
squeue -u $USER          # PD=pending, R=running
cat logs/12345.out       # view output
tail -f logs/12345.out   # live follow
scancel 12345            # cancel if needed
```

### G. Evaluate

```bash
srun --partition=gpu --gres=gpu:v100:1 --mem=32G --time=00:30:00 --pty bash
conda activate qwen-asd
python evaluate.py
```

---

## sbatch Script (`train.sh`)

```bash
#!/bin/bash
#SBATCH --job-name=qwen-asd
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1         # request 1 V100 (32GB)
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --time=04:00:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

module load cuda/12.1
module load anaconda3

conda activate qwen-asd

mkdir -p logs
python train.py
```

**GPU choices on Explorer:**
- K80 (12GB) — too small for this model
- V100 (32GB) — good fit
- H200 (141GB) — overkill but works

---

## Verification

- Training loss should decrease across epochs
- Evaluate accuracy, precision, recall, F1 on validation subset
- Compare against random baseline (50%)
- Generate confusion matrix
