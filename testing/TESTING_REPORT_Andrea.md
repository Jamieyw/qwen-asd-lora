# Part 2 Testing Report

**Testing Lead:** Andrea
**Date:** April 2026

## Testing Work Completed

### 1. Environment Setup
- Platform: Google Colab Pro (GPU)
- Installed Qwen2.5-Omni compatible packages
- Verified GPU availability (V100/A100)

### 2. Data Preparation
- Downloaded 500 UniTalk-ASD validation samples using prepare_data.py
- Format: 10 face frames + 1 audio per entity track
- Distribution: 271 NOT_SPEAKING (54.2%), 229 SPEAKING (45.8%)

### 3. Evaluations Performed

#### Baseline (Zero-shot Qwen2.5-Omni-3B):
- Tested in Colab: **51.2% accuracy**
- Verified on cluster: 51.4%
- ✓ Results match - methodology validated

#### LoRA Models:
- Output1 (early): 47.6% - ineffective
- Output2 (train.py v1): 51.4% - no improvement over zero-shot
- Output3 (train_v2.py): 55.6% - tested on cluster (weights too large for Colab)

### 4. Final Comparison

| Model | Accuracy | Notes |
|-------|----------|-------|
| Part 1 Simple CNN-LSTM | 63.8% | Best overall - task-specific design |
| Zero-shot Qwen | 51.4% | Foundation model baseline |
| LoRA v2 (output3) | 55.6% | Train_v2 improvements effective (+4.2%) |

## Key Findings

1. **Reproducibility verified**: Colab and cluster results consistent
2. **Train.py v1 ineffective**: No improvement over baseline
3. **Train_v2.py effective**: Vision encoder LoRA + label smoothing work
4. **Task-specific architecture superior**: Part 1 simple model outperforms LLMs

## Testing Coordination

- Coordinated fair comparison on unified test set (same 500 samples)
- Tested models available in Colab environment
- Collaborated with Training Lead for cluster-only models
- Ensured consistent evaluation methodology across all models

## Results Files

Available in repo:
- `eval_results_baseline.json` - Zero-shot summary metrics
- `eval_results_lora.json` - LoRA v2 (output3) summary metrics

**Testing Status: Complete ✓**
