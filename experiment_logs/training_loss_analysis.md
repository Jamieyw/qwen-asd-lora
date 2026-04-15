# Training Loss Analysis: Why 0.001 Loss is Suspicious

Date: 2026-04-15

## Observation

After epoch 1/3, the training loss was 0.0011 — suspiciously low for a binary classification task on a balanced 50/50 dataset.

## Training Setup

- **Loss function**: Custom binary cross-entropy on two token logits (SPEAKING vs NOT) at the last position, instead of standard causal LM loss
- **LoRA target**: Only the thinker (text LLM) component — vision and audio encoders are frozen
- **Dataset**: ~50/50 balanced SPEAKING/NOT_SPEAKING (enforced by prepare_data.py sampling)

## Why This Loss is Too Low

A loss of 0.001 on binary classification means the model is ~100% confident on every prediction. For a genuine ASD task where some cases are ambiguous (mouth slightly open, background noise, unclear lip movements), that's unrealistic.

Expected healthy loss range: **0.05–0.15** after convergence.

## Likely Cause: LoRA Shortcut Learning

The LoRA only adapts the thinker (text LLM). The vision and audio encoders are frozen. This means the training gradient does not flow back through the multimodal encoders.

The LoRA may be learning a text-level shortcut — e.g., a bias after the prompt template — rather than genuinely learning to interpret lip movements and audio content. It can get near-zero loss by memorizing a pattern in the prompt/template structure without attending to the actual visual and audio inputs.

Evidence: the fine-tuned model performed worse than the base model on evaluation (46.8% vs 51.4% accuracy), suggesting the LoRA learned something degenerate rather than useful.

## What Good Training Would Look Like

- Loss starts at ~0.5–0.7 (random chance on binary)
- Gradually decreases to ~0.05–0.15 over multiple epochs
- Model shows some uncertainty on ambiguous cases
- Evaluation accuracy meaningfully exceeds the baseline

## Possible Mitigations

1. Lower learning rate (e.g., 1e-5 instead of 2e-4) to prevent rapid collapse
2. Improve the prompt to force the model to attend to visual content (e.g., mention audio may not belong to this person)
3. Consider unfreezing parts of the vision/audio encoder with a very small LR
4. Add regularization or label smoothing to prevent overconfident predictions
