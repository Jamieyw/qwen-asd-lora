# Logit Approach: Why We Use Binary Classification Loss Instead of Standard LM Loss

Date: 2026-04-15

## The Problem with Standard Causal LM Loss

With standard training, the model learns to predict the next token given all previous tokens. For the answer "SPEAKING":

```
Given: ...assistant\n  →  predict "S"     (the actual decision)
Given: ...assistant\nS  →  predict "P"    (trivial — what else follows "S"?)
Given: ...assistant\nSP →  predict "E"    (trivial)
Given: ...assistant\nSPE → predict "A"    (trivial)
...and so on for all 8 tokens
```

The loss is averaged across all 8 tokens. But tokens 2-8 are trivially predictable from the previous characters — once you see "S", "PEAKING" is basically guaranteed. So 7 out of 8 token losses are near-zero regardless of whether the model actually looked at the images or audio.

The only meaningful decision was the first token, but its gradient signal gets diluted by the 7 trivial tokens. Result: average loss is 0.001, the model thinks it's done, and the gradient is too weak to push the visual/audio processing to improve.

## What the Logit Approach Does

Instead of computing loss on all 8 characters, we only look at the decision point — the last position before the answer, where the model chooses between "SPEAKING" and "NOT":

```python
# Extract logits at the answer position
speaking_logit = last_logits[:, speaking_token_id]
not_speaking_logit = last_logits[:, not_token_id]

# Binary classification loss on just these two logits
class_logits = torch.stack([not_speaking_logit, speaking_logit], dim=1)
loss = cross_entropy(class_logits, class_labels)
```

Example:
```
logit for "SPEAKING" token = 2.5
logit for "NOT" token = -1.2
→ softmax → P(SPEAKING)=0.85, P(NOT_SPEAKING)=0.15
→ if true label is NOT_SPEAKING: loss = -log(0.15) = 1.9  ← strong signal!
```

That single comparison is the entire loss. No dilution from trivial follow-up tokens. If the model gets it wrong, the loss is high, and the full gradient flows back through the model to improve how it processes the images and audio.

## How This Helps at Inference

At inference, the model still generates text character by character — this doesn't change. But the first token it generates ("S" for SPEAKING vs "N" for NOT_SPEAKING) is now better informed because training pushed the model to actually attend to the visual/audio content when making that first-token decision. The rest of the characters follow automatically.

The logit approach doesn't change how the model writes the answer — it changes how well it learned to make the decision that determines which answer to write.

## Results Comparison

| Training approach | Epoch 1 loss | Issue |
|-------------------|-------------|-------|
| Standard causal LM loss | 0.001 | Near-zero because 7/8 tokens are trivially predictable; no real learning signal |
| Logit approach (binary CE) | 0.849 | Realistic loss; model must actually discriminate based on input content |
| Logit + label smoothing 0.1 | 0.849 | Same range, but prevents overconfident collapse over time |

## Implementation

The logit approach is implemented in both `train.py` and `train_v2.py` at the forward pass inside the training loop. The key code is in the `collate_fn` (which strips the assistant answer so the model doesn't see it) and the loss computation (which compares logits for two specific token IDs instead of computing loss on the full generated sequence).
