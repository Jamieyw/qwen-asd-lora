# Prompt Investigation: Why the Model Always Predicted NOT_SPEAKING

Date: 2026-04-15

## Problem

After training, the model predicted NOT_SPEAKING for every single frame across all 500 validation samples. The baseline (no LoRA adapter) also predicted all NOT_SPEAKING. Training loss looked good (0.09 → 0.03), but evaluation showed 0% recall for SPEAKING.

## Investigation: Interactive Testing on V100

We ran a series of tests on the base Qwen2.5-Omni-3B model (no fine-tuning) to isolate the cause.

### Test 1: Can the model see faces and hear audio at all?

```python
conv = [{"role": "user", "content": [
    {"type": "image", "image": imgs[0]},
    {"type": "audio", "audio": audio},
    {"type": "text", "text": "Describe this face image and what you hear in the audio."}
]}]
```

**Result:** "The image shows a man with a white complexion, wearing a white shirt, and has a black beard. He is speaking."

The model CAN see faces and hear audio. It even correctly identifies the person is speaking.

### Test 2: Simple ASD question with 1 image

```python
conv = [{"role": "user", "content": [
    {"type": "image", "image": imgs[0]},
    {"type": "audio", "audio": audio},
    {"type": "text", "text": "Is this person currently speaking? Answer with only SPEAKING or NOT_SPEAKING."}
]}]
```

**Result:** `SPEAKING` (correct)

### Test 3: How many images before it breaks?

| Images | Simple question (no system prompt) | Result |
|--------|-----------------------------------|--------|
| 1 | SPEAKING or NOT_SPEAKING? | SPEAKING |
| 2 | SPEAKING or NOT_SPEAKING? | SPEAKING |
| 3 | SPEAKING or NOT_SPEAKING? | SPEAKING |
| 5 | SPEAKING or NOT_SPEAKING? | SPEAKING |
| 6 | SPEAKING or NOT_SPEAKING? | SPEAKING |
| 7 | SPEAKING or NOT_SPEAKING? | SPEAKING |
| 8 | SPEAKING or NOT_SPEAKING? | SPEAKING |
| 9 | SPEAKING or NOT_SPEAKING? | SPEAKING |
| 10 | SPEAKING or NOT_SPEAKING? | SPEAKING |

All work with a simple question. The number of images is NOT the problem.

### Test 4: The exact format used in training

```python
# System prompt + per-frame output format + 10 images
system = "You are an active speaker detection system. Given sequential face images 
          and audio from a video, analyze each frame to determine whether the person 
          is speaking at that moment. Look at lip movements across frames and match 
          them with the audio signal."
question = "These are 10 sequential frames... For each frame, determine whether... 
            Output one line per frame in the format: Frame N: SPEAKING or NOT_SPEAKING"
```

**Result:** ALL NOT_SPEAKING for every frame.

### Test 5: Isolating the cause

| Setup | Result |
|-------|--------|
| System prompt + simple question | **SPEAKING** (correct) |
| No system prompt + per-frame format | Model refuses ("I cannot determine...") |
| System prompt + per-frame format | **All NOT_SPEAKING** (broken) |

## Root Cause

The **per-frame output format** is what breaks the model. When asked to output structured per-frame labels ("Frame 1: SPEAKING or NOT_SPEAKING, Frame 2: ..."), the model defaults to NOT_SPEAKING for every frame.

### The broken prompt (verbose, per-frame):

```
System: "You are an active speaker detection system. Given sequential face images 
and audio from a video, analyze each frame to determine whether the person is 
speaking at that moment. Look at lip movements across frames and match them with 
the audio signal."

User: "These are 10 sequential frames of a person's face extracted from a video, 
along with the corresponding audio. For each frame, determine whether this person 
is actively speaking at that moment by analyzing their lip movements and the audio. 
Output one line per frame in the format: Frame N: SPEAKING or NOT_SPEAKING"
```

### The working prompt (simple, direct):

```
System: "You are an active speaker detection system."

User: "Is this person currently speaking? Answer with only SPEAKING or NOT_SPEAKING."
```

## Why This Happens

The broken version overloads the model with complex instructions: "analyze each frame," "look at lip movements," "output one line per frame." The model gets confused by the detailed instructions and defaults to the safe answer (NOT_SPEAKING for everything).

The working version is simple and direct. The model already knows how to look at faces and listen to audio — it just needs a short, clear question.

## Key Lesson

With large language models, **simpler prompts often work better than detailed instructions**. The model is smart enough to figure out what to do without being told exactly how to think. Overloading with instructions can actually degrade performance by confusing the model's reasoning.

## Fix

Switch back to the simple single-answer format:
- System prompt: "You are an active speaker detection system."
- 10 images + audio
- Question: "Is this person currently speaking? Answer with only SPEAKING or NOT_SPEAKING."
- Answer: "SPEAKING" or "NOT_SPEAKING"

The base model already gets this right for clear cases. Fine-tuning with LoRA should improve accuracy on harder/ambiguous cases.
