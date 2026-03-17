# How the Parkinson's Prediction Model Works — Plain-Language Guide

This guide explains the algorithm so that **anyone reading it can follow the flow**: what goes in, what each part does, and what comes out. No deep math required.

---

## What the System Does (One Sentence)

**We take three sets of numbers (voice, handwriting, and walking) and combine them so the model can decide: “Does this person show signs of Parkinson’s?” and how confident it is.**

---

## The Big Picture

```
  VOICE (22 numbers)  ──┐
  HANDWRITING (10)    ──┼──► ENCODERS ──► COMBINE WITH ATTENTION ──► DECISION ──► "Parkinson's or not?" + confidence
  WALKING / GAIT (10) ──┘
```

- **Encoders**: Turn each set of numbers into a fixed “summary” (embedding).
- **Attention**: Decide how much to trust voice vs handwriting vs gait for this person.
- **Decision**: One final yes/no plus a confidence score.

Below we go through each part in order.

---

## 1. What Goes In (Inputs)

We don’t feed raw audio or video. We feed **already-computed features** (numbers):

| Source      | How many numbers? | What they represent (in simple terms) |
|------------|--------------------|----------------------------------------|
| **Speech** | 22                 | Voice stability, shakiness, pitch, etc. |
| **Handwriting** | 10             | Pressure, tremor, smoothness of writing |
| **Gait**   | 10                 | Step length, speed, symmetry of walking |

Each person (each sample) has three such lists. The model gets all three and uses them together.

---

## 2. Part One: The Encoder (SE-ResNet 1D CNN)

**In simple terms:** Each list of numbers (speech, handwriting, or gait) is turned into a single **summary vector of 64 numbers** that captures “patterns that matter for detection.” The same encoder design is used for all three; only the length of the input list changes (22 or 10).

### Step 2.1 — Treat the list as a 1D signal

- We take the list of 22 (or 10) numbers and treat it like a **one-dimensional signal** (e.g. one row of values).
- So the encoder always sees: **one channel**, length = 22 or 10.

### Step 2.2 — First layer (stem): find local patterns

- A **1D convolution** slides a small window (size 3) along this signal and produces 32 new “channels” (32 different ways of looking at the same length).
- **BatchNorm** and **ReLU** are applied so values stay stable and non-negative.
- **Meaning:** The model starts by detecting small local patterns (e.g. “these three features together look unusual”).

### Step 2.3 — Residual + SE blocks (where most of the work happens)

We use **two blocks** in a row. Each block does three things:

1. **Two convolutions**  
   - Again, small windows along the 1D signal.  
   - First block: 32 → 32 channels. Second block: 32 → 64 channels.  
   - **Meaning:** Build up more complex patterns (e.g. “tremor + slowness in this region”).

2. **Squeeze-and-Excitation (SE)**  
   - For each channel we **average over the whole length** → one number per channel.  
   - A small network (two layers) turns those numbers into **weights** between 0 and 1.  
   - We **multiply each channel by its weight**.  
   - **Meaning:** “Which of these pattern channels are most important for this sample?” The model learns to boost important ones and dampen less useful ones.

3. **Skip connection (residual)**  
   - We **add the block’s input** (or a 1×1 conv version of it if channel count changes) to the block’s output.  
   - **Meaning:** The block only has to learn *changes*; it doesn’t have to repeat what was already there. This makes training easier and more stable.

So inside the encoder: **stem conv → Block1 (conv, conv, SE, + skip) → Block2 (same) → then we summarize.**

### Step 2.4 — Summarize into one vector per modality

- **Global average pooling:** For each of the 64 channels we take the average over the whole length → 64 numbers.
- **Dropout** (for training) and a **linear layer** (64 → 64) give the final **embedding**.
- **Result:** One vector of **64 numbers** per modality (speech, handwriting, gait). That’s the “summary” the rest of the model uses.

---

## 3. Part Two: Attention Fusion

**In simple terms:** We have three summary vectors (one from voice, one from handwriting, one from gait). Instead of treating them equally, we **learn how much to trust each source** for this person, then form a single combined vector.

### Step 3.1 — Put the three summaries side by side

- We stack the three 64-d vectors into one array: **3 × 64** (or, with batch: **B × 3 × 64**).

### Step 3.2 — Score each modality

- A small **two-layer network** (64 → 32 → 1) takes each of the three 64-d vectors and outputs **one score** per modality.
- So we get three scores (one for speech, one for handwriting, one for gait).

### Step 3.3 — Turn scores into weights

- We apply **softmax** on those three scores so they become **weights that add up to 1** (e.g. 0.5, 0.3, 0.2).
- **Meaning:** “For this person, put 50% weight on voice, 30% on handwriting, 20% on gait” (numbers are learned, not fixed).

### Step 3.4 — Combine into one vector

- We take a **weighted sum** of the three 64-d vectors using these weights.
- **Result:** One **fused** 64-d vector that mixes all three modalities according to how useful each one is.

We also keep the three weights (e.g. `[w_speech, w_hand, w_gait]`) so we can show the user “the model relied most on voice for this prediction.”

---

### When only one modality is provided (voice OR handwriting OR gait)

In your setup the user may send **only one** type of input: only voice, or only handwriting, or only gait. There is no "combined" input; it is always **one of the three**. The model is unchanged. What happens:

1. **Missing modalities are zero-filled.** Only voice → handwriting and gait are zeros. Only handwriting → speech and gait are zeros. Only gait → speech and handwriting are zeros.
2. **All three still go through the network.** The encoder with real data produces a meaningful 64-d embedding; the encoders with zeros produce weak embeddings.
3. **Attention is still used.** The fusion layer scores all three embeddings and applies softmax. The model learns to put almost all weight on the modality that has real data (e.g. voice only → weights like `[0.95, 0.03, 0.02]`). So the fused vector is effectively that one modality's embedding.
4. **Result:** Prediction comes from the one modality; `attention_weights` show the user the model relied on it (e.g. 95% speech).

**Summary:** User sends one of {voice, handwriting, gait}. We zero-fill the other two and run the same model; attention down-weights the missing modalities and trusts the one provided. Attention is still used and useful.

---

## 4. Part Three: Dense Classifier

**In simple terms:** The fused 64-d vector is passed through a small **two-layer neural network** that outputs a single number (the “logit”). That number is then turned into a probability.

### What happens inside

- **Layer 1:** Linear (64 → 32), then ReLU, then Dropout.  
  - **Meaning:** Compress and non-linearly transform the fused summary.
- **Layer 2:** Linear (32 → 1).  
  - **Meaning:** Produce one raw score (logit).

No activation is applied to that last number inside the classifier. The **probability** is computed only when we apply the next step.

---

## 5. Part Four: Output

**In simple terms:** We convert the single raw score into a probability and a yes/no decision.

### Probability

- We apply the **sigmoid** function to the logit:  
  `probability = 1 / (1 + exp(-logit))`.  
- This gives a number between 0 and 1: **“How likely is it that this person has Parkinson’s?”**

### Decision

- If **probability ≥ 0.5** → predict **Parkinson’s (1)**.  
- If **probability < 0.5** → predict **Healthy (0)**.

### What the model actually returns

- **probability**: Confidence score (0–1).  
- **logit**: The raw score before sigmoid (for debugging or further use).  
- **attention_weights**: The three weights (speech, handwriting, gait) so you can see which modality was trusted most.  
- **Per-modality info**: Extra data (e.g. SE weights, last conv maps) used for explainability (e.g. Grad-CAM).

---

## 6. Full Flow (Summary)

| Stage | What happens | What you get |
|-------|----------------|-------------|
| **Input** | Three feature lists (22 + 10 + 10 numbers) | Ready for encoders |
| **Encoders** | Each list → 1D CNN (stem + 2 residual SE blocks) → pool + linear | Three 64-d vectors |
| **Attention** | Score each vector → softmax → weighted sum | One 64-d vector + 3 weights |
| **Classifier** | 64 → 32 → 1 (ReLU, dropout in between) | One logit |
| **Output** | Sigmoid(logit) and threshold at 0.5 | Probability + Parkinson’s / Healthy |

If you follow this order in the diagram or in the code, you’ll see exactly how the numbers move from **inputs** → **encoders** → **attention fusion** → **dense classifier** → **output**.

---

## 7. Diagram: Where to Look

- **ALGORITHM_INTERNALS.puml** — Same flow as above, in diagram form: inputs → encoder steps → attention → classifier → output.  
- **ALGORITHM_DIAGRAM.puml** — Higher-level view: “What is the goal?” and “What are the three main steps?”

Reading the diagram from top to bottom should match this document: first inputs, then encoder, then attention, then classifier, then final result.
