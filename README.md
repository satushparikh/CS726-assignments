# Medusa: Parallel Decoding for LLMs

Medusa introduces a novel approach to accelerate autoregressive language models by leveraging **speculative parallel decoding**. Unlike traditional models that generate text token by token, Medusa generates multiple tokens in parallel using specialized heads, significantly improving speed while maintaining reasonable accuracy. This is especially useful in latency-sensitive applications like real-time translation and conversational AI.

---

## 🧠 Understanding Medusa: Parallel Decoding for LLMs

### 🔍 What Is Medusa?

Medusa introduces a **speculative parallel decoding** method for accelerating autoregressive generation in large language models (LLMs). Instead of predicting one token at a time, Medusa leverages **multiple parallel prediction heads** to draft several future tokens, which are then verified by the main model.

This significantly reduces latency — ideal for real-time applications like translation and chat systems.

---

## 🏗️ Medusa Architecture

Medusa enhances standard Transformer models with:

- **Multiple Prediction Heads**: Draft tokens for future positions in parallel.
- **Verification Module**: Uses the base model to verify the correctness of predicted tokens.

These Medusa heads predict multiple tokens simultaneously. If their predictions match the base model’s outputs (within a confidence threshold), the tokens are accepted — skipping redundant forward passes.

---

## ⚙️ Medusa Decoding Workflow

The decoding algorithm follows this general procedure:

1. Generate candidate continuations using Medusa heads.
2. Rank candidates using a scoring function.
3. Verify candidates using the base model.
4. Accept sequences that meet a confidence threshold.
5. Repeat until an end-of-sequence token is generated or the length limit is reached.

This results in fewer base model evaluations, drastically improving decoding speed.

---

## 📊 Performance Summary

Medusa offers a trade-off between speed and accuracy. Here's a comparison of decoding configurations:

| Decoding Strategy       | BLEU | ROUGE-1 | ROUGE-2 | ROUGE-L | RTF  |
| ----------------------- | ---- | ------- | ------- | ------- | ---- |
| **Single Medusa Head**  | 0.29 | 0.40    | 0.15    | 0.32    | 0.07 |
| Beam 2, Medusa Heads 2  | 0.12 | 0.30    | 0.09    | 0.22    | 0.03 |
| Beam 2, Medusa Heads 5  | 0.12 | 0.30    | 0.09    | 0.22    | 0.03 |
| Beam 5, Medusa Heads 2  | 0.11 | 0.29    | 0.10    | 0.23    | 0.08 |
| Beam 5, Medusa Heads 5  | 0.11 | 0.29    | 0.10    | 0.23    | 0.07 |
| Beam 10, Medusa Heads 2 | 0.11 | 0.28    | 0.10    | 0.22    | 0.16 |
| Beam 10, Medusa Heads 5 | 0.11 | 0.28    | 0.10    | 0.22    | 0.17 |

> 📝 **RTF** = Real Time Factor (lower is better)

- **Best Speedup**: Beam=2, Heads=2 (RTF = 0.03)
- **Best Accuracy**: Single-Head decoding (BLEU = 0.29)

---

## 🧪 Key Innovations

- **Speculative Drafting**: Predict tokens in parallel.
- **Tree-Based Verification**: Organize draft sequences in a tree structure for efficient validation.
- **Beam Search Integration**: Allows exploration of multiple hypotheses for better accuracy.
- **Adaptive Control**: Dynamic head selection and parameter sharing for performance and memory efficiency.

---

## ⚖️ Trade-Offs

- **Speed vs Accuracy**: More heads = faster decoding but potentially less accurate predictions.
- **Beam Width**: Wider beams increase accuracy but also inference cost.
- **Verification Threshold**: Controls strictness in accepting predicted tokens.

---

## 🚀 How to Use Medusa

### Prerequisites

- **Llama Weights**: Obtain access to the model from [Llama's site](https://www.llama.com/docs/getting-the-models/hugging-face/).
- **IN22 Dataset**: Sign in to your Hugging Face account and accept the agreement [here](https://huggingface.co/datasets/ai4bharat/IN22-Gen).

### Setup

1. **Install Dependencies**:

   Create a conda environment and install dependencies:

   ```bash
   conda env create -f environment.yml
   ```

   Install specific PyTorch version:

   ```bash
   pip install torch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0 --index-url https://download.pytorch.org/whl/cu118
   ```

2. **Clone and Install Medusa**:

   ```bash
   conda activate cs726_a3
   git clone https://github.com/Darshan7575/Medusa.git
   cd Medusa
   pip install -e .
   ```

3. **Create Hugging Face Token**:

   [Follow these instructions](https://huggingface.co/docs/hub/en/security-tokens) to create a Hugging Face token to access model weights and datasets.

---

### Running Medusa

1. **Single Head Decoding**:

   ```bash
   CUDA_VISIBLE_DEVICES=0 python medusa.py --hf-token "<your_hf_token>" --decoding-strategy "single-head"
   ```

2. **Multiple Head Decoding**:

   ```bash
   CUDA_VISIBLE_DEVICES=0 python medusa.py --hf-token "<your_hf_token>" --decoding-strategy "multi-head" --beam-width <beam width> --use-no-medusa-heads <number of medusa heads>
   ```

---

## 🎯 Conclusion

Medusa decoding presents a promising technique for accelerating large language model inference by leveraging parallel token prediction. Despite minor accuracy trade-offs, its modular design allows for further refinements in:

- Improved head architectures to reduce prediction errors.
- More sophisticated verification mechanisms to enhance token acceptance.
- Adaptive parameter tuning for better speed-accuracy balance.

This technique is particularly beneficial for latency-sensitive applications, such as machine translation and dialogue systems, where fast response times outweigh minor reductions in output fidelity.

---
