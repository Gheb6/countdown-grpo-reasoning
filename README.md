# Reasoning with Verifiable Rewards (GRPO) on Countdown

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/)

**University Project - Human Language Technologies (HLT 2026)**  
*Prof. Gallicchio, Dr. Ceni*  
Group 2 - *Reasoning with Verifiable Rewards*

---

## 🎯 Project Overview

This project aims to enhance the mathematical reasoning capabilities of small language models (0.5B to 1.5B parameters, specifically `Qwen/Qwen2.5-0.5B-Instruct` or `1.5B`) using **Group Relative Policy Optimization (GRPO)** reinforcement learning on a verifiable task: the arithmetic game **Countdown**.

We compare three methodologies:
1. **Zero-Shot Direct**: The model answers directly with the final equation without explanation.
2. **Chain-of-Thought (CoT) Prompting**: The model is prompted to think step-by-step before producing the final answer (*Wei et al., 2022*).
3. **GRPO with Verifiable Rewards**: The model is fine-tuned via RL with rule-based deterministic rewards (*DeepSeekMath / DeepSeek-R1-Zero*), encouraging reasoning within `<think>...</think>` and equation output in `<answer>...</answer>`.

---

## 📁 Repository Structure

```text
countdown-grpo-reasoning/
├── notebooks/
│   ├── 01_dataset_and_verifier.ipynb  # AST Verifier, Unit Tests, and Puzzle Generator
│   ├── 02_baseline_evaluation.ipynb   # Method 1 (Zero-Shot) and Method 2 (CoT) Evaluation
│   └── 03_grpo_training.ipynb        # RL Training with GRPOTrainer and Verifiable Rewards
├── src/
│   ├── verifier.py                    # CountdownVerifier class (secure AST parsing)
│   ├── dataset_generator.py           # Recursive puzzle generator with guaranteed solvability
│   └── rewards.py                     # Reward functions for TRL GRPOTrainer
├── papers/                            # Reference papers and Duplex Study Reader
│   ├── Dispensa_Estratti_Studio_FronteRetro.pdf
│   └── ...
├── data/                              # Generated train and test datasets (JSONL)
│   ├── test_countdown.jsonl           # 200 benchmark test puzzles
│   └── train_countdown.jsonl          # 1,000 training puzzles
└── results/                           # Evaluation metrics and baselines
```

---

## 💻 Compute Requirements: Google Colab (GPU vs CPU)

| Notebook | Environment | GPU Needed? | Hardware Details |
| :--- | :--- | :---: | :--- |
| **`01_dataset_and_verifier.ipynb`** | Colab / Local | ❌ **No (CPU)** | Pure Python combinatorial logic and AST parsing. Completes in ~1 minute. |
| **`02_baseline_evaluation.ipynb`** | Colab T4 | ⚡ **Recommended (Free T4)** | On T4 GPU, evaluates 50–100 puzzles in **2–3 minutes**. Possible on CPU for 0.5B, but noticeably slower. |
| **`03_grpo_training.ipynb`** | Colab T4 / Kaggle | ✅ **Yes (Mandatory T4)** | RL requires sampling $G=4$ parallel completions per prompt. Takes ~1 hour on free T4 with LoRA. |

---

## 🚀 Execution Guide on Google Colab

1. **Notebook 1 (Dataset & Verifier)**:
   * Open in Colab (standard CPU).
   * Runs unit tests and exports `train_countdown.jsonl` (1,000 puzzles) and `test_countdown.jsonl` (200 puzzles).
2. **Notebook 2 (Baselines)**:
   * Set runtime to **T4 GPU** (`Runtime -> Change runtime type -> T4 GPU`).
   * Evaluates baseline Zero-Shot Direct vs Chain-of-Thought prompting and saves accuracy and token counts to `results/baselines.json`.
3. **Notebook 3 (GRPO Training)**:
   * Keeps **T4 GPU** enabled and launches GRPO RL training using Hugging Face TRL and LoRA (PEFT).
   * Produces the final comparative table (Zero-Shot vs CoT vs GRPO) for the university report.

---

## 📚 References

* **DeepSeekMath**: Shao et al. (2024). *DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models*. arXiv:2402.03300.
* **DeepSeek-R1**: DeepSeek-AI (2025). *DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning*. arXiv:2501.12948.
* **Gandhi et al. (2025)**: *Cognitive Behaviors that Enable Self-Improving Reasoners, or, Four Habits of Highly Effective STaRs*. arXiv:2503.01307.
* **Wei et al. (2022)**: *Chain-of-Thought Prompting Elicits Reasoning in Large Language Models*. NeurIPS 2022.

---

## ?? In-Depth Theory & Credit Assignment Guide

For a step-by-step mathematical breakdown of how GRPO assigns advantages and reinforces tokens without a Critic network, read our guide:  
?? **[Understanding Credit Assignment in GRPO (Step-by-Step Numerical Example)](docs/GRPO_CREDIT_ASSIGNMENT.md)**
