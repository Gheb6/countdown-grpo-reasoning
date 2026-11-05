# Reasoning with Verifiable Rewards (GRPO) on Countdown

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/)

**University Project - Human Language Technologies (HLT 2026)**  
*Prof. Gallicchio, Dr. Ceni*  
Group 2 - *Reasoning with Verifiable Rewards*

---

## 🎯 Panoramica del Progetto

Il progetto ha l'obiettivo di migliorare le capacità di ragionamento matematico di un Language Model di piccola taglia (0.5B / 1.5B parametri, nello specifico `Qwen/Qwen2.5-0.5B-Instruct` o `1.5B`) applicando l'algoritmo di **Reinforcement Learning GRPO (Group Relative Policy Optimization)** su un task interamente verificabile in modo deterministico: il puzzle aritmetico **Countdown**.

Il progetto confronta tre metodologie:
1. **Zero-Shot Diretto**: Il modello risponde direttamente con l'equazione finale senza passaggi intermedi.
2. **Chain-of-Thought (CoT) Prompting**: Il modello è spronato a ragionare passo-passo prima di fornire l'equazione finale (*Wei et al., 2022*).
3. **GRPO con Verifiable Rewards**: Il modello è addestrato con RL basato su reward deterministici (*DeepSeekMath / DeepSeek-R1-Zero*), premiando accuratezza e formattazione `<think>...</think><answer>...</answer>`.

---

## 📁 Struttura della Repository

```text
countdown-grpo-reasoning/
├── notebooks/
│   ├── 01_dataset_and_verifier.ipynb  # Verificatore AST, Unit Test e Generatore Puzzle
│   ├── 02_baseline_evaluation.ipynb   # Valutazione Metodi 1 (Zero-Shot) e 2 (CoT)
│   └── 03_grpo_training.ipynb        # Training RL con GRPOTrainer e Reward Verificabili
├── src/
│   ├── verifier.py                    # Classe CountdownVerifier (parsing AST sicuro)
│   ├── dataset_generator.py           # Generatore ricorsivo di puzzle garantiti solubili
│   └── rewards.py                     # Funzioni di reward per TRL GRPOTrainer
├── papers/                            # Paper di riferimento e Dispensa Fronte-Retro
│   ├── Dispensa_Estratti_Studio_FronteRetro.pdf
│   └── ...
├── data/                              # Train e Test set generati (jsonl)
└── results/                           # Metriche e log per la relazione
```

---

## 💻 Requisiti di Esecuzione: Google Colab (GPU vs Senza GPU)

| Notebook | Esecuzione su Colab | GPU Necessaria? | Note Hardware |
| :--- | :--- | :---: | :--- |
| **`01_dataset_and_verifier.ipynb`** | Colab / Locale | ❌ **No (CPU)** | Pura logica matematica e AST in Python. Termina in ~1 minuto. |
| **`02_baseline_evaluation.ipynb`** | Colab T4 | ⚡ **Consigliata (T4 Gratuita)** | Su GPU T4 valuta 50-100 problemi in ~2-3 min. Su CPU è possibile per 0.5B ma lenta. |
| **`03_grpo_training.ipynb`** | Colab T4 / Kaggle | ✅ **Sì (Obbligatoria T4)** | L'RL deve campionare $G=4$ output paralleli per prompt. Richiede circa 1 ora su T4 gratuita con LoRA. |

---

## 🚀 Come Eseguire su Google Colab

1. **Notebook 1 (Dataset)**:
   * Aprilo su Colab (anche con CPU standard).
   * Esegui tutte le celle per creare i file `train_countdown.jsonl` e `test_countdown.jsonl`.
2. **Notebook 2 (Baselines)**:
   * Vai su `Runtime` -> `Change runtime type` -> Seleziona **T4 GPU**.
   * Esegui il benchmark per registrare l'accuratezza di partenza di Zero-shot e CoT.
3. **Notebook 3 (GRPO Training)**:
   * Con **T4 GPU** attiva, avvia l'addestramento.
   * Il modello impara spontaneamente a riflettere tra `<think>...</think>` per massimizzare il reward di accuratezza.

---

## 📚 Riferimenti Bibliografici

* **DeepSeekMath**: Shao et al. (2024). *DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models*. arXiv:2402.03300.
* **DeepSeek-R1**: DeepSeek-AI (2025). *DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning*. arXiv:2501.12948.
* **Gandhi et al. (2025)**: *Cognitive Behaviors that Enable Self-Improving Reasoners, or, Four Habits of Highly Effective STaRs*. arXiv:2503.01307.
* **Wei et al. (2022)**: *Chain-of-Thought Prompting Elicits Reasoning in Large Language Models*. NeurIPS 2022.
