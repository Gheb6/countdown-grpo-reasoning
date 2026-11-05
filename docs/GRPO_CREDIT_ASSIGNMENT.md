# Understanding Credit Assignment in GRPO (Group Relative Policy Optimization)

**Theoretical Documentation - HLT 2026 University Project**  
*References: DeepSeekMath (Shao et al., 2024), DeepSeek-R1 (DeepSeek-AI, 2025)*

---

## 1. The Core Problem: Why PPO is Too Heavy for Small Resources

In standard **Proximal Policy Optimization (PPO)**:
* Training requires an **Actor network** (the LLM being fine-tuned) and a **Critic network** (a Value Model of roughly the same size as the Actor).
* The Critic estimates the expected return $V(s_t)$ at every token step to calculate Generalized Advantage Estimation (GAE).
* **The bottleneck**: Keeping two full LLMs (Actor + Critic) plus reference models and optimizer states in GPU memory causes severe **CUDA Out-Of-Memory (OOM)** errors on free tiers like Kaggle or Google Colab (15–16 GB VRAM).

---

## 2. The GRPO Innovation: Zero Critic, Group Relative Advantage

**Group Relative Policy Optimization (GRPO)** eliminates the Critic network entirely:
1. For each input prompt $q$, the policy generates a **group** of $G$ independent completions: $\{o_1, o_2, \dots, o_G\}$.
2. Each completion is scored by our deterministic AST reward functions: $\{r_1, r_2, \dots, r_G\}$.
3. The baseline is computed directly from the **mean ($\mu$)** and **standard deviation ($\sigma$)** of the group itself:
   $$A_i = \frac{r_i - \text{mean}(\{r_1, \dots, r_G\})}{\text{std}(\{r_1, \dots, r_G\})}$$
4. The policy parameters $\theta$ are updated using this standardized relative advantage $A_i$.

---

## 3. Step-by-Step Numerical Example (Countdown Arithmetic)

Suppose the model receives the following prompt from our training dataset:
> **Prompt $q$**: *"Using the numbers [7, 35, 12, 99], create an equation that equals 60."*

### Step A: Group Rollout Sampling ($G = 4$)
The model policy $\pi_\theta$ generates 4 candidate outputs:

| Attempt | Generated Completion | Description |
| :---: | :--- | :--- |
| **$o_1$** | `<answer>10 + 50</answer>` | **Exploit / Hallucination**: Invented numbers 10 and 50 not in the prompt. |
| **$o_2$** | `The answer is 60 because 12 times 5 is 60.` | **Format Failure**: Ignored `<think>` and `<answer>` XML tags. |
| **$o_3$** | `<think>Let's try: 99 - 35 = 64.</think><answer>99 - 35</answer>` | **Arithmetic Error**: Valid format and valid numbers, but $64 \neq 60$. |
| **$o_4$** | `<think>12 * 35 = 420. Then 420 / 7 = 60!</think><answer>(12 * 35) / 7</answer>` | **Exact Match**: Valid format, valid numbers, exact target $60$. |

---

### Step B: Deterministic Verifier Reward Evaluation ($r_i$)
Our AST verifier scores each output independently:
* **Format Reward** ($r_{\text{format}}$): $+0.2$ for `<think>...</think><answer>...</answer>`
* **Validity Reward** ($r_{\text{valid}}$): $+0.1$ for valid arithmetic AST using only provided numbers
* **Accuracy Reward** ($r_{\text{acc}}$): $+1.0$ if evaluated equation equals target $60$

Total Reward: $r_i = r_{\text{format}} + r_{\text{valid}} + r_{\text{acc}}$

| Attempt | $r_{\text{format}}$ | $r_{\text{valid}}$ | $r_{\text{acc}}$ | Total Raw Reward ($r_i$) |
| :---: | :---: | :---: | :---: | :---: |
| **$o_1$** | $0.0$ | $0.0$ | $0.0$ | **$0.0$** |
| **$o_2$** | $0.0$ | $0.0$ | $0.0$ | **$0.0$** |
| **$o_3$** | $0.2$ | $0.1$ | $0.0$ | **$0.3$** |
| **$o_4$** | $0.2$ | $0.1$ | $1.0$ | **$1.3$** |

Raw rewards for the group: $\mathbf{r} = [0.0, 0.0, 0.3, 1.3]$

---

### Step C: Group Advantage Whitening ($A_i$)
We compute the sample statistics for the group:
1. **Group Mean**:
   $$\mu = \frac{0.0 + 0.0 + 0.3 + 1.3}{4} = \mathbf{0.40}$$
2. **Group Standard Deviation**:
   $$\sigma = \sqrt{\frac{1}{4} \sum_{i=1}^{4} (r_i - 0.40)^2} \approx \mathbf{0.543}$$
3. **Normalized Advantage $A_i = \frac{r_i - \mu}{\sigma}$**:

| Attempt | Raw Reward ($r_i$) | Relative Advantage ($A_i$) | Gradient Update Direction |
| :---: | :---: | :---: | :--- |
| **$o_1$** | $0.0$ | $\frac{0.0 - 0.40}{0.543} = \mathbf{-0.74}$ | 🔻 **Suppressed**: Tokens that hallucinated numbers are penalized. |
| **$o_2$** | $0.0$ | $\frac{0.0 - 0.40}{0.543} = \mathbf{-0.74}$ | 🔻 **Suppressed**: Missing XML tags penalized. |
| **$o_3$** | $0.3$ | $\frac{0.3 - 0.40}{0.543} = \mathbf{-0.18}$ | 🔻 **Slightly Penalized**: Correct format, but wrong target. |
| **$o_4$** | $1.3$ | $\frac{1.3 - 0.40}{0.543} = \mathbf{+1.66}$ | 🔺 **Strongly Reinforced**: Token trajectory for trial-and-error reasoning is boosted. |

---

### Step D: Policy Gradient Update
The objective function maximizes:
$$\mathcal{L}_{\text{GRPO}}(\theta) = \mathbb{E} \left[ \frac{1}{G} \sum_{i=1}^{G} \min \left( \frac{\pi_\theta(o_i|q)}{\pi_{\text{old}}(o_i|q)} A_i, \; \text{clip}\left(\frac{\pi_\theta(o_i|q)}{\pi_{\text{old}}(o_i|q)}, 1-\varepsilon, 1+\varepsilon\right) A_i \right) - \beta \, D_{\text{KL}}(\pi_\theta \parallel \pi_{\text{ref}}) \right]$$

* Tokens generated in **$o_4$** have $A_4 > 0$: their log-probabilities $\log \pi_\theta(o_4|q)$ are pushed upwards.
* Tokens in **$o_1, o_2, o_3$** have $A_i < 0$: their log-probabilities are pushed downwards.
* The **KL penalty** term ensures the model's vocabulary and coherent language abilities do not collapse during RL.

---

## 4. Why Does the Verifier Return 0.0 Instead of Crashing or Giving -100?

1. **No Runtime Crashes**: If invalid syntax raised an unhandled Python exception, PyTorch training would abort. The verifier catches exceptions gracefully and returns `valid_syntax = False`.
2. **Why Not a Large Negative Penalty (e.g. -100)?**:
   * If an exploit returned $-100$, the group mean and variance would become severely skewed (e.g., $\mu \approx -25$).
   * This would create massive gradients, destabilizing training and causing policy divergence.
   * In GRPO, **getting $0.0$ is already an automatic relative penalty** whenever another rollout in the group achieves $0.3$ or $1.3$.
