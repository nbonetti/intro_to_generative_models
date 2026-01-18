# Power Sampling: Latent Reasoning Extraction via MCMC

This repository contains the implementation and experimental evaluation of **Power Sampling**, a training-free inference method designed to elicit hidden reasoning capabilities from Large Language Models (LLMs). By leveraging a **Metropolis-Hastings (MCMC)** framework, Power Sampling enables global optimization of sequence log-likelihood, allowing models to escape local optima and improve reasoning quality without additional training.

---

## Core Implementations

### 1. Linearity Analysis (Gap Test)
* **File:** `powersampling_vs_temperature`
* **Purpose:** Validates Hypothesis **H1** by comparing Temperature Scaling with Power Sampling ($\alpha$).
* **Key Finding:** Power Sampling scales the log-likelihood gap linearly, offering a more predictable sharpening mechanism than temperature adjustments.

### 2. Performance Benchmark (Greedy vs. Power Sampling)
* **File:** `greedy_vs_powersampling`
* **Purpose:** Validates Hypothesis **H3** by benchmarking Greedy Search against Power Sampling.
* **Key Finding:** Greedy Search often falls into repetitive loops (local optima), while Power Sampling identifies higher-probability sequences, improving overall performance.

### 3. MCMC Transition Kernel (H2 Optimization)
* **File:** `mcmc`
* **Purpose:** Implements the **Metropolis-Hastings** algorithm to optimize sequence coherence.
* **Key Features:**
  - `compute_log_likelihood`: Calculates joint probabilities for sequences.
  - `run_correction_tracker`: Iteratively refines sequences using a sharpened target distribution $p(x)^\alpha$.
  - **Acceptance Logic:** Uses $\alpha=16$ to prioritize high-probability reasoning steps.

---

# Reasoning with Sampling: Your Base Model is Smarter Than You Think 

This paper argues that significant reasoning improvements can be achieved in LLMs without additional training, by employing an inference strategy based on **Power Sampling**.

## Context

Language models generate responses token by token. To improve reasoning tasks (e.g., math, code, science), reinforcement learning (RL) is often used to fine-tune models. However, the paper hypothesizes that RL primarily **reweights** existing outputs rather than introducing new capabilities. Power Sampling achieves similar gains by targeting a sharpened distribution:

$$
p(\text{response})^\alpha \quad \text{with } \alpha > 1
$$

This approach amplifies the probability of plausible sequences while suppressing less likely ones.

### Why Not Just Lower the Temperature?

Unlike temperature scaling, which operates locally (token-level), Power Sampling modifies the global sequence distribution. The paper demonstrates that these two methods yield distinct outcomes, with Power Sampling offering better control over sequence plausibility.

---

## Methodology: Metropolis-Hastings (MCMC) for Sampling $$p^\alpha$$

Direct sampling from $$p^\alpha$$ is computationally intractable. The authors use **Metropolis-Hastings (MCMC)** to approximate this distribution.

### Key Steps:
1. Start with a current sequence.
2. Propose a candidate sequence.
3. Accept or reject the candidate based on Metropolis-Hastings rules.
4. Repeat for multiple iterations to refine the sequence.

### Autoregressive Adaptation:
Instead of resampling entire sequences, the method:
- Selects a position in the sequence.
- Resamples a suffix or block.
- Applies the Metropolis-Hastings acceptance ratio.

---

## Evaluation

### Benchmarks:
- **Datasets:** MATH500, HumanEval, GPQA Diamond, AlpacaEval 2.0.
- **Models:** Qwen2.5-Math-7B, Qwen2.5-7B, Phi-3.5-mini-instruct.
- **Baseline:** GRPO (Reinforcement Learning post-training on MATH).

### Results:
Power Sampling consistently improves base model performance, often surpassing low-temperature baselines and approaching or exceeding GRPO in some cases.

#### Highlights (Table 1):
- **Qwen2.5-Math-7B**
  - MATH500: 0.496 → 0.748 (GRPO: 0.785)
  - HumanEval: 0.329 → 0.573 (GRPO: 0.537)
- **Phi-3.5-mini-instruct**
  - HumanEval: 0.213 → 0.732 (GRPO: 0.134)

---

## Analysis

- **Concentration vs. RL:** Power Sampling focuses on high-likelihood regions, similar to RL, but without additional training.
- **Solution Lengths:** Comparable to GRPO on MATH500, without explicitly optimizing for length.
- **Diversity (pass@k):** Maintains diversity at higher k values, outperforming GRPO in some scenarios.
- **Hyperparameters:** Performance stabilizes with intermediate $$\alpha$$ values and a few MCMC iterations.

---

## Implications and Limitations

- **Test-Time Scaling:** Shifts some RL benefits to inference time, requiring additional computational resources during response generation.
- **Inference Cost:** Higher computational cost compared to standard decoding, but significantly lower than RL training.
- **Scope:** Demonstrates that RL is not always necessary for reasoning improvements, but does not claim to replace RL entirely.

---

## Project Structure

### 1. Toy World & Theoretical Validation (`/toy_model`)
A controlled environment to validate the algorithm's mathematical foundations.
* **Key Components:** Convergence diagnostics (KL Divergence, Total Variation distance), *pass@k* analysis.
* **Objective:** Prove MCMC kernel convergence and analyze block size impact on mixing efficiency.

### 2. Linearity Analysis & Benchmarks (`/experiments`)
Comparative tests on distributional properties and decoding performance.
* **`temperature_vs_power_sampling` (H1):** Compares $\alpha$ (Power Sampling) with $T$ (Temperature Scaling).
* **`greedy_vs_powersampling` (H3):** Benchmarks Greedy Search against Power Sampling on logic-based prompts.

### 3. Core MCMC Optimizer (`mcmc`)
The main algorithmic engine applied to the **OPT-125M** model.
* **`compute_log_likelihood`:** Calculates sequence probabilities.
* **`run_correction_tracker`:** Implements Metropolis-Hastings with detailed logging.
* **Key Hyperparameters:** $\alpha=16$, `block_size=15`.

---

Power Sampling provides a novel, training-free approach to enhance reasoning in LLMs, bridging the gap between standard decoding and reinforcement learning.
