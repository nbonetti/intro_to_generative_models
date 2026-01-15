# Power Sampling: Latent Reasoning Extraction via MCMC

This repository contains the implementation and experimental evaluation of **Power Sampling**, a training-free inference method designed to elicit hidden reasoning capabilities from Large Language Models (LLMs).

##  Project Overview

Power Sampling leverages a **Metropolis-Hastings (MCMC)** transition kernel to sample from a "sharpened" distribution $p(x)^\alpha$. Unlike standard greedy decoding or temperature scaling, this method performs a global optimization of the sequence log-likelihood. This allows the model to escape local optima (greedy traps) and "correct" its own factual or logical errors during inference.



##  Project Structure

The project is organized into three main pillars corresponding to our research hypotheses (H1, H2, H3):

### 1. Toy World & Theoretical Validation (`/toy_model`)
A controlled synthetic environment used to validate the mathematical foundations of the algorithm.
* **Key Components:** Convergence diagnostics (KL Divergence, Total Variation distance), *pass@k* analysis.
* **Objective:** To prove that the MCMC kernel converges to the target distribution and to analyze the impact of block sizes on mixing efficiency.

### 2. Linearity Analysis & Benchmarks (`/experiments`)
Comparative tests on distributional properties and decoding performance.
* **`temperature vs power sampling` (H1):** Compares the impact of $\alpha$ (Power Sampling) vs. $T$ (Temperature Scaling). Demonstrates that $\alpha$ preserves the relative geometry of the distribution linearly.
* **`greedy vs powersampling` (H3):** A direct benchmark between standard Greedy Search and Power Sampling on logic-based prompts.

### 3. Core MCMC Optimizer (`mcmc`)
The main algorithmic engine applied to the **OPT-125M** model.
* **`compute_log_likelihood`:** Calculates the joint probability of a sequence using the model's cross-entropy loss.
* **`run_correction_tracker`:** Iterative Metropolis-Hastings implementation with step-by-step acceptance/rejection logging.
* **Key Hyperparameters:** $\alpha=16$ for strict filtering of high-probability reasoning chains and a `block_size=15` for effective state transitions.

