# Power Sampling: Latent Reasoning Extraction via MCMC

This repository contains the implementation of **Power Sampling** on OPT-125M, demonstrating how to bypass greedy decoding limitations using a Metropolis-Hastings framework.

##  Core Implementations

### 1. Linearity Analysis (Gap Test)
* **File:** `powersamplingvs temperature`
* **Purpose:** Validates Hypothesis **H1**. It compares Temperature scaling vs. Power Sampling ($\alpha$).
* **Key Finding:** Confirms that Power Sampling scales the log-likelihood gap linearly, providing a more predictable sharpening tool than temperature.

### 2. Performance Benchmark (Greedy vs. Power)
* **File:** `greedy vs powersampling`
* **Purpose:** Validates Hypothesis **H3**. It benchmarks standard Greedy Search against the Power Sampling global search.
* **Key Finding:** Demonstrates that Greedy Search often falls into local optima (repetitive loops), while Power Sampling identifies significantly higher probability sequences.

### 3. MCMC Transition Kernel (H2 Optimization)
* **File:** ` mcmc`
* **Purpose:** Implements the core **Metropolis-Hastings** block-wise algorithm to optimize sequence coherence.
* **Features:** * `compute_log_likelihood`: Joint probability calculation.
    * `run_correction_tracker`: Iterative refinement using a sharpened target $p(x)^\alpha$.
    * **Acceptance Logic:** Uses $\alpha=16$ to strictly filter and "lock" high-probability reasoning steps.



