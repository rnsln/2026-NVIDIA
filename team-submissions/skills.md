### AI Agent Context & Constraints

## 1. CUDA-Q 0.13.0 Syntax & Backends
* **Targeting:** All quantum kernels must explicitly use `cudaq.set_target("nvidia")`.
* **Fallback:** Agents must implement a try-except block to fall back to the `qpp-cpu` target if GPUs are unavailable.
* **Optimization:** Implement `cudaq.optimize` passes to fuse single-qubit gates and reduce overall circuit depth.

## 2. High-Performance Computing (HPC) Scaling
* **Multi-GPU Strategy:** For N > 30, implement `nvidia-mgpu` logic with MPI decomposition.

## 3. Physics & Algorithmic Guardrails
* **Delta-Evaluation:** Energy calculations for the MTS phase must use O(N) Delta-Evaluation logic instead of O(N^2) recalculations.
* **Symmetry Invariants:** Every implementation must pass unit tests for Reversal Symmetry.
* **Transfer Learning:** Implement Zero-Padding for input vectors and Output Truncation (Masking) for GQE.
