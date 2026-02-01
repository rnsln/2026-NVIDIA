# 🚀 Hybrid Generative Quantum-Enhanced Memetic Tabu Search for LABS (GQE-MTS)

## Team QAT — iQuHACK 2026

### Eren Aslan, Chang Jen Yu, Huseyin Umut Isik, Hatice Boyar, Ilayda Dilek

🎥 **Project Demo Video:**  
OUR WEBSITE IS GOING TO BE LIVE SOON! FOR NOW, JUST CHECK OUT THE VIDEO!
https://drive.google.com/file/d/1z29MC4YeSgWLd6gvUHuDGQTEys9I7PIe/view?usp=sharing

This repository contains our iQuHACK 2026 project on solving the **Low Autocorrelation Binary Sequences (LABS)** problem using a **hybrid quantum–classical optimization pipeline**, combining:

- **Generative Quantum Eigensolvers (GQE)**
- **GPU-accelerated Classical Memetic Tabu Search (MTS)**
- **Rigorous test-driven verification** to avoid silent errors

---

## 🧠 What is the LABS Problem?

The **LABS problem** asks for binary sequences whose autocorrelation energy is minimized.  
It is a **hard combinatorial optimization problem** with applications in:

- Communications
- Radar and signal processing
- Coding theory
- Physics-inspired spin models

As the sequence length increases, brute-force search becomes infeasible — making LABS an ideal benchmark for **hybrid quantum–classical algorithms**.

---

## ✨ Our Idea (High-Level)

We combine **global quantum exploration** with **local classical refinement**.

### 🔹 Phase 1 — Generative Quantum Optimization (GQE)

- Encode LABS energy as a Hamiltonian
- Use problem-aware **2-body and 4-body operators**
- Apply a **Generative Quantum Eigensolver** to concentrate probability mass on low-energy states
- Accelerate learning via **transfer learning** from smaller problem sizes

---

### 🔹 Phase 2 — Bridging & Filtering

- Sample bitstrings from the optimized quantum circuit
- Compute exact LABS energy classically
- Select a small set of elite **“Golden Seeds”**

---

### 🔹 Phase 3 — Classical Memetic Tabu Search (MTS)

- Inject Golden Seeds into a GPU-accelerated MTS
- Perform parallel neighborhood exploration with tabu memory
- Rapidly converge to near-optimal or optimal solutions

📌 **Key idea:**  
Quantum helps us start in the right region of the search space; classical heuristics finish the job efficiently.

---

## ⚡ Acceleration Strategy

### Quantum (CUDA-Q)

- NVIDIA backend (**CUDA-Q 0.13.0**)
- Statevector simulation
- Optional multi-GPU execution for large \( N \)
- Circuit-level optimizations using `cudaq.optimize`

### Classical (GPU-accelerated MTS)

- Delta-energy evaluation reduces complexity from O(N^2) to O(N)
- CuPy-based CUDA kernels evaluate all single-bit flips in parallel
- Kernel granularity tuned for high occupancy and low launch overhead

---

## 🧪 Verification & Guardrails

We take correctness seriously.

### ✔ What we test (`tests.py`)

- Symmetry checks (reversal, negation invariance)
- Ground-truth calibration:
  - \( N = 7 \Rightarrow E = 3 \) (exact)
  - \( N = 20 \Rightarrow E = 16 \) (known benchmark)
- Energy bounds and sanity checks
- End-to-end pipeline validation

---

### 🛑 AI Hallucination Protection

All code changes must pass `tests.py` **before** being run on expensive GPU resources.  
Broken or unstable code is rejected early to prevent wasted compute credits.

---

## 📊 Success Metrics

We evaluate performance empirically against known benchmarks:

- 🎯 Reach the known global optimum for \( N = 27 \) within time limits
- ⚡ Achieve significant GPU speedup compared to a CPU baseline
- 📈 Maintain **>90% approximation quality** for larger \( N \) where the optimum is unknown

---

## 📁 Repository Structure

```text
.
├── 01_quantum_enhanced_optimization_LABS.ipynb   # Main experiment notebook
├── tests.py                                      # Automated correctness tests
├── README.md                                     # Project overview
