# 🚀 Hybrid Generative Quantum-Enhanced Memetic Tabu Search for LABS (GQE-MTS)

## Team QAT — iQuHACK 2026
### Eren Aslan, Chang Jen Yu, Huseyin Umut Isik, Hatice Boyar, Ilayda Dilek

🎥 **Project Demo Video:** [Watch here](https://drive.google.com/file/d/1z29MC4YeSgWLd6gvUHuDGQTEys9I7PIe/view?usp=sharing)  
🌐 **Live Website:** *Coming Soon!*

---

## 🧠 The Challenge: Barren Plateaus in LABS
Standard Variational Quantum Eigensolvers (VQE) typically fail on the **Low Autocorrelation Binary Sequences (LABS)** problem due to **Barren Plateaus**. The energy landscape is exponentially flat, causing gradients to vanish ($O(e^{-\alpha N})$), meaning simple gradient descent gets stuck immediately.

## 🏗️ Our Architecture: Hybrid GQE-MTS
Our approach prioritizes learning the **"Genetic Code"** (structural patterns) of the solution at a small scale ($N=10$) rather than brute-forcing large-scale optimization ($N=40$).

### 1. GQE Training (The Discovery Phase)
* **Target**: Learn optimal $Y$-operators, also known as Geometric Kernels.
* **Operator Pool**: We generate a custom pool using **2-body ($G_2$)** and **4-body ($G_4$)** interactions.
* **Optimization**: A **Transformer-based** Generative Quantum Eigensolver (GQE) iteratively updates gradients to converge on an optimal Ansatz.



### 2. Transfer Learning via Translational Symmetry
**Why does training on $N=10$ work for $N=40$?** Because the physics of the LABS problem is **Translationally Invariant** in the bulk.
* **Tiling (Copy/Paste)**: We take the learned kernels from the $N=10$ training and "tile" them across the larger $N=40$ lattice.
* **Efficiency**: This avoids the exponential $O(N^3)$ cost of large-scale optimization while constructing a high-quality Ansatz.

### 3. Quantum Seeding & Sampling
* We sample **1,000 shots** from the tiled Quantum Circuit using CUDA-Q.
* **Basin Identification**: We select the top 20 candidate sequences, referred to as **"Golden Seeds"**, which represent the most promising regions of the search space.

### 4. MTS Refinement (The Finish Line)
* **Classical MTS**: A GPU-accelerated Memetic Tabu Search performs a local search.
* **Final Output**: The search refines the Golden Seeds to find the **Ground Truth** optimal LABS sequence.

---

## ⚡ Acceleration & "Zombie" Prevention

### 🚀 CuPy Supercharge
We swapped standard NumPy for **CuPy** to handle the heavy lifting in our classical refinement phase.
* **100X Speedup**: By moving array operations directly to the GPU, we achieved order-of-magnitude speedups for large matrix operations and element-wise flips.
* **Zero Overhead**: Leveraging `cp.asarray()` for high-speed data transfer from host to device memory.

### 🧟 Zombie Process Prevention
Large-scale optimization tasks can leave lingering processes that clutter the system's process table. Our pipeline includes built-in **Zombie Prevention**:
* **Automatic Reaping**: The parent process uses `os.waitpid()` to ensure child processes are properly cleaned up after execution.
* **Signal Handling**: We implement `SIGCHLD` handlers to catch terminated children immediately, preventing them from becoming "defunct" placeholders that could block new process creation.
* **System Stability**: This ensures that even during massive parallel searchers, the process table remains finite and responsive.

---

## 📊 Best Result Analysis
Our most significant breakthrough was achieving near-instant convergence for **$N=30$**.
* **The "Sweet Spot"**: Using the `nvidia (mgpu)` backend combined with CuPy-accelerated MTS, we reduced the total execution time (Phase 2+3) to **under 1 second**.
* **Precision vs. Performance**: Our testing revealed that **FP32** (Single Precision) provided sufficient accuracy for the LABS energy landscape while running **2X faster** than FP64.
* **Golden Seed Quality**: The quantum-generated seeds were significantly closer to the global optima than random starts, proving that the **GQE Genetic Code** successfully captured the problem's underlying physics.

---

## 🛠️ Tech Stack

* **Quantum Core**: `CUDA-Q` (NVIDIA backend) for statevector simulation.
* **Classical Core**: `CuPy` for GPU-accelerated array computing and parallel search.
* **Hardware Optimization**: Multi-GPU (`mgpu`) and Multi-QPU (`mqpu`) support.
* **Verification**: Automated `tests.py` suite for symmetry checks and ground-truth calibration.

---

## 📁 Repository Structure
```text
.
├── 01_quantum_enhanced_optimization_LABS.ipynb   # Main hybrid pipeline
├── tests.py                                      # Correctness & calibration tests
├── cudaq_solvers/                                # GQE algorithm implementation
└── README.md                                     # Project documentation
