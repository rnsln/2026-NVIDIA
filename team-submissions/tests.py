import cudaq
import cudaq_solvers as solvers
from cudaq import spin
from cudaq_solvers.gqe_algorithm.gqe import get_default_config

import math
import numpy as np
import matplotlib.pyplot as plt
import random
import time  # <--- Imported for timing
from typing import List, Tuple, Optional

# ==============================================================================
# 0. Backend Configuration
# ==============================================================================
def setup_backend():
    """Configures the quantum backend (NVIDIA GPU or CPU fallback)."""
    try:
        cudaq.set_target("nvidia")
        print("[Backend] NVIDIA GPU configured.")
    except:
        cudaq.set_target('qpp-cpu')
        print("[Backend] Fallback to CPU.")

setup_backend()

# ==============================================================================
# 1. Physics & Math Utilities (LABS Core)
# ==============================================================================
class LABSPhysics:
    """Handles the physical definitions and energy calculations for the LABS problem."""
    
    @staticmethod
    def get_interactions(N: int):
        """
        Generates interaction indices for the Hamiltonian.
        Returns:
            G2: List of [i, j] for 2-body terms.
            G4: List of [i, j, k, l] for 4-body terms.
        """
        G2 = [[i-1, i+k-1] for i in range(1, N) for k in range(1, (N-i)//2 + 1)]
        G4 = []
        for i in range(1, N-1):
            for t in range(1, (N-i-1)//2 + 1):
                for k in range(t+1, N-i-t+1):
                    G4.append([i-1, i+t-1, i+k-1, i+k+t-1])
        return G2, G4

    @staticmethod
    def build_hamiltonian(N: int):
        """Constructs the Pauli-Z Hamiltonian for the given system size N."""
        G2, G4 = LABSPhysics.get_interactions(N)
        ham = 0
        for i, j in G2: ham += spin.z(i) * spin.z(j)
        for i, j, k, l in G4: ham += spin.z(i) * spin.z(j) * spin.z(k) * spin.z(l)
        return ham

    @staticmethod
    def compute_energy(s: np.ndarray) -> int:
        """
        Computes the classical energy (Merit Factor metric).
        E = Sum of squared autocorrelation coefficients (excluding lag 0).
        """
        N = len(s)
        E = 0
        for k in range(1, N):
            Ck = np.sum(s[:N-k] * s[k:])
            E += Ck**2
        return int(E)

# ==============================================================================
# 2. Quantum Engine: GQE & Transfer Learning
# ==============================================================================
class QuantumEngine:
    """Manages quantum operator pools and the transfer learning (tiling) logic."""

    @staticmethod
    def generate_pool(N: int, depth=4):
        """
        Generates a pool of operators tagged with geometric metadata.
        Metadata is used to identify the 'shape' (gaps) of the operator for tiling.
        """
        G2, G4 = LABSPhysics.get_interactions(N)
        params = [math.pi / (2**i) for i in range(depth)]
        pool, metadata = [], []

        # Helper to add op
        def add_op(base_op, meta_dict):
            op = cudaq.SpinOperator(base_op)
            for p in params:
                pool.extend([p * op, -p * op])
                # Store coefficients and geometry type
                metadata.extend([{**meta_dict, 'coeff': p}, {**meta_dict, 'coeff': -p}])

        # 2-Body Operators
        for i, j in G2:
            add_op(spin.z(i)*spin.z(j), {'type': '2-body', 'gap': j-i})
        # 4-Body Operators
        for i, j, k, l in G4:
            add_op(spin.z(i)*spin.z(j)*spin.z(k)*spin.z(l), {'type': '4-body', 'gaps': (j-i, k-j, l-k)})
            
        return pool, metadata

    @staticmethod
    def transfer_features(best_indices, meta_src, N_target):
        """
        Implements Transfer Learning via Feature Tiling.
        Takes optimal features from a small system and tiles them across the larger target system.
        """
        coeffs, words = [], []
        for idx in best_indices:
            feat = meta_src[idx]
            theta = feat['coeff']
            
            if feat['type'] == '2-body':
                gap = feat['gap']
                # Sliding window: apply the gap to all valid positions
                for i in range(N_target - gap):
                    s = ['I'] * N_target
                    s[i] = s[i+gap] = 'Z'
                    words.append(cudaq.pauli_word("".join(s)))
                    coeffs.append(theta)
            
            elif feat['type'] == '4-body':
                g1, g2, g3 = feat['gaps']
                span = g1 + g2 + g3
                # Sliding window for 4-body terms
                for i in range(N_target - span):
                    s = ['I'] * N_target
                    s[i] = s[i+g1] = s[i+g1+g2] = s[i+g1+g2+g3] = 'Z'
                    words.append(cudaq.pauli_word("".join(s)))
                    coeffs.append(theta)
        return coeffs, words

# --- CUDA-Q Kernels (Global scope required for JIT compilation) ---
@cudaq.kernel
def kernel_train(n: int, coeffs: list[float], words: list[cudaq.pauli_word]):
    """Variational kernel for training on small N."""
    q = cudaq.qvector(n)
    h(q) # Initialize in superposition
    for i in range(len(coeffs)):
        exp_pauli(coeffs[i], q, words[i])

@cudaq.kernel
def kernel_target(n: int, coeffs: list[float], words: list[cudaq.pauli_word]):
    """Inference kernel for large N (constructed via tiling)."""
    q = cudaq.qvector(n)
    h(q)
    for i in range(len(coeffs)):
        exp_pauli(coeffs[i], q, words[i])

# ==============================================================================
# 3. Classical Optimizer: Memetic Tabu Search (MTS)
# ==============================================================================
class MTSOptimizer:
    """
    Implements Memetic Tabu Search: 
    A hybrid of Evolutionary Algorithm (Global) and Tabu Search (Local).
    """
    def __init__(self, N, pop_size=20, p_mut=0.05, tabu_tenure=7):
        self.N = N
        self.pop_size = pop_size
        self.p_mut = p_mut
        self.tabu_tenure = tabu_tenure

    def _tabu_search(self, s, max_iter=100):
        """Performs local refinement using Tabu Search."""
        N = len(s)
        curr, best = s.copy(), s.copy()
        curr_E, best_E = LABSPhysics.compute_energy(curr), LABSPhysics.compute_energy(curr)
        tabu_list = np.zeros(N, dtype=int)

        for it in range(max_iter):
            best_neigh_E = float('inf')
            move_idx = -1
            
            # Evaluate all 1-bit flip neighbors
            for i in range(N):
                # Temporary flip to calculate energy
                neigh = curr.copy()
                neigh[i] *= -1
                e = LABSPhysics.compute_energy(neigh)

                # Check Tabu and Aspiration criteria
                is_tabu = tabu_list[i] > it
                if (not is_tabu or e < best_E) and e < best_neigh_E:
                    best_neigh_E = e
                    move_idx = i
            
            if move_idx == -1: break # No valid moves
            
            # Make the move
            curr[move_idx] *= -1
            curr_E = best_neigh_E
            tabu_list[move_idx] = it + self.tabu_tenure
            
            # Update global best
            if curr_E < best_E:
                best, best_E = curr.copy(), curr_E
                
        return best, best_E

    def run(self, max_gens=100, initial_pop=None, verbose=True):
        """Runs the main evolutionary loop."""
        # Initialize population
        if initial_pop:
            pop = [np.array(s) for s in initial_pop[:self.pop_size]]
            # Fill remaining slots with random sequences if needed
            while len(pop) < self.pop_size: pop.append(np.random.choice([-1, 1], self.N))
        else:
            pop = [np.random.choice([-1, 1], self.N) for _ in range(self.pop_size)]
            
        energies = [LABSPhysics.compute_energy(s) for s in pop]
        best_idx = np.argmin(energies)
        global_best, global_min_E = pop[best_idx].copy(), energies[best_idx]
        history = [global_min_E]

        for gen in range(max_gens):
            # 1. Selection & Crossover
            p1, p2 = random.sample(pop, 2)
            k = random.randint(1, self.N - 1)
            child = np.concatenate([p1[:k], p2[k:]])
            
            # 2. Mutation
            for i in range(self.N):
                if random.random() < self.p_mut: child[i] *= -1
            
            # 3. Local Search (Memetic Step)
            child, child_E = self._tabu_search(child)
            
            # 4. Replacement (Survival of the Fittest)
            worst_idx = np.argmax(energies)
            if child_E < energies[worst_idx]:
                pop[worst_idx] = child
                energies[worst_idx] = child_E
                
            if child_E < global_min_E:
                global_best, global_min_E = child.copy(), child_E
                if verbose and gen % 10 == 0: print(f"Gen {gen}: New Best E = {global_min_E}")
            
            history.append(global_min_E)
            
        return global_best, global_min_E, pop, history

# ==============================================================================
# 4. Utilities & Visualization
# ==============================================================================
def visualize_comparison(hist_rand, hist_quant, N):
    """Plots the convergence history of Random vs Quantum initialization."""
    plt.figure(figsize=(10, 6))
    plt.plot(hist_rand, label='Random Seed', linestyle='--')
    plt.plot(hist_quant, label='Quantum Seed', linewidth=2)
    plt.title(f'MTS Convergence Comparison (N={N})')
    plt.xlabel('Generations')
    plt.ylabel('Energy')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

def bitstring_to_arr(bs):
    """Converts a bitstring '010...' to a numpy array [+1, -1, +1...]"""
    return np.array([1 if b == '0' else -1 for b in bs])

# ==============================================================================
# 5. MAIN EXECUTION WITH TIMING
# ==============================================================================
if __name__ == "__main__":
    # Experiment Settings
    N_TRAIN = 5
    N_TARGET = 10  # Change this to scale up (e.g., 20, 30)
    POP_SIZE = 20
    
    # Trackers for timing
    time_log = {}
    total_start_time = time.time()

    # --- Step 1: GQE Training (N=5) ---
    print(f"\n=== 1. GQE Training (N={N_TRAIN}) ===")
    t_start = time.time()
    
    ham_train = LABSPhysics.build_hamiltonian(N_TRAIN)
    pool, meta = QuantumEngine.generate_pool(N_TRAIN)
    
    # Define cost wrapper for GQE (Extracts coefficients/words for the kernel)
    def cost_fn(sampled_ops, **kwargs):
        coeffs = [[t.evaluate_coefficient() for t in op][0].real for op in sampled_ops]
        words = [[t.get_pauli_word(N_TRAIN) for t in op][0] for op in sampled_ops]
        return cudaq.observe(kernel_train, ham_train, N_TRAIN, coeffs, words).expectation()

    cfg = get_default_config()
    cfg.max_iters, cfg.ngates, cfg.verbose = 20, 100, False
    _, best_idx = solvers.gqe(cost_fn, pool, config=cfg)
    
    t_end = time.time()
    time_log['GQE Training'] = t_end - t_start
    print(f"-> Identified {len(best_idx)} key geometric features.")
    print(f"[Time] GQE Training: {time_log['GQE Training']:.4f} sec")

    # --- Step 2: Transfer (N=5 -> N=Target) ---
    print(f"\n=== 2. Transfer to Target (N={N_TARGET}) ===")
    t_start = time.time()
    
    tl_coeffs, tl_words = QuantumEngine.transfer_features(best_idx, meta, N_TARGET)
    
    t_end = time.time()
    time_log['Transfer'] = t_end - t_start
    print(f"[Time] Transfer Learning: {time_log['Transfer']:.4f} sec")
    
    # --- Step 3: Sampling ---
    print(f"\n=== 3. Quantum Sampling ===")
    t_start = time.time()
    
    res = cudaq.sample(kernel_target, N_TARGET, tl_coeffs, tl_words, shots_count=2000)
    
    # Process samples into MTS seeds (Sort by frequency)
    sorted_samples = sorted(res.items(), key=lambda x: x[1], reverse=True)
    seeds = []
    for bs, _ in sorted_samples[:POP_SIZE]:
        # Filter for valid length and convert to numpy array
        if len(bs) == N_TARGET: seeds.append(bitstring_to_arr(bs))
        
    t_end = time.time()
    time_log['Quantum Sampling'] = t_end - t_start
    print(f"[Time] Sampling: {time_log['Quantum Sampling']:.4f} sec")
    
    # --- Step 4: MTS Optimization ---
    print("\n=== 4. Running MTS (Random vs Quantum) ===")
    t_start = time.time()
    
    mts = MTSOptimizer(N_TARGET, pop_size=POP_SIZE)
    
    # Run Random Initialization Baseline
    print("-> Running Random Initialization...")
    _, e_rand, _, h_rand = mts.run(max_gens=80, verbose=False)
    
    # Run Quantum Initialization
    print("-> Running Quantum Initialization...")
    _, e_quant, _, h_quant = mts.run(max_gens=80, initial_pop=seeds, verbose=False)
    
    t_end = time.time()
    time_log['MTS Optimization'] = t_end - t_start
    print(f"[Time] MTS Optimization: {time_log['MTS Optimization']:.4f} sec")
    
    # --- Final Report ---
    total_end_time = time.time()
    total_duration = total_end_time - total_start_time
    
    print("\n" + "="*40)
    print("       EXECUTION TIME REPORT       ")
    print("="*40)
    for stage, duration in time_log.items():
        print(f"{stage:<20} : {duration:>8.4f} sec")
    print("-" * 40)
    print(f"{'TOTAL TIME':<20} : {total_duration:>8.4f} sec")
    print("="*40)
    
    print(f"\nResult: Random E={e_rand} | Quantum E={e_quant}")
    visualize_comparison(h_rand, h_quant, N_TARGET)