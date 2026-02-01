import math
import numpy as np
import pytest
from hypothesis import given, strategies as st

# ---- import from your project ----
# Adjust these imports to match your repo structure.
from labs.energy import labs_energy_spins
from labs.mapping import bitstring_to_spins, spins_to_bitstring
from pipeline.filtering import select_top_k

# Optional CUDA-Q imports; skip tests if cudaq is not available
try:
    import cudaq
    CUDAQ_AVAILABLE = True
except Exception:
    CUDAQ_AVAILABLE = False


# ---------------------------
# Helpers
# ---------------------------
def random_spins(n: int, rng=None) -> np.ndarray:
    rng = rng or np.random.default_rng()
    return rng.choice([-1, 1], size=n).astype(np.int8)


def reverse_spins(s: np.ndarray) -> np.ndarray:
    return s[::-1].copy()


def negate_spins(s: np.ndarray) -> np.ndarray:
    return (-s).copy()


# ---------------------------
# A) Classical correctness
# ---------------------------

@given(st.integers(min_value=2, max_value=64))
def test_energy_is_nonnegative_integer(n):
    s = random_spins(n)
    e = labs_energy_spins(s)
    assert isinstance(e, (int, np.integer))
    assert e >= 0


@given(st.integers(min_value=2, max_value=64))
def test_reversal_symmetry(n):
    s = random_spins(n)
    assert labs_energy_spins(s) == labs_energy_spins(reverse_spins(s))


@given(st.integers(min_value=2, max_value=64))
def test_negation_symmetry(n):
    s = random_spins(n)
    assert labs_energy_spins(s) == labs_energy_spins(negate_spins(s))


@given(st.integers(min_value=2, max_value=64))
def test_bitstring_spin_mapping_roundtrip(n):
    # Make a random bitstring
    bits = ''.join(np.random.choice(['0', '1'], size=n))
    spins = bitstring_to_spins(bits)
    bits2 = spins_to_bitstring(spins)
    assert bits2 == bits


def test_known_small_n_regression():
    """
    Keep this test VERY simple + stable.
    If you claim "N=7 must return E=3", then you need the actual optimal sequence
    or brute-force verification.
    """
    # One known optimal for N=7 (±1 form) can be used if you have it.
    # If you don't, replace with brute-force computation (see below).
    # Example placeholder sequence (you MUST replace if not optimal):
    spins = np.array([1, 1, 1, -1, -1, 1, -1], dtype=np.int8)
    e = labs_energy_spins(spins)

    # If you're not 100% sure of the sequence, don't hardcode it.
    # Instead do brute-force for N=7:
    e_star = brute_force_min_energy(7)
    assert e == e_star


def brute_force_min_energy(n: int) -> int:
    """
    Brute-force LABS energy for small n only (n<=20 gets expensive).
    For tests, use n=7 or n=8.
    """
    best = None
    for x in range(2**n):
        bits = format(x, f"0{n}b")
        spins = bitstring_to_spins(bits)
        e = labs_energy_spins(spins)
        if best is None or e < best:
            best = e
    return int(best)


def test_bruteforce_n7_matches_claim():
    # If your PRD claims N=7 -> E=3, verify it from brute-force once.
    assert brute_force_min_energy(7) == 3


def test_filtering_select_top_k():
    # Make fake bitstrings and ensure selection is unique + sorted by energy
    n = 10
    bitstrings = [''.join(np.random.choice(['0','1'], size=n)) for _ in range(200)]
    top = select_top_k(bitstrings, k=20)

    assert len(top) <= 20
    assert len(set(top)) == len(top)
    assert all(len(b) == n for b in top)

    energies = [labs_energy_spins(bitstring_to_spins(b)) for b in top]
    assert energies == sorted(energies)


# ---------------------------
# B) CUDA-Q sanity checks
# ---------------------------

@pytest.mark.skipif(not CUDAQ_AVAILABLE, reason="cudaq not installed/available")
def test_cudaq_smoke_build_and_sample():
    """
    This test is intentionally a 'smoke test':
    - circuit can be built
    - sampling returns correct bitstring lengths
    """
    from quantum.gqe import build_circuit, sample_circuit

    n = 8
    params = np.zeros(4, dtype=float)  # adjust param count to your ansatz
    kernel = build_circuit(n, params)
    shots = 50
    bitstrings = sample_circuit(kernel, shots=shots)

    assert isinstance(bitstrings, list)
    assert len(bitstrings) > 0
    assert all(isinstance(b, str) for b in bitstrings)
    assert all(len(b) == n for b in bitstrings)
    assert set(''.join(bitstrings)).issubset({'0', '1'})


@pytest.mark.skipif(not CUDAQ_AVAILABLE, reason="cudaq not installed/available")
def test_cudaq_pipeline_produces_valid_seeds():
    """
    End-to-end sanity: quantum sample -> map -> energy -> pick K
    """
    from quantum.gqe import build_circuit, sample_circuit

    n = 10
    k = 10
    params = np.zeros(4, dtype=float)
    kernel = build_circuit(n, params)
    samples = sample_circuit(kernel, shots=200)

    top = select_top_k(samples, k=k)
    assert 1 <= len(top) <= k
    assert len(set(top)) == len(top)

    # Energies must be finite, nonnegative ints
    for b in top:
        e = labs_energy_spins(bitstring_to_spins(b))
        assert isinstance(e, (int, np.integer))
        assert e >= 0
