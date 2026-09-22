"""
Step 5 (prep) - Benchmark circuit-evaluation throughput to decide training
set size within a runtime budget.
"""
import time
import numpy as np
from qiskit.circuit.library import ZFeatureMap, RealAmplitudes
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import StatevectorEstimator

NUM_QUBITS = 3
BATCH_SIZES = [50, 200, 1000, 5000]
RNG = np.random.default_rng(0)


def main():
    feature_map = ZFeatureMap(NUM_QUBITS, reps=1)
    ansatz = RealAmplitudes(NUM_QUBITS, reps=2, entanglement="linear")
    circuit = feature_map.compose(ansatz)

    observables = [SparsePauliOp("Z" + "I" * (NUM_QUBITS - 1)),
                   SparsePauliOp("I" + "Z" + "I"),
                   SparsePauliOp("I" + "I" + "Z")]  # one per class (S/A/SS)

    estimator = StatevectorEstimator()
    n_params = feature_map.num_parameters + ansatz.num_parameters

    print(f"{'batch_size':>10} {'seconds':>10} {'evals/sec':>12}")
    for batch in BATCH_SIZES:
        param_sets = RNG.uniform(0, 2 * np.pi, (batch, n_params))
        pubs = [(circuit, obs, param_sets) for obs in observables]

        start = time.perf_counter()
        estimator.run(pubs).result()
        elapsed = time.perf_counter() - start

        evals_per_sec = batch / elapsed
        print(f"{batch:>10} {elapsed:>10.3f} {evals_per_sec:>12.1f}")


if __name__ == "__main__":
    main()