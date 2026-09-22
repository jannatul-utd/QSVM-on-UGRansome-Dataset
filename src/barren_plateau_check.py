"""
Step 4 (final item) - Barren-plateau sanity check.

Measures the variance of the gradient of a Z-expectation observable
w.r.t. a randomly-sampled ansatz parameter, as a function of the number
of ansatz reps (depth proxy) and qubit count. A barren plateau shows up
as gradient variance collapsing exponentially as either grows.
"""
import numpy as np
from qiskit.circuit.library import ZFeatureMap, RealAmplitudes
from qiskit.quantum_info import SparsePauliOp
from qiskit_machine_learning.gradients import ParamShiftEstimatorGradient
from qiskit.primitives import StatevectorEstimator

N_SAMPLES = 50  # random parameter draws per (qubits, reps) setting
RNG = np.random.default_rng(0)


def sample_grad_variance(num_qubits, reps):
    feature_map = ZFeatureMap(num_qubits, reps=1)
    ansatz = RealAmplitudes(num_qubits, reps=reps, entanglement="linear")
    circuit = feature_map.compose(ansatz)

    # Observable: Z on qubit 0 (one of the 3 per-class observables from step 5's plan)
    observable = SparsePauliOp("Z" + "I" * (num_qubits - 1))

    estimator = StatevectorEstimator()
    grad = ParamShiftEstimatorGradient(estimator)

    grads_wrt_first_param = []
    for _ in range(N_SAMPLES):
        x_vals = RNG.uniform(0, 2 * np.pi, feature_map.num_parameters)
        theta_vals = RNG.uniform(0, 2 * np.pi, ansatz.num_parameters)
        all_params = np.concatenate([x_vals, theta_vals])

        result = grad.run([circuit], [observable], [all_params]).result()
        # gradient w.r.t. the first ansatz parameter (index right after feature-map params)
        g = result.gradients[0][feature_map.num_parameters]
        grads_wrt_first_param.append(g)

    return np.var(grads_wrt_first_param)


def main():
    print(f"{'qubits':>6} {'reps':>5} {'grad_variance':>15}")
    for num_qubits in (3, 4, 6, 8):
        for reps in (1, 2, 4, 6):
            var = sample_grad_variance(num_qubits, reps)
            print(f"{num_qubits:>6} {reps:>5} {var:>15.6e}")


if __name__ == "__main__":
    main()