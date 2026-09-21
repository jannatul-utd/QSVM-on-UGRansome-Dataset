# src/build_feature_map.py — updated
from qiskit.circuit.library import z_feature_map

n_qubits = 3
reps = 1

feature_map = z_feature_map(feature_dimension=n_qubits, reps=reps)
print(feature_map)
print(f"\nDepth: {feature_map.depth()}")
print(f"Parameters (should match n_qubits): {feature_map.num_parameters}")