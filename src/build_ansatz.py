"""
Step 4 - Build the full circuit (ZFeatureMap encoding + RealAmplitudes ansatz),
transpile it, and report depth / gate counts.

Decisions carried over from the roadmap doc:
  - Step 3: ZFeatureMap(3, reps=1)          -- encoding layer, depth 2, 3 params
  - Step 4: RealAmplitudes(3, reps=2, entanglement="linear") -- ansatz,
            depth 7, 9 trainable params (already confirmed on its own)
"""
from qiskit.circuit.library import ZFeatureMap, RealAmplitudes
from qiskit import transpile

NUM_QUBITS = 3
BASIS_GATES = ["rz", "sx", "x", "cx"]  # a realistic 1q+2q hardware-like basis


def main():
    feature_map = ZFeatureMap(NUM_QUBITS, reps=1)
    ansatz = RealAmplitudes(NUM_QUBITS, reps=2, entanglement="linear")

    # Encoding layer feeds into the trainable ansatz -- compose in sequence.
    circuit = feature_map.compose(ansatz)
    circuit.measure_all()

    print("=== Untranspiled (logical) circuit ===")
    print(circuit.draw(output="text"))
    print(f"Logical depth: {circuit.depth()}")
    print(f"Feature-map params: {feature_map.num_parameters}, "
          f"Ansatz params: {ansatz.num_parameters}, "
          f"Total trainable params: {ansatz.num_parameters}")
    print()

    # Transpile against a basis gate set so depth reflects what real
    # hardware would actually run, not just the logical gate list.
    transpiled = transpile(circuit, basis_gates=BASIS_GATES, optimization_level=1)

    print("=== Transpiled circuit (basis: rz, sx, x, cx) ===")
    print(f"Transpiled depth: {transpiled.depth()}")
    print(f"Gate counts: {dict(transpiled.count_ops())}")
    print(f"Total gate count: {sum(transpiled.count_ops().values())}")

    # Also report depth restricted to 2-qubit gates only (the metric that
    # matters most for near-term noise / barren-plateau risk).
    two_q_depth = transpiled.depth(lambda instr: instr.operation.num_qubits == 2)
    print(f"2-qubit-gate depth (CX layers): {two_q_depth}")


if __name__ == "__main__":
    main()