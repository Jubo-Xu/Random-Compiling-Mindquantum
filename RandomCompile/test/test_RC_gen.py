# import sys
# import os

# # Add the parent directory of RandomCompiling to sys.path
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))


import pytest
import math
import cmath
from hypothesis import given, strategies, assume, example, settings
from RandomCompile.CircuitGenerate import QuantumCircuitRC

# The hypothesis test to check whether the quantum circuit logic is the same for ideal one and RC one

# @pytest.mark.parametrize("n_qubit, n_trials, n_max_cycle", [
#     (2, 1, 1)
#     # Add more test cases as needed
# ])
@given(strategies.integers(min_value=2, max_value=5), strategies.integers(min_value=1, max_value=10), strategies.integers(min_value=1, max_value=10))
def test_RC_gate_functionality_hypothesis(n_qubit, n_trials, n_max_cycle):
    qc_RC = QuantumCircuitRC(n_qubit)
    qc_RC.generate_ideal_circuit_random(n_max_cycle, 1, single_multi_qubit_gate=False)
    qc_RC.generate_trials_circuit(n_trials)
    qc_RC.apply_circuit_all_trials()
    ideal_sv = qc_RC.ideal_circuit_sim_result.get_pure_state_vector()
    trial_sv = []
    for i in range(len(qc_RC.trials_circuit_sim_result)):
        trial_sv.append(qc_RC.trials_circuit_sim_result[i].get_pure_state_vector())
    for i in range(len(trial_sv)):
        for j in range(len(ideal_sv)):
            # Check the real part of the amplitude
            assert round(ideal_sv[j].real, 12) == round(trial_sv[i][j].real, 12)
            # Check the imaginary part of the amplitude
            assert round(ideal_sv[j].imag, 12) == round(trial_sv[i][j].imag, 12)


# test_RC_gate_functionality_hypothesis(2, 1, 1)