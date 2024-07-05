import numpy as np
from mindquantum.simulator import Simulator
from mindquantum.core.circuit import Circuit
from mindquantum.core.gates import Measure
from mindquantum.core.gates import H, X, Y, Z, CNOT, SWAP, RX, RY, RZ, U3
from mindquantum.core.circuit import GateSelector, SequentialAdder, MixerAdder, NoiseChannelAdder
import random
import math
import cmath

# Define the table for supported multi-qubit gates
MultiQubitGateTable = {
    'CNOT'
}

# Define the table for supported RC gates
RCGateTable = {
    0: 'X',
    1: 'Y',
    2: 'Z',
    3: 'I'
}

# Define the mapping table for RC gate and its complementary
RCGateComplementaryMapping = {
    "XI": ["XX", 0, 0],# the first element is the complementary gate, the second and third elements are sign of two gates, 0 --> +, 1 --> -
    "IX": ["IX", 0, 0],
    "XX": ["XI", 0, 0],
    "YX": ["YY", 0, 0],
    "ZX": ["ZY", 0, 0],
    "YI": ["YX", 0, 0],
    "IY": ["ZY", 0, 0],
    "XY": ["XZ", 0, 0],
    "YY": ["YX", 0, 1],
    "ZY": ["ZZ", 0, 0],
    "ZI": ["ZI", 0, 0],
    "IZ": ["IZ", 0, 0],
    "XZ": ["XY", 0, 0],
    "YZ": ["YY", 0, 1],
    "ZZ": ["ZI", 0, 0],
    "II": ["II", 0, 0]
}

class RandomCompile:
    def __init__(self):
        self.test_trials = [] # list of test circuits 
        
    # The RC gate only applies for multi-qubit gates
    @staticmethod
    def applyRC(RC_apply=True):
        def decorator(func):
            def wrapper(*args, **kwargs):
                qc = func(*args, **kwargs)
                if RC_apply:
                    # Apply the RC gate logic
                    n_qubit = len(qc.gate_list)
                    # total_cycle_added = 0
                    # sec_idx = 0
                    for qubit_idx in range(n_qubit):
                        total_cycle_origin = len(qc.gate_list[qubit_idx])
                        total_cycle_added = 0
                        # for cycle in range(len(qc.gate_list[qubit_idx])):
                        for cycle_origin in range(total_cycle_origin):
                            cycle = cycle_origin + total_cycle_added
                            # if cycle > qc.sec_list[sec_idx][1]:
                            #     qc.sec_list[sec_idx][1] += total_cycle_added
                            #     sec_idx += 1
                            #     if sec_idx < len(qc.sec_list):
                            #         qc.sec_list[sec_idx][0] += total_cycle_added
                            # Check if the gate is a multi-qubit gate
                            if (qc.gate_list[qubit_idx][cycle][0] in MultiQubitGateTable) and (qc.gate_list[qubit_idx][cycle][1][0] == 0):  # only consider target qubit
                                control_rng = random.randint(0, len(RCGateTable) - 1)
                                target_rng = random.randint(0, len(RCGateTable) - 1)
                                control_RC = RCGateTable[control_rng]
                                target_RC = RCGateTable[target_rng]
                                RC_str = control_RC + target_RC
                                RC_comp = RCGateComplementaryMapping[RC_str]
                                control_RC_comp = RC_comp[0][0]
                                target_RC_comp = RC_comp[0][1]
                                control_RC_comp_sign = RC_comp[1]
                                target_RC_comp_sign = RC_comp[2]
                                control_qubit_idx = qc.gate_list[qubit_idx][cycle][1][1]
                                qc.gate_list[qubit_idx].insert(cycle, [target_RC, 0])  # Insert RC gate
                                qc.gate_list[qubit_idx].insert(cycle+2, [target_RC_comp, target_RC_comp_sign])
                                qc.gate_list[control_qubit_idx].insert(cycle, [control_RC, 0])
                                qc.gate_list[control_qubit_idx].insert(cycle+2, [control_RC_comp, control_RC_comp_sign])
                                # sec_idx += 2
                                total_cycle_added += 2
                                continue
                            if qc.gate_list[qubit_idx][cycle][0] == "BARRIER":
                                RC_rng = random.randint(0, len(RCGateTable) - 1)
                                RC_gate = RCGateTable[RC_rng]
                                qc.gate_list[qubit_idx].insert(cycle, [RC_gate, 0])
                                qc.gate_list[qubit_idx].insert(cycle+2, [RC_gate, 0])
                                # sec_idx += 2
                                total_cycle_added += 2
                                continue
                    # update the sec_list
                    start_add = 0
                    end_add = 2
                    for i in range(len(qc.sec_list)):
                        if i == len(qc.sec_list) - 1:
                            qc.sec_list[i][0] += start_add
                            qc.sec_list[i][1] = qc.sec_list[i][0]
                            continue
                        qc.sec_list[i][0] += start_add
                        qc.sec_list[i][1] += end_add
                        start_add = end_add
                        end_add += 2
                return qc
            return wrapper
        return decorator
    
    
    # @staticmethod
    # def applyRC(func, RC_apply=True):
    #     def wrapper(*args, **kwargs):
    #         qc = func(*args, **kwargs)
    #         if RC_apply:
    #             # apply the RC gate
    #             # loop through the gate list
    #             n_qubit = len(qc.gate_list)
    #             for qubit_idx in range(n_qubit):
    #                 total_cycle_added = 0
    #                 sec_idx = 0
    #                 for cycle in range(len(qc.gate_list[qubit_idx])):
    #                     if cycle > qc.sec_list[sec_idx][1]:
    #                         qc.sec_list[sec_idx][1] += total_cycle_added
    #                         sec_idx += 1
    #                         if sec_idx < len(qc.sec_list):
    #                             qc.sec_list[sec_idx][0] += total_cycle_added
    #                     # check if the gate is a multi-qubit gate
    #                     if (qc.gate_list[qubit_idx][cycle][0] in MultiQubitGateTable) and (qc.gate_list[qubit_idx][cycle][1][0] == 0): # only consider about target qubit
    #                         # randomly select two RC gates applied on control and target qubits
    #                         # randomly generate the index 
    #                         control_rng = random.randint(0, len(RCGateTable) - 1)
    #                         target_rng = random.randint(0, len(RCGateTable) - 1)
    #                         # select the corresponding RC gates based on the index
    #                         control_RC = RCGateTable[control_rng]
    #                         target_RC = RCGateTable[target_rng]
    #                         RC_str = control_RC + target_RC
    #                         # choose the complementary RC gates
    #                         RC_comp = RCGateComplementaryMapping[RC_str]
    #                         control_RC_comp = RC_comp[0][0]
    #                         target_RC_comp = RC_comp[0][1]
    #                         control_RC_comp_sign = RC_comp[1]
    #                         target_RC_comp_sign = RC_comp[2]
    #                         # add the RC gates to target qubit
    #                         qc.gate_list[qubit_idx].insert(cycle, [target_RC, 0]) # the RC gate info to gate_list is [gate name, 0/1] with 0 --> +, 1 --> -
    #                         # add the complementary RC gates
    #                         qc.gate_list[qubit_idx].insert(cycle+2, [target_RC_comp, target_RC_comp_sign])
    #                         # add the RC gates to control qubit
    #                         control_qubit_idx = qc.gate_list[qubit_idx][cycle][1][1]
    #                         qc.gate_list[control_qubit_idx].insert(cycle, [control_RC, 0])
    #                         qc.gate_list[control_qubit_idx].insert(cycle+2, [control_RC_comp, control_RC_comp_sign])
    #                         # update the sec_idx
    #                         sec_idx += 2
    #                         continue
    #                     # check if the gate is a Barrier 
    #                     if qc.gate_list[qubit_idx][cycle][0] == "BARRIER":
    #                         # randomly select a RC gate to apply
    #                         RC_rng = random.randint(0, len(RCGateTable) - 1)
    #                         RC_gate = RCGateTable[RC_rng]
    #                         # add the RC gate to the qubit
    #                         qc.gate_list[qubit_idx].insert(cycle, [RC_gate, 0])
    #                         # add the complementary RC gate to the qubit
    #                         qc.gate_list[qubit_idx].insert(cycle+2, [RC_gate, 0])
    #                         # update the sec_idx
    #                         sec_idx += 2
    #                         continue
    #         return qc
    #     return wrapper
                    
                    