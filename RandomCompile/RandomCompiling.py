import numpy as np
from mindquantum.simulator import Simulator
from mindquantum.core.circuit import Circuit
from mindquantum.core.gates import Measure
from mindquantum.core.gates import H, X, Y, Z, CNOT, SWAP, RX, RY, RZ, U3, I
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

RCGateSet = {
    'X',
    'Y',
    'Z',
    'I'
}

# RC_matrix_table = {
#     "X": np.array([[0, 1], [1, 0]]),
#     "Y": np.array([[0, -1j], [1j, 0]]),
#     "Z": np.array([[1, 0], [0, -1]]),
#     "I": np.array([[1, 0], [0, 1]])
# }
RC_matrix_table = {
    "X": X.matrix(),
    "Y": Y.matrix(),
    "Z": Z.matrix(),
    "I": I.matrix()
}

# Define the mapping table for RC gate and its complementary
RCGateComplementaryMapping = {
    "XI": ["XX", 0, 0],# the first element is the complementary gate, the second and third elements are sign of two gates, 0 --> +, 1 --> -
    "IX": ["IX", 0, 0],
    "XX": ["XI", 0, 0],
    "YX": ["YI", 0, 0],
    "ZX": ["ZX", 0, 0],
    "YI": ["YX", 0, 0],
    "IY": ["ZY", 0, 0],
    "XY": ["YZ", 0, 0],
    "YY": ["XZ", 0, 1],
    "ZY": ["IY", 0, 0],
    "ZI": ["ZI", 0, 0],
    "IZ": ["ZZ", 0, 0],
    "XZ": ["YY", 0, 1],
    "YZ": ["XY", 0, 0],
    "ZZ": ["IZ", 0, 0],
    "II": ["II", 0, 0]
}

class RandomCompile:
    def __init__(self):
        self.test_trials = [] # list of test circuits 
    
    # The function to get the matrix representation of the U3 gate
    @staticmethod
    def get_U3_matrix(theta, phi, lambda_):
        return np.array([[math.cos(theta/2), -cmath.exp(1j*lambda_)*math.sin(theta/2)], \
            [cmath.exp(1j*phi)*math.sin(theta/2), cmath.exp(1j*(phi+lambda_))*math.cos(theta/2)]])
    
    # The function to get the U3 gate from the matrix representation
    @staticmethod
    def get_U3_gate(matrix):
        # Ensure U is a unitary matrix
        assert np.allclose(matrix @ matrix.conj().T, np.eye(2)), "The gate is not unitary"
        theta = 2*math.acos(np.abs(matrix[0][0]))
        phi = cmath.phase(matrix[1][0]) - cmath.phase(matrix[0][0])
        lambda_ = cmath.phase(matrix[1][1]) - cmath.phase(matrix[1][0])
        return [theta, phi, lambda_]
        # # Extract the elements of U
        # U = matrix
        # U00, U01 = U[0, 0], U[0, 1]
        # U10, U11 = U[1, 0], U[1, 1]
        # # Calculate theta
        # theta = 2 * np.arccos(np.abs(U00))
        # # Calculate phi
        # if np.abs(np.sin(theta / 2)) > 1e-10:  # To avoid division by zero
        #     phi = np.angle(U10) - np.angle(U00)
        # else:
        #     phi = 0
        # # Calculate lambda
        # if np.abs(np.sin(theta / 2)) > 1e-10:
        #     lambda_ = np.angle(U01) - np.angle(U10)
        # else:
        #     lambda_ = np.angle(U11) - np.angle(U00) - phi
        # return [theta, phi, lambda_]
    
    # The function to apply the RC gate for a single unit
    @staticmethod
    def applyRC_single_unit(qc):
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
        # find target index and control index
        target_idx = 0 if qc.gate_list[0][0][1][0] == 0 else 1
        control_idx = 1 if target_idx == 0 else 0
        qc.gate_list[target_idx].insert(0, [target_RC, 0])  # Insert RC gate
        qc.gate_list[target_idx].insert(2, [target_RC_comp, target_RC_comp_sign])
        qc.gate_list[control_idx].insert(0, [control_RC, 0])
        qc.gate_list[control_idx].insert(2, [control_RC_comp, control_RC_comp_sign])
        # update the sec_list
        qc.sec_list[0][1] += 2
        return qc
    
    # The RC gate only applies for multi-qubit gates
    @staticmethod
    def applyRC(RC_apply=True, Singleunit='SINGLE_UNIT'):
        def decorator(func):
            def wrapper(self, *args, **kwargs):
                SINGLE_UNIT = getattr(self, Singleunit, None)
                qc = func(self, *args, **kwargs)
                # ideal_copy = args[1]
                ideal_copy = kwargs.get('ideal_copy', False)
                if RC_apply and (not ideal_copy):
                    if SINGLE_UNIT:
                        return RandomCompile.applyRC_single_unit(qc)
                    # Apply the RC gate logic
                    n_qubit = len(qc.gate_list)
                    # total_cycle_added = 0
                    # sec_idx = 0
                    total_cycle_origin = len(qc.gate_list[0])
                    total_cycle_added = 0
                    # for qubit_idx in range(n_qubit):
                    for cycle_origin in range(total_cycle_origin):
                        # total_cycle_origin = len(qc.gate_list[qubit_idx])
                        # total_cycle_added = 0
                        # for cycle in range(len(qc.gate_list[qubit_idx])):
                        # for cycle_origin in range(total_cycle_origin):
                        CYCLE_ADD = False
                        for qubit_idx in range(n_qubit):
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
                                # total_cycle_added += 2
                                CYCLE_ADD = True
                                continue
                            if qc.gate_list[qubit_idx][cycle][0] == "BARRIER":
                                RC_rng = random.randint(0, len(RCGateTable) - 1)
                                RC_gate = RCGateTable[RC_rng]
                                qc.gate_list[qubit_idx].insert(cycle, [RC_gate, 0])
                                qc.gate_list[qubit_idx].insert(cycle+2, [RC_gate, 0])
                                # sec_idx += 2
                                # total_cycle_added += 2
                                CYCLE_ADD = True
                                continue
                        if CYCLE_ADD:
                            total_cycle_added += 2
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
    
    # The function to combine the RC gates with previous and next single qubit gates
    @staticmethod
    def applyRC_combine(qc_comb, qc):
        # update the gate_list
        for qubit_idx in range(qc.qubits_num):
            cycle = 0
            while cycle < len(qc.gate_list[qubit_idx]):
                # check if the current gate is a forward RC gate
                if (qc.gate_list[qubit_idx][cycle][0] in RCGateSet) and (qc.gate_list[qubit_idx][cycle+1][0] in MultiQubitGateTable):
                    gate_info_last = qc_comb.gate_list[qubit_idx].pop()
                    gate_info_new = []
                    U_pre_Matrix = RandomCompile.get_U3_matrix(gate_info_last[1][0], gate_info_last[1][1], gate_info_last[1][2])
                    RC_Matrix = RC_matrix_table[qc.gate_list[qubit_idx][cycle][0]]
                    U_Matrix = RC_Matrix @ U_pre_Matrix
                    U_param = RandomCompile.get_U3_gate(U_Matrix)
                    gate_info_new.append("U3")
                    gate_info_new.append(U_param)
                    qc_comb.gate_list[qubit_idx].append(gate_info_new)
                    cycle += 1
                # check if the current gate is a backward RC gate
                elif (qc.gate_list[qubit_idx][cycle][0] in RCGateSet) and (qc.gate_list[qubit_idx][cycle-1][0] in MultiQubitGateTable):
                    RC_gate_info = qc.gate_list[qubit_idx][cycle]
                    U_next_gate_info = qc.gate_list[qubit_idx][cycle+1]
                    RC_matrix = RC_matrix_table[RC_gate_info[0]]
                    U_next_matrix = RandomCompile.get_U3_matrix(U_next_gate_info[1][0], U_next_gate_info[1][1], U_next_gate_info[1][2])
                    U_Matrix = U_next_matrix @ RC_matrix
                    U_param = RandomCompile.get_U3_gate(U_Matrix)
                    gate_info_new = ["U3", U_param]
                    qc_comb.gate_list[qubit_idx].append(gate_info_new)
                    cycle += 2
                # check if the RC gate is applied for barrier
                elif (qc.gate_list[qubit_idx][cycle][0] in RCGateSet) and (qc.gate_list[qubit_idx][cycle+1][0] == "BARRIER"):
                    qc_comb.gate_list[qubit_idx].append(["BARRIER"])
                    cycle += 3
                # check if the current gate is a multiple qubit gate or normal U3 gate
                else:
                    gate_info_new = []
                    for i in range(len(qc.gate_list[qubit_idx][cycle])):
                        if qc.gate_list[qubit_idx][cycle][i] is list:
                            elem_info = []
                            for j in range(len(qc.gate_list[qubit_idx][cycle][i])):
                                elem_info.append(qc.gate_list[qubit_idx][cycle][i][j])
                            gate_info_new.append(elem_info)
                        else:
                            gate_info_new.append(qc.gate_list[qubit_idx][cycle][i])
                    qc_comb.gate_list[qubit_idx].append(gate_info_new)
                    cycle += 1
        # update the sec_list
        sec_list_comb = []
        cycle_sub = 0
        for i in range(len(qc.sec_list)):
            sec = []
            if i == len(qc.sec_list) - 1:
                sec.append(qc.sec_list[i][0]-cycle_sub)
                sec.append(qc.sec_list[i][1]-cycle_sub)
                sec_list_comb.append(sec)
                continue
            sec.append(qc.sec_list[i][0]-cycle_sub)
            cycle_sub += 2
            sec.append(qc.sec_list[i][1]-cycle_sub)
            sec_list_comb.append(sec)
        qc_comb.sec_list = sec_list_comb
        return qc_comb
    
    ############################## The metric to evaluate the fidelity of the compiled circuit ##############################
    # Total Variation Distance
    @staticmethod
    def Total_Variation_Distance(prob_ideal, prob_noise):
        abs_sum = 0
        for i in range(len(prob_ideal)):
            abs_sum += abs(prob_ideal[i] - prob_noise[i])
        return 0.5*abs_sum
    
    @staticmethod
    def Fidelity_Evaluation(prob_ideal, prob_no_RC, prob_RC, metric="TVD"):
        no_RC_fidelity = 0
        RC_fidelity = 0
        if metric == "TVD":
            no_RC_fidelity = RandomCompile.Total_Variation_Distance(prob_ideal, prob_no_RC)
            RC_fidelity = RandomCompile.Total_Variation_Distance(prob_ideal, prob_RC)
        return no_RC_fidelity, RC_fidelity
    
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

# print(U3(0.5, 0.5, 0.2).matrix()[0][0].real)

# matrix = U3(0, 0, 1).matrix()
# print(matrix)
# print(RandomCompile.get_U3_gate(matrix))

# matrix = U3(3.9, 0.3, 1.9).matrix()
# # M_result = matrix @ RC_matrix_table['Z']
# M_result = RC_matrix_table['X'] @ matrix
# print(RandomCompile.get_U3_gate(M_result)) 

# print(np.kron(U3(4.2, 3.5, 3.7).matrix(), U3(3.1, 3.4, 6.1).matrix()) @ np.kron(Z.matrix(), Y.matrix()))
# U_matrix_1 = RandomCompile.get_U3_matrix(2.1, -2.8, 0.5)
# U_matrix_0 = RandomCompile.get_U3_matrix(0.1, -0.3, -3.3)
# print(np.kron(U_matrix_1, U_matrix_0))

U_1_matrix = U3(4.2, 3.5, 3.7).matrix()
U_0_matrix = U3(3.1, 3.4, 6.1).matrix()
RC_1_matrix = Z.matrix()
RC_0_matrix = Y.matrix()
param_1 = RandomCompile.get_U3_gate(RC_1_matrix @ U_1_matrix)
# print(RC_1_matrix @ U_1_matrix)
# print(U3(param_1[0], param_1[1], param_1[2]).matrix())
# param_0 = RandomCompile.get_U3_gate(RC_0_matrix @ U_0_matrix)
# print(RC_0_matrix @ U_0_matrix)
# print(U3(param_0[0], param_0[1], param_0[2]).matrix())

param = RandomCompile.get_U3_gate(U_1_matrix)
# print(U_1_matrix)
# print(RandomCompile.get_U3_matrix(param[0], param[1], param[2]))
