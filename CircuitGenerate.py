import numpy as np
from mindquantum.simulator import Simulator
from mindquantum.core.circuit import Circuit
from mindquantum.core.gates import Measure
from mindquantum.core.gates import H, X, Y, Z, CNOT, SWAP, RX, RY, RZ, U3, GlobalPhase, I
from mindquantum.core.circuit import GateSelector, SequentialAdder, MixerAdder, NoiseChannelAdder
from mindquantum.core.operators import Hamiltonian
from mindquantum.core.operators import QubitOperator
import random
import math
import cmath
import json
import RandomCompiling

class QuantumCircuit:
    def __init__(self, qubits_num, circuit_name="test"):
        self.name = circuit_name
        self.qubits_num = qubits_num
        self.qubits_idx_list = [i for i in range(qubits_num)] # a list containing the qubit index
        self.gate_list = [[] for i in range(qubits_num)] # a two dimensional list containing the gates applied to each qubit, the first dimension is the qubit index, the second dimension is the quantum cycle
        self.sec_list = [] # a list containing the start and end indices of each section in a 2 element list, len(sec_list) = number of sections
        self.circuit = Circuit()
    
    def to_json(self):
        with open(self.name+'.json', 'w') as json_file:
            gate_dict = {"gate_list": self.gate_list, "sec_list": self.sec_list}
            json.dump(gate_dict, json_file)
    
    @staticmethod
    def from_json(json_file):
        with open(json_file, 'r') as jsonfile:
            gate_dict = json.load(jsonfile)
        gate_list = gate_dict["gate_list"]
        sec_list = gate_dict["sec_list"]
        qubits_num = len(gate_list)
        # the name of the quantum circuit is the json file name without the extension
        qc = QuantumCircuit(qubits_num, json_file.split(".")[0])
        qc.gate_list = gate_list
        qc.sec_list = sec_list
        return qc
        
    def add_gate(self, qubit_index, gate_info):
        if qubit_index >= self.qubits_num:
            raise ValueError("qubit index out of range")
        self.gate_list[qubit_index].append(gate_info)
    
    def remove_gate(self, qubit_index, cycle):
        if qubit_index >= self.qubits_num:
            raise ValueError("qubit index out of range")
        if cycle >= len(self.gate_list[qubit_index]):
            raise ValueError("cycle out of range")
        self.gate_list[qubit_index] = self.gate_list[qubit_index][:cycle].extend(self.gate_list[qubit_index][cycle+1:])
    
    def change_gate(self, qubit_index, cycle, gate_info):
        if qubit_index >= self.qubits_num:
            raise ValueError("qubit index out of range")
        if cycle >= len(self.gate_list[qubit_index]):
            raise ValueError("cycle out of range")
        self.gate_list[qubit_index][cycle] = gate_info
    
    
    # This function is used to randomly generate the gates for each quantum cycle
    # The quantum cycle in this case is divided with respect to multi-qubits controlled gate --> each cycle contains one multi-qubits controlled gate
    # Each cycle contains at most max_single_num_per_cycle single-qubit gates before the multi-qubits controlled gate
    # For now the only multi-qubits controlled gate considered is CNOT gate
    def generate_gate_random(self, max_cycle, max_single_num_per_cycle, single_multi_qubit_gate=True):
        sec_start_idx = 0
        sec_end_idx = -1
        for cycle in range(max_cycle):
            self.sec_list.append([])
            sec = [sec_start_idx]
            if max_single_num_per_cycle == 1:
                for i in range(self.qubits_num):
                    # randomly generate the three parameters of the U3 gate
                    theta = random.uniform(0, 2*math.pi)
                    phi = random.uniform(0, 2*math.pi)
                    lam = random.uniform(0, 2*math.pi)
                    self.add_gate(i, ["U3", [theta, phi, lam]])
                sec_end_idx += 1
            else:
                # randomly generate the number of single qubit gate sections before the multi-qubits controlled gate
                single_num = random.randint(1, max_single_num_per_cycle)
                for i in range(single_num):
                    for j in range(self.qubits_num):
                        # randomly generate the three parameters of the U3 gate
                        theta = random.uniform(0, 2*math.pi)
                        phi = random.uniform(0, 2*math.pi)
                        lam = random.uniform(0, 2*math.pi)
                        self.add_gate(j, ["U3", [theta, phi, lam]])
                sec_end_idx += single_num
            # randomly generate the target and control qubit pair for the CNOT gate
            # if there's only one CNOT gate 
            if single_multi_qubit_gate:
                target_idx = random.randint(0, self.qubits_num-1)
                control_idx = random.randint(0, self.qubits_num-1)
                # make sure target qubit and control qubit are not the same
                if target_idx == control_idx:
                    control_idx = (target_idx+1) if target_idx < self.qubits_num-1 else (target_idx-1)
                # add the gates
                self.add_gate(target_idx, ["CNOT", [0, control_idx]])
                self.add_gate(control_idx, ["CNOT", [1, target_idx]])
                # add the barrier for other qubits
                for i in range(self.qubits_num):
                    if i != target_idx and i != control_idx:
                        self.add_gate(i, ["BARRIER"])
            else:
                target_control_idx_pair = select_target_control_pair_rand(self.qubits_idx_list)
                # Check which qubits are not used
                # create a dictionary to store the qubit index that is not used
                valid_dict = {i: -1 for i in range(self.qubits_num)}
                for pair in target_control_idx_pair:
                    if pair[0] != -1:
                        valid_dict[pair[0]] = 1
                    if pair[1] != -1:
                        valid_dict[pair[1]] = 1
                # add the gates for valid target and control pairs
                for pair in target_control_idx_pair:
                    if pair[0] == -1 and pair[1] == -1:
                        continue
                    else:
                        target_idx = pair[0]
                        control_idx = pair[1]
                        self.add_gate(target_idx, ["CNOT", [0, control_idx]]) # 0 indicating target qubit, 1 indicating control qubit
                        self.add_gate(control_idx, ["CNOT", [1, target_idx]])
                # add the gates for invalid target and control pairs
                for qubit_idx in valid_dict:
                    if valid_dict[qubit_idx] == 1:
                        continue
                    self.add_gate(qubit_idx, ["BARRIER"])
            # add the section index to the section list
            sec_end_idx += 1
            sec.append(sec_end_idx)
            self.sec_list[-1] = sec
            sec_start_idx = sec_end_idx + 1
        # add the last section with only single qubit gates
        for qubit_idx in range(self.qubits_num):
            # randomly generate the three parameters of the U3 gate
            theta = random.uniform(0, 2*math.pi)
            phi = random.uniform(0, 2*math.pi)
            lam = random.uniform(0, 2*math.pi)
            self.add_gate(qubit_idx, ["U3", [theta, phi, lam]])
        sec_end_idx += 1
        self.sec_list.append([sec_start_idx, sec_end_idx])
    
    # The function to apply the quantum circuit to the simulator
    def apply_circuit(self, simulator='mqvector'):
        for qubit_idx in range(self.qubits_num):
            for cycle in range(len(self.gate_list[qubit_idx])):
                gate_info = self.gate_list[qubit_idx][cycle]
                if gate_info[0] == "U3":
                    self.circuit += U3(gate_info[1][0], gate_info[1][1], gate_info[1][2]).on(qubit_idx)
                elif gate_info[0] == "CNOT":
                    if gate_info[1][0] == 0:
                        self.circuit += X.on(qubit_idx, gate_info[1][1])
                    else:
                        continue
                elif gate_info[0] == "X":
                    if gate_info[1] == 0:
                        self.circuit += X.on(qubit_idx)
                    else:
                        self.circuit += X.on(qubit_idx)
                        self.circuit += GlobalPhase(math.pi).on(qubit_idx)
                elif gate_info[0] == "Y":
                    if gate_info[1] == 0:
                        self.circuit += Y.on(qubit_idx)
                    else:
                        self.circuit += Y.on(qubit_idx)
                        self.circuit += GlobalPhase(math.pi).on(qubit_idx)
                elif gate_info[0] == "Z":
                    if gate_info[1] == 0:
                        self.circuit += Z.on(qubit_idx)
                    else:
                        self.circuit += Z.on(qubit_idx)
                        self.circuit += GlobalPhase(math.pi).on(qubit_idx)
                elif gate_info[0] == "I":
                    if gate_info[1] == 0:
                        self.circuit += I.on(qubit_idx)
                    else:
                        self.circuit += I.on(qubit_idx)
                        self.circuit += GlobalPhase(math.pi).on(qubit_idx)
                else:
                    continue
        sim = Simulator(simulator, self.qubits_num)
        sim.apply_circuit(self.circuit)
        return sim
    
    # Test draw of the quantum circuit
    def test_draw_circuit_from_list(self, qubit_space=15, idx_space=5, sec_space=10):
        testdraw = ""
        # show the name of the quantum circuit
        testdraw += "Quantum Circuit: "+self.name+"\n"
        initial_qubit_space = idx_space+sec_space
        testdraw += initial_qubit_space*" "
        # add the qubit indices
        for i in range(self.qubits_num):
            testdraw += f"q{i}".ljust(qubit_space)
        testdraw += "\n"
        testdraw += initial_qubit_space*" "
        for i in range(self.qubits_num):
            testdraw += "|".ljust(qubit_space)
        testdraw += "\n"
        # add the gates
        # loop through each section
        check_idx = 0
        for sec in range(len(self.sec_list)):
            # add the section index
            sec_str = f"sec{sec}:"
            testdraw += sec_str+(sec_space-len(sec_str))*" "
            first_line_of_sec = True
            for idx in range(self.sec_list[sec][0], self.sec_list[sec][1]+1):
                if first_line_of_sec:
                    idx_str = f"{idx}:"
                    testdraw += idx_str+(idx_space-len(idx_str))*" "
                    first_line_of_sec = False
                else:
                    testdraw += (sec_space)*" "
                    testdraw += f"{idx}:"+(idx_space-len(str(idx))-1)*" "
                # add the gates for each qubit
                for qubit_idx in range(self.qubits_num):
                    if self.gate_list[qubit_idx][check_idx][0] == "BARRIER":
                        testdraw += " ".ljust(qubit_space)
                    elif self.gate_list[qubit_idx][check_idx][0] == "U3":
                        param = self.gate_list[qubit_idx][check_idx][1]
                        gate_str = f"U({param[0]:.1f},{param[1]:.1f},{param[2]:.1f})"
                        testdraw += gate_str + (qubit_space-(len(gate_str)))*" "
                    elif self.gate_list[qubit_idx][check_idx][0] == "CNOT":
                        # check whether the qubit is a target qubit or a control qubit
                        if self.gate_list[qubit_idx][check_idx][1][0] == 0:
                            gate_str = "CX o"
                            testdraw += gate_str + (qubit_space-(len(gate_str)))*" "
                        else:
                            gate_str = f"CX q[{self.gate_list[qubit_idx][check_idx][1][1]}]"
                            testdraw += gate_str + (qubit_space-(len(gate_str)))*" "
                    elif self.gate_list[qubit_idx][check_idx][0] == "X":
                        gate_str = "X" if self.gate_list[qubit_idx][check_idx][1] == 0 else "-X"
                        testdraw += gate_str + (qubit_space-(len(gate_str)))*" "
                    elif self.gate_list[qubit_idx][check_idx][0] == "Y":
                        gate_str = "Y" if self.gate_list[qubit_idx][check_idx][1] == 0 else "-Y"
                        testdraw += gate_str + (qubit_space-(len(gate_str)))*" "
                    elif self.gate_list[qubit_idx][check_idx][0] == "Z":
                        gate_str = "Z" if self.gate_list[qubit_idx][check_idx][1] == 0 else "-Z"
                        testdraw += gate_str + (qubit_space-(len(gate_str)))*" "
                    elif self.gate_list[qubit_idx][check_idx][0] == "I":
                        gate_str = "I" if self.gate_list[qubit_idx][check_idx][1] == 0 else "-I"
                        testdraw += gate_str + (qubit_space-(len(gate_str)))*" "
                    else:
                        continue
                testdraw += "\n"
                testdraw += (sec_space+idx_space)*" "
                for i in range(self.qubits_num):
                    testdraw += "|".ljust(qubit_space)
                testdraw += "\n"
                check_idx += 1
        print(testdraw)
    
    def debug_gate_list(self):
        for qubit_idx in range(self.qubits_num):
            print(f"qubit {qubit_idx}: \n {self.gate_list[qubit_idx]}")
    
    def debug_sec_list(self):
        for sec in range(len(self.sec_list)):
            print(f"section {sec}: \n {self.sec_list[sec]}")
                    
############# Helper functions ################

# The function to randomly select target and control qubit pair for CNOT gate
# return a list of lists, each list contains two elements, the first element is the target qubit index, the second element is the control qubit index
# -1 indicates no target or control qubit, and at least there should be one pair valid
def select_target_control_pair_rand(qubit_num_list):
    target_control_pair_list = []
    idx = len(qubit_num_list)-1
    valid_total = 0
    while idx > 0:
        # randomly select whether to have a target and control pair
        target_control_pair = random.randint(0, 1)
        if target_control_pair == 0:
            # check whether there are no valid pair until now, if so, and this is the last chance to have a pair, then force to have a pair
            if not ((valid_total == 0) and (idx == 1 or idx == 2)):
                target_control_pair_list.append([-1, -1])
                idx -= 2
                continue
        # randomly select the target qubit
        target_idx = random.randint(0, idx)
        target = qubit_num_list[target_idx]
        qubit_num_list[target_idx] = qubit_num_list[idx]
        qubit_num_list[idx] = target
        idx -= 1
        # randomly select the control qubit
        control_idx = random.randint(0, idx)
        control = qubit_num_list[control_idx]
        qubit_num_list[control_idx] = qubit_num_list[idx]
        qubit_num_list[idx] = control
        idx -= 1
        target_control_pair_list.append([target, control])
        valid_total += 1
    return target_control_pair_list



############################### quantum circuit under RC ##################################
class QuantumCircuitRC:
    def __init__ (self, qubits_num, circuit_name="test"):
        self.name = circuit_name
        self.qubits_num = qubits_num
        self.ideal_circuit = None
        self.ideal_circuit_sim_result = None # the simulator result for the ideal quantum circuit
        self.trials_qc_gate_list = [] # the list of gate_list based quantum circuits for trials
        self.trials_circuit_sim_result = [] # the list of simulator results for quantum circuits of all trials
    
    @staticmethod
    def from_json(json_file):
        qc = QuantumCircuit.from_json(json_file)
        qc_RC = QuantumCircuitRC(qc.qubits_num, qc.name)
        qc_RC.ideal_circuit = qc
        qc_RC.ideal_circuit_sim_result = qc_RC.ideal_circuit.apply_circuit()
        return qc_RC
    
    def generate_ideal_circuit_random(self, max_cycle, max_single_num_per_cycle, single_multi_qubit_gate=True):
        self.ideal_circuit = QuantumCircuit(self.qubits_num, self.name)
        self.ideal_circuit.generate_gate_random(max_cycle, max_single_num_per_cycle, single_multi_qubit_gate)
        self.ideal_circuit_sim_result = self.ideal_circuit.apply_circuit()
    
    @RandomCompiling.RandomCompile.applyRC(True)
    def deepcopy_ideal_circuit(self, circuit_name):
        if self.ideal_circuit is None:
            raise ValueError("Ideal circuit is not generated yet")
        qc = QuantumCircuit(self.qubits_num, circuit_name)
        # deep copy the gate_list
        for qubit_idx in range(len(self.ideal_circuit.gate_list)):
            for cycle in range(len(self.ideal_circuit.gate_list[qubit_idx])):
                # deep copy the gate info
                gate_info = []
                for i in range(len(self.ideal_circuit.gate_list[qubit_idx][cycle])):
                    if self.ideal_circuit.gate_list[qubit_idx][cycle][i] is list:
                        temp_list = []
                        for j in range(len(self.ideal_circuit.gate_list[qubit_idx][cycle][i])):
                            temp_list.append(self.ideal_circuit.gate_list[qubit_idx][cycle][i][j])
                        gate_info.append(temp_list)
                    else:
                        gate_info.append(self.ideal_circuit.gate_list[qubit_idx][cycle][i])
                qc.add_gate(qubit_idx, gate_info)
        # deep copy the sec_list
        for sec in range(len(self.ideal_circuit.sec_list)):
            sec_info = []
            for i in range(len(self.ideal_circuit.sec_list[sec])):
                sec_info.append(self.ideal_circuit.sec_list[sec][i])
            qc.sec_list.append(sec_info)
        return qc
    
    # The function to generate the trials quantum circuits based on the ideal quantum circuit
    def generate_trials_circuit(self, trials_num):
        for i in range(trials_num):
            qc = self.deepcopy_ideal_circuit(f"{self.name}_trial{i}")
            self.trials_qc_gate_list.append(qc)
    
    # The function to apply the quantum circuit to all trials
    def apply_circuit_all_trials(self, simulator='mqvector'):
        if len(self.trials_qc_gate_list) == 0:
            raise ValueError("No trials quantum circuit generated yet")
        for i in range(len(self.trials_qc_gate_list)):
            self.trials_circuit_sim_result.append(self.trials_qc_gate_list[i].apply_circuit(simulator))
            
    ####################### The following functions are for visualization ############################
    
    # The function to visualize the RC quantum circuit based on the trial number
    def visualize_RC_circuit_trial(self, trial_idx):
        if trial_idx >= len(self.trials_qc_gate_list):
            raise ValueError("Trial index out of range")
        self.trials_qc_gate_list[trial_idx].test_draw_circuit_from_list()
    
    # The function to visualize all trials
    def visualize_all_RC_circuit_trials(self):
        for i in range(len(self.trials_qc_gate_list)):
            self.visualize_RC_circuit_trial(i)
            print("\n")
    
    # The function to visualize the simulator results for all trials
    def visualize_all_RC_sim_results(self):
        if len(self.trials_circuit_sim_result) == 0:
            raise ValueError("No simulator results for trials generated yet")
        for i in range(len(self.trials_circuit_sim_result)):
            print(f"Trial {i}: \n {self.trials_circuit_sim_result[i].get_qs(True)}")
            print("\n")
    
 

# qubit_idx_list = [i for i in range(5)]
# print(select_target_control_pair_rand(qubit_idx_list))    
# qc = Circuit()
# qc += H(0) 
# qc += H(1)

# qc -= H(1)

# sim = Simulator('mqvector', 2)
# sim.apply_circuit(qc)
# print(sim)

# qc = QuantumCircuit(2)
# qc.generate_gate_random(3, 3, single_multi_qubit_gate=False)
# qc.test_draw_circuit_from_list()
# qc.to_json()

# qc = QuantumCircuit.from_json("test.json")
# qc.test_draw_circuit_from_list()
# sim = qc.apply_circuit()
# print(sim.get_qs(True))

# qubit = [i for i in range(2)]
# print(select_target_control_pair_rand(qubit))


# qc_RC = QuantumCircuitRC.from_json("test.json")
# qc_RC.generate_trials_circuit(4)
# qc_RC.visualize_all_RC_circuit_trials()


##### test single unit #####
qc_RC = QuantumCircuitRC(2, "test_single_unit")
qc_RC.generate_ideal_circuit_random(1, 1, single_multi_qubit_gate=False)
qc_RC.generate_trials_circuit(1)
qc_RC.visualize_all_RC_circuit_trials()
qc_RC.apply_circuit_all_trials()
# print(qc_RC.trials_circuit_sim_result[0].get_qs(True))
# print(qc_RC.ideal_circuit_sim_result)
sv = qc_RC.ideal_circuit_sim_result.get_pure_state_vector()


# qc_RC.visualize_all_RC_sim_results()




