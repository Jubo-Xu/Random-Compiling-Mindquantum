import numpy as np
from mindquantum.simulator import Simulator
from mindquantum.core.circuit import Circuit
from mindquantum.core.gates import Measure
from mindquantum.core.gates import H, X, Y, Z, CNOT, SWAP, RX, RY, RZ, U3, GlobalPhase, I
from mindquantum.core.circuit import GateSelector, SequentialAdder, MixerAdder, NoiseChannelAdder
from mindquantum.core.gates import DepolarizingChannel, AmplitudeDampingChannel, PhaseDampingChannel
from mindquantum.core.operators import Hamiltonian
from mindquantum.core.operators import QubitOperator
import random
import math
import cmath
import json
# import RandomCompile.RandomCompiling as RandomCompiling
import RandomCompiling

class QuantumCircuit:
    def __init__(self, qubits_num, circuit_name="test", NOISE=False):
        self.name = circuit_name
        self.qubits_num = qubits_num
        self.qubits_idx_list = [i for i in range(qubits_num)] # a list containing the qubit index
        self.gate_list = [[] for i in range(qubits_num)] # a two dimensional list containing the gates applied to each qubit, the first dimension is the qubit index, the second dimension is the quantum cycle
        self.sec_list = [] # a list containing the start and end indices of each section in a 2 element list, len(sec_list) = number of sections
        self.circuit = Circuit()
        self.NOISE = NOISE # The noise flag
        self.NOISE_LIST = [[] for i in range(qubits_num)] # The list contains whether a noise channel should be added after each gate 
    
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
    
    # Generate a single unit, which contains only one CNOT gate
    @staticmethod
    def generate_single_unit(circuit_name="test_single_unit"):
        qc = QuantumCircuit(2, circuit_name)
        # randomly generate the control and target qubit pair
        ct_pair = select_target_control_pair_rand(qc.qubits_idx_list)
        control_idx = ct_pair[0][1]
        target_idx = ct_pair[0][0]
        qc.add_gate(target_idx, ["CNOT", [0, control_idx]])
        qc.add_gate(control_idx, ["CNOT", [1, target_idx]])
        qc.sec_list.append([0, 0])
        return qc
    
    # The function to generate the noise list for single unit
    def generate_noise_list_single_unit(self):
        for i in range(self.qubits_num):
            self.NOISE_LIST[i].extend([False, True, False])
    
    def AddNoise(func):
        def wrapper(self, *args, **kwargs):
            if self.NOISE:
                cycle = args[0]
                # if cycle == 0:
                #     self.circuit += RX(np.pi/2).on(1)
                # loop through each qubit
                for qubit_idx in range(self.qubits_num):
                    # check whether a noise channel should be added after each gate
                    if self.NOISE_LIST[qubit_idx][cycle]:
                        if self.gate_list[qubit_idx][cycle][0] == "CNOT":
                            if self.gate_list[qubit_idx][cycle][1][0] == 1:
                                self.circuit += RX(0.5).on(qubit_idx)
                        # add the depolarizing noise
                        # self.circuit += DepolarizingChannel(0.2).on(qubit_idx)
                        # # add the amplitude damping noise
                        # self.circuit += AmplitudeDampingChannel(0.2).on(qubit_idx)
                        # # add the phase damping noise
                        # self.circuit += PhaseDampingChannel(0.2).on(qubit_idx)
                        continue
            # Call the actual function
            func(self, *args, **kwargs)
            # Add noise to the circuit if needed
            if self.NOISE:
                cycle = args[0]
                # loop through each qubit
                for qubit_idx in range(self.qubits_num):
                    # check whether a noise channel should be added after each gate
                    # self.circuit += RX(0.5).on(1)
                    if self.NOISE_LIST[qubit_idx][cycle]:
                        # # add the depolarizing noise
                        # self.circuit += DepolarizingChannel(0.1).on(qubit_idx)
                        # # add the amplitude damping noise
                        # self.circuit += AmplitudeDampingChannel(0.1).on(qubit_idx)
                        # # add the phase damping noise
                        # self.circuit += PhaseDampingChannel(0.1).on(qubit_idx)
                        continue
            return 
        return wrapper
    
    # The function to apply the quantum gate 
    @AddNoise
    def apply_gate_each_cycle(self, cycle):
        for qubit_idx in range(self.qubits_num):
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
    
    # The function to apply the quantum circuit to the simulator
    def mindspore_circuit_gen(self):
        for cycle in range(len(self.gate_list[0])):
            self.apply_gate_each_cycle(cycle)
    
    def apply_circuit(self, simulator='mqvector'):
        self.mindspore_circuit_gen()
        sim = Simulator(simulator, self.qubits_num)
        sim.apply_circuit(self.circuit)
        return sim
    
    # Test draw of the quantum circuit
    def test_draw_circuit_from_list(self, qubit_space=20, idx_space=5, sec_space=10):
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

# The function to transfer a bitstring into corresponding decimal index
def bitstring_to_decimal_idx(bitstring):
    idx = 0
    for i in range(len(bitstring)):
        idx += int(bitstring[i])*(2**(len(bitstring)-i-1))
    return idx



############################### quantum circuit under RC ##################################
class QuantumCircuitRC:
    def __init__ (self, qubits_num, circuit_name="test", single_unit=False):
        self.SINGLE_UNIT = single_unit
        self.name = circuit_name
        self.qubits_num = qubits_num
        self.ideal_circuit = None
        self.ideal_circuit_sim_result = None # the simulator result for the ideal quantum circuit
        self.ideal_circuit_noise = None # the noise version of ideal circuit
        self.ideal_circuit_noise_count_list = [0 for i in range(2**qubits_num)] # the list of count of each basis for the noise version of ideal circuit
        self.trials_qc_gate_list = [] # the list of gate_list based quantum circuits for trials
        self.trials_circuit_sim_result = [] # the list of simulator results for quantum circuits of all trials
        self.trials_count_list = [0 for i in range(2**qubits_num)] # the list of count of each basis for all trials
        self.trials_combined_circuits = [] # the list of combined quantum circuits for all trials
        self.trials_combined_circuit_sim_result = [] # the list of simulator results for combined quantum circuits for all trials
        self.ideal_circuit_noise_prob_list = [0 for i in range(2**qubits_num)] # the list of probability of each basis for the noise version of ideal circuit
        self.trials_prob_list = [0 for i in range(2**qubits_num)] # the list of probability of each basis for all trials
        self.ideal_prob_list = [0 for i in range(2**qubits_num)] # the list of probability of each basis for the ideal circuit
    
    @staticmethod
    def from_json(json_file):
        qc = QuantumCircuit.from_json(json_file)
        qc_RC = QuantumCircuitRC(qc.qubits_num, qc.name)
        qc_RC.ideal_circuit = qc
        qc_RC.ideal_circuit_sim_result = qc_RC.ideal_circuit.apply_circuit()
        return qc_RC
    
    def generate_ideal_circuit_random(self, max_cycle=1, max_single_num_per_cycle=1, single_multi_qubit_gate=True):
        self.ideal_circuit = QuantumCircuit(self.qubits_num, self.name)
        if self.SINGLE_UNIT:
            self.ideal_circuit = QuantumCircuit.generate_single_unit(self.name)
        else:
            self.ideal_circuit.generate_gate_random(max_cycle, max_single_num_per_cycle, single_multi_qubit_gate)
        self.ideal_circuit_sim_result = self.ideal_circuit.apply_circuit()
        state_vector = self.ideal_circuit_sim_result.get_pure_state_vector()
        # update the probability list for the ideal circuit
        for i in range(len(state_vector)):
            self.ideal_prob_list[i] = (state_vector[i].real)**2+(state_vector[i].imag)**2
    
    @RandomCompiling.RandomCompile.applyRC(True, 'SINGLE_UNIT')
    def deepcopy_ideal_circuit(self, circuit_name, ideal_copy=False, noise=False):
        if self.ideal_circuit is None:
            raise ValueError("Ideal circuit is not generated yet")
        qc = QuantumCircuit(self.qubits_num, circuit_name, noise)
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
    def generate_trials_circuit(self, trials_num, noise=True):
        for i in range(trials_num):
            qc = self.deepcopy_ideal_circuit(f"{self.name}_trial{i}", noise=noise)
            self.trials_qc_gate_list.append(qc)
    
    # The function to generate the noise version of ideal_circuit
    def generate_ideal_circuit_noise(self):
        qc = self.deepcopy_ideal_circuit(f"{self.name}_noise", ideal_copy=True, noise=True)
        # generate the noise list, here just set all to True
        for i in range(self.qubits_num):
            for j in range(len(qc.gate_list[i])):
                qc.NOISE_LIST[i].append(True)
        # apply the circuit
        qc.mindspore_circuit_gen()
        self.ideal_circuit_noise = qc
        return
    
    # The function to apply the quantum circuit to all trials
    def apply_circuit_all_trials(self, simulator='mqvector'):
        if len(self.trials_qc_gate_list) == 0:
            raise ValueError("No trials quantum circuit generated yet")
        for i in range(len(self.trials_qc_gate_list)):
            self.trials_circuit_sim_result.append(self.trials_qc_gate_list[i].apply_circuit(simulator))
    
    # The function to apply the combination for the RC quantum circuits
    def generate_circuit_combination_all_trials(self):
        if len(self.trials_qc_gate_list) == 0:
            raise ValueError("No trials quantum circuit generated yet")
        for i in range(len(self.trials_qc_gate_list)):
            qc_comb = QuantumCircuit(self.qubits_num, f"{self.name}_trial{i}_combined")
            self.trials_combined_circuits.append(RandomCompiling.RandomCompile.applyRC_combine(qc_comb, self.trials_qc_gate_list[i]))
    
    # The function to apply the quantum circuit to all combined trials
    def apply_circuit_combination_all_trials(self, simulator='mqvector'):
        if len(self.trials_combined_circuits) == 0:
            raise ValueError("No combined quantum circuit generated yet")
        for i in range(len(self.trials_combined_circuits)):
            self.trials_combined_circuit_sim_result.append(self.trials_combined_circuits[i].apply_circuit(simulator))   
            
    # The function to apply the circuit to single units
    def apply_circuit_single_unit(self):
        if not self.SINGLE_UNIT:
            raise ValueError("The circuit is not a single unit")
        for i in range(len(self.trials_qc_gate_list)):
            # generate the noise lists for all trials
            self.trials_qc_gate_list[i].generate_noise_list_single_unit()
            # apply the circuit to all trials
            self.trials_qc_gate_list[i].mindspore_circuit_gen()
        return
    
    # The function to get the count list and probability list for all trials
    def get_count_prob_list_all_trials(self, sim):
        for trial in range(len(self.trials_qc_gate_list)):
            # add the measurement gate for all qubits
            for qubit_idx in range(self.qubits_num):
                self.trials_qc_gate_list[trial].circuit += Measure(f"q{qubit_idx}").on(qubit_idx)
            # sampling the circuit
            result_dict = sim.sampling(self.trials_qc_gate_list[trial].circuit, shots=1).bit_string_data
            # update the count list
            for bitstring in result_dict:
                idx = bitstring_to_decimal_idx(bitstring)
                self.trials_count_list[idx] += result_dict[bitstring]
        # update the probability list
        num_trials = len(self.trials_qc_gate_list)
        for i in range(len(self.trials_count_list)):
            self.trials_prob_list[i] = self.trials_count_list[i]/num_trials
        return
    
    # The function to generate a test mainly for single unit
    @staticmethod
    def test_single_unit(num_trials=1000, simulator="mqvector"):
        qc_RC = QuantumCircuitRC(2, "test_single_unit", True)
        qc_RC.generate_ideal_circuit_random()
        qc_RC.generate_trials_circuit(num_trials)
        qc_RC.generate_ideal_circuit_noise()
        qc_RC.apply_circuit_single_unit()
        # sampling the noise version of ideal circuit to get the count list
        # add the measurement gate for all qubits
        for qubit_idx in range(qc_RC.qubits_num):
            qc_RC.ideal_circuit_noise.circuit += Measure(f"q{qubit_idx}").on(qubit_idx)
        # sampling the circuit
        sim = Simulator(simulator, qc_RC.qubits_num)
        result_dict_ideal = sim.sampling(qc_RC.ideal_circuit_noise.circuit, shots=num_trials).bit_string_data
        for bitstring in result_dict_ideal:
            idx = bitstring_to_decimal_idx(bitstring)
            qc_RC.ideal_circuit_noise_count_list[idx] = result_dict_ideal[bitstring]
        # update the probability list for the noise version of ideal circuit
        for i in range(len(qc_RC.ideal_circuit_noise_count_list)):
            qc_RC.ideal_circuit_noise_prob_list[i] = qc_RC.ideal_circuit_noise_count_list[i]/num_trials
        
        # sampling all trials
        qc_RC.get_count_prob_list_all_trials(sim)
        return qc_RC
    
        
            
    ####################### The following functions are for visualization ############################
    
    # The function to visualize the ideal quantum circuit
    def visualize_ideal_circuit(self):
        if self.ideal_circuit is None:
            raise ValueError("Ideal circuit is not generated yet")
        self.ideal_circuit.test_draw_circuit_from_list()
    
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
    
    # The function to visualize the combined circuits of all trials
    def visualize_combined_circuit_all_trials(self):
        for i in range(len(self.trials_combined_circuits)):
            self.trials_combined_circuits[i].test_draw_circuit_from_list()
            print("\n")
    
    # The function to visualize the simulator results for all combined trials
    def visualize_all_RC_combined_sim_results(self):
        if len(self.trials_combined_circuit_sim_result) == 0:
            raise ValueError("No simulator results for combined trials generated yet")
        for i in range(len(self.trials_combined_circuit_sim_result)):
            print(f"Trial {i}: \n {self.trials_combined_circuit_sim_result[i].get_qs(True)}")
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
# qc_RC = QuantumCircuitRC(3, "test_single_unit")
# qc_RC.generate_ideal_circuit_random(3, 1, single_multi_qubit_gate=False)
# qc_RC.generate_trials_circuit(1)
# qc_RC.visualize_ideal_circuit()
# qc_RC.visualize_all_RC_circuit_trials()
# qc_RC.apply_circuit_all_trials()
# # print(qc_RC.trials_circuit_sim_result[0].get_qs(True))
# # print(qc_RC.ideal_circuit_sim_result)
# sv = qc_RC.ideal_circuit_sim_result.get_pure_state_vector()
# print(sv)
# print(qc_RC.trials_circuit_sim_result[0].get_pure_state_vector())
# # qc_RC.visualize_all_RC_sim_results()




# def test_RC_gate_functionality_hypothesis(n_qubit, n_trials, n_max_cycle):
#     qc_RC = QuantumCircuitRC(n_qubit)
#     qc_RC.generate_ideal_circuit_random(n_max_cycle, 1, single_multi_qubit_gate=False)
#     qc_RC.generate_trials_circuit(n_trials)
#     qc_RC.apply_circuit_all_trials()
#     ideal_sv = qc_RC.ideal_circuit_sim_result.get_pure_state_vector()
#     print(ideal_sv)
    
#     print(qc_RC.trials_circuit_sim_result[0].get_pure_state_vector())
#     # trial_sv = []
#     # for i in range(len(qc_RC.trials_circuit_sim_result)):
#     #     trial_sv.append(qc_RC.trials_circuit_sim_result[i].get_pure_state_vector())
#     # print(ideal_sv)
#     # print(trial_sv)
#     # for i in range(len(trial_sv)):
#     #     for j in range(len(ideal_sv)):
#     #         # Check the real part of the amplitude
#     #         assert ideal_sv[j].real == trial_sv[i][j].real
#     #         # Check the imaginary part of the amplitude
#     #         assert ideal_sv[j].imag == trial_sv[i][j].imag


# test combined circuit
# qc_RC = QuantumCircuitRC(2, "test")
# qc_RC.generate_ideal_circuit_random(1, 1, single_multi_qubit_gate=False)
# qc_RC.generate_trials_circuit(1)
# qc_RC.generate_circuit_combination_all_trials()
# qc_RC.apply_circuit_combination_all_trials()
# qc_RC.apply_circuit_all_trials()
# qc_RC.visualize_ideal_circuit()
# qc_RC.visualize_all_RC_circuit_trials()
# qc_RC.visualize_combined_circuit_all_trials()
# sv = qc_RC.ideal_circuit_sim_result.get_pure_state_vector()
# print(sv)
# print(qc_RC.trials_circuit_sim_result[0].get_pure_state_vector())
# print(qc_RC.trials_combined_circuit_sim_result[0].get_pure_state_vector())
# # qc_RC.visualize_all_RC_sim_results()
# print(qc_RC.trials_combined_circuits[0].gate_list)
# print(qc_RC.trials_combined_circuits[0].circuit)




# test single unit 
# qc = QuantumCircuit.generate_single_unit()
# qc.test_draw_circuit_from_list()

# qc_RC = QuantumCircuitRC(2, "test_single_unit", True)
# qc_RC.generate_ideal_circuit_random()
# qc_RC.generate_trials_circuit(1)
# qc_RC.visualize_ideal_circuit()
# qc_RC.visualize_all_RC_circuit_trials()
# qc_RC.generate_ideal_circuit_noise()
# qc_RC.apply_circuit_single_unit()
# print(qc_RC.ideal_circuit_noise.circuit)
# print(qc_RC.trials_qc_gate_list[0].circuit)

# qc_RC.ideal_circuit_noise.circuit += Measure("q0").on(0)
# qc_RC.ideal_circuit_noise.circuit += Measure("q1").on(1)
# sim = Simulator('mqvector', 2)
# result = sim.sampling(qc_RC.ideal_circuit_noise.circuit, shots=10)
# print(result)
# # print(result.samples)
# print(result.bit_string_data)
# for key in result.bit_string_data:
#     print(f"{key}: {result.bit_string_data[key]}")
# print(qc_RC.ideal_circuit_sim_result.get_pure_state_vector())



# qc_RC = QuantumCircuitRC.test_single_unit(num_trials=1000)
# print(qc_RC.ideal_circuit_noise.circuit)
# print(qc_RC.trials_qc_gate_list[0].circuit)
# no_RC_fidelity, RC_fidelity = RandomCompiling.RandomCompile.Fidelity_Evaluation(qc_RC.ideal_prob_list, qc_RC.ideal_circuit_noise_prob_list, qc_RC.trials_prob_list)
# print(no_RC_fidelity)
# print(RC_fidelity)

# # print("ideal:")
# # print(qc_RC.ideal_prob_list)
# print("no RC:")
# print(qc_RC.ideal_circuit_noise_prob_list)
# print("RC:")
# print(qc_RC.trials_prob_list)




# print(qc_RC.ideal_prob_list)
# print(qc_RC.ideal_circuit_noise_prob_list)
# print(qc_RC.trials_prob_list)
# print(qc_RC.ideal_circuit_noise.circuit)
# for i in range(len(qc_RC.trials_qc_gate_list)):
#     print(qc_RC.trials_qc_gate_list[i].circuit)

# qc_test = Circuit()
# qc_test += RX(np.pi/2).on(1)
# qc_test += X.on(0, 1)

# sim = Simulator('mqvector', 2)
# sim.apply_circuit(qc_test)
# print(qc_test)
# print(sim.get_qs(True))
