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
import utils


class QAOAMaxCut_QuantumCircuit:
    def __init__(self, name, num_qubit, p, graph, noise=None):
        self.name = name
        self.graph = graph
        self.num_qubit = num_qubit
        self.p = p
        self.noise = noise
        self.circuit = Circuit()
        self.gate_list = [[] for i in range(num_qubit)]
        self.sec_list = []
        self.sim = None
    
    # The function to generate the cost layer
    def cost_layer_gen(self, gamma, RC=False):
        # the list to track the used qubits
        used_qubits = [0 for i in range(self.num_qubit)]
        for edge, weight in zip(self.graph["edges"], self.graph["weights"]):
            control_qubit = edge[0]
            target_qubit = edge[1]
            if used_qubits[control_qubit] != used_qubits[target_qubit]:
                if used_qubits[control_qubit] < used_qubits[target_qubit]:
                    for i in range(used_qubits[target_qubit]-used_qubits[control_qubit]):
                        self.gate_list[control_qubit].append(["BARRIER"])
                    used_qubits[control_qubit] = used_qubits[target_qubit]
                else:
                    for i in range(used_qubits[control_qubit]-used_qubits[target_qubit]):
                        self.gate_list[target_qubit].append(["BARRIER"])
                    used_qubits[target_qubit] = used_qubits[control_qubit]
            if not RC:
                self.gate_list[control_qubit].append(["CNOT", [0, target_qubit]]) # the control qubit is set as 0
                self.gate_list[target_qubit].append(["CNOT", [1, control_qubit]]) # the target qubit is set as 1
                self.gate_list[control_qubit].append(["BARRIER"])
                self.gate_list[target_qubit].append(["RZ", [-2*gamma*weight]])
                self.gate_list[control_qubit].append(["CNOT", [0, target_qubit]]) # the control qubit is set as 0
                self.gate_list[target_qubit].append(["CNOT", [1, control_qubit]]) # the target qubit is set as 1
                used_qubits[target_qubit] += 3
                used_qubits[control_qubit] += 3
            else:
                # generate the first RC group for the first CNOT
                control_rng = random.randint(0, len(RandomCompiling.RCGateTable) - 1)
                target_rng = random.randint(0, len(RandomCompiling.RCGateTable) - 1)
                control_RC = RandomCompiling.RCGateTable[control_rng]
                target_RC = RandomCompiling.RCGateTable[target_rng]
                RC_str = control_RC + target_RC
                RC_comp = RandomCompiling.RCGateComplementaryMapping[RC_str]
                control_RC_comp = RC_comp[0][0]
                target_RC_comp = RC_comp[0][1]
                control_RC_comp_sign = RC_comp[1]
                target_RC_comp_sign = RC_comp[2]
                # add the gates
                self.gate_list[control_qubit].append([control_RC, 0]) # the first element is the gate name, the second element is the sign
                self.gate_list[target_qubit].append([target_RC, 0])
                self.gate_list[control_qubit].append(["CNOT", [0, target_qubit]])
                self.gate_list[target_qubit].append(["CNOT", [1, control_qubit]])
                self.gate_list[control_qubit].append([control_RC_comp, control_RC_comp_sign])
                self.gate_list[target_qubit].append([target_RC_comp, target_RC_comp_sign])
                # add the RZ gate
                self.gate_list[control_qubit].append(["BARRIER"])
                self.gate_list[target_qubit].append(["RZ", [-2*gamma*weight]])
                # generate the second RC group for the second CNOT
                control_rng_2 = random.randint(0, len(RandomCompiling.RCGateTable) - 1)
                target_rng_2 = random.randint(0, len(RandomCompiling.RCGateTable) - 1)
                control_RC_2 = RandomCompiling.RCGateTable[control_rng_2]
                target_RC_2 = RandomCompiling.RCGateTable[target_rng_2]
                RC_str_2 = control_RC_2 + target_RC_2
                RC_comp_2 = RandomCompiling.RCGateComplementaryMapping[RC_str_2]
                control_RC_comp_2 = RC_comp_2[0][0]
                target_RC_comp_2 = RC_comp_2[0][1]
                control_RC_comp_sign_2 = RC_comp_2[1]
                target_RC_comp_sign_2 = RC_comp_2[2]
                # add the gates
                self.gate_list[control_qubit].append([control_RC_2, 0])
                self.gate_list[target_qubit].append([target_RC_2, 0])
                self.gate_list[control_qubit].append(["CNOT", [0, target_qubit]])
                self.gate_list[target_qubit].append(["CNOT", [1, control_qubit]])
                self.gate_list[control_qubit].append([control_RC_comp_2, control_RC_comp_sign_2])
                self.gate_list[target_qubit].append([target_RC_comp_2, target_RC_comp_sign_2])
                # update the used_qubits
                used_qubits[control_qubit] += 7
                used_qubits[target_qubit] += 7
        # add the barrier for the remaining qubits
        max_len = 0
        for i in range(len(used_qubits)):
            max_len = used_qubits[i] if used_qubits[i] > max_len else max_len
        for i in range(len(used_qubits)):
            if used_qubits[i] < max_len:
                for j in range(max_len-used_qubits[i]):
                    self.gate_list[i].append(["BARRIER"])
        # update the sec_list
        self.sec_list[-1][1] += max_len-1
        return  
    
    # The function to generate the mixer layer
    def mixer_layer_gen(self, beta):
        for i in range(self.num_qubit):
            self.gate_list[i].append(["RX", [2*beta]])
        # update the sec_list
        self.sec_list[-1][1] += 1
    
    # The function to generate the QAOA circuit
    def QAQA_circuit_gen(self, gamma_list, beta_list, RC=False):
        # first apply the hadamard gate
        for i in range(self.num_qubit):
            self.gate_list[i].append(["H"])
        # update the sec_list
        sec = [0,0]
        self.sec_list.append(sec)
        # apply the cost and mixer layers 
        for num_p in range(self.p):
            self.sec_list.append([self.sec_list[-1][1]+1, self.sec_list[-1][1]+1])
            self.cost_layer_gen(gamma_list[num_p], RC)
            self.mixer_layer_gen(beta_list[num_p])
    
    # The decorator to add the noise channel
    def AddNoise(func):
        def wrapper(self, *args, **kwargs):
            qubit = args[0]
            cycle = args[1]
            if self.noise:
                # add the coherent noise for control qubit of the CNOT gate
                if self.gate_list[qubit][cycle][0] == "CNOT" and self.gate_list[qubit][cycle][1][0] == 0:
                    self.circuit += RX(0.5).on(qubit)
                # add the depolarizing noise
                # self.circuit += DepolarizingChannel(0.02).on(qubit)
                # # add the amplitude damping noise
                # self.circuit += AmplitudeDampingChannel(0.02).on(qubit)
                # # add the phase damping noise
                # self.circuit += PhaseDampingChannel(0.02).on(qubit)
            # Call the actual function
            func(self, *args, **kwargs)
            # Add noise to the circuit if needed
            # if self.noise:
            #     # add the depolarizing noise
            #     # self.circuit += DepolarizingChannel(0.2).on(qubit)
            #     # # add the amplitude damping noise
            #     # self.circuit += AmplitudeDampingChannel(0.2).on(qubit)
            #     # # add the phase damping noise
            #     # self.circuit += PhaseDampingChannel(0.2).on(qubit)
            return 
        return wrapper
    
    
    # The function to generate the mindspore circuit
    @AddNoise
    def circuit_gen_single_gate(self, qubit, cycle):
        gate_info = self.gate_list[qubit][cycle]
        # check whether the CNOT gate is applied and the qubit is the control qubit
        if gate_info[0] == "CNOT" and gate_info[1][0] == 0:
            self.circuit += X.on(gate_info[1][1], qubit)
        elif gate_info[0] == "RX":
            self.circuit += RX(gate_info[1][0]).on(qubit)
        elif gate_info[0] == "RZ":
            self.circuit += RZ(gate_info[1][0]).on(qubit)
        elif gate_info[0] == "H":
            self.circuit += H.on(qubit)
        elif gate_info[0] == "X":
            if gate_info[1] == 0:
                self.circuit += X.on(qubit)
            else:
                self.circuit += X.on(qubit)
                self.circuit += GlobalPhase(math.pi).on(qubit)
        elif gate_info[0] == "Y":
            if gate_info[1] == 0:
                self.circuit += Y.on(qubit)
            else:
                self.circuit += Y.on(qubit)
                self.circuit += GlobalPhase(math.pi).on(qubit)
        elif gate_info[0] == "Z":
            if gate_info[1] == 0:
                self.circuit += Z.on(qubit)
            else:
                self.circuit += Z.on(qubit)
                self.circuit += GlobalPhase(math.pi).on(qubit)
        elif gate_info[0] == "I":
            if gate_info[1] == 0:
                self.circuit += I.on(qubit)
            else:
                self.circuit += I.on(qubit)
                self.circuit += GlobalPhase(math.pi).on(qubit)
        else:
            return
    
    def circuit_gen(self):
        for cycle in range(len(self.gate_list[0])):
            for qubit in range(self.num_qubit):
                self.circuit_gen_single_gate(qubit, cycle)
        return
    
    def apply_circuit(self, simulator='mqvector'):
        self.sim = Simulator(simulator, self.num_qubit)
        self.sim.apply_circuit(self.circuit)
        return
    
    @staticmethod
    def from_graph(name, graph, num_qubit, beta_list, gamma_list, noise=False, RC=False):
        p = len(beta_list)
        qc = QAOAMaxCut_QuantumCircuit(name, num_qubit, p, graph, noise)
        qc.QAQA_circuit_gen(gamma_list, beta_list, RC)
        qc.circuit_gen()
        return qc

    def test_gate_list(self):
        for i in range(self.num_qubit):
            print(f"q{i}")
            print(self.gate_list[i])
            
    # The function to visualize the circuit
    def test_draw_circuit_from_list(self, qubit_space=20, idx_space=5, sec_space=10):
        testdraw = ""
        # show the name of the quantum circuit
        testdraw += "Quantum Circuit: "+self.name+"\n"
        initial_qubit_space = idx_space+sec_space
        testdraw += initial_qubit_space*" "
        # add the qubit indices
        for i in range(self.num_qubit):
            testdraw += f"q{i}".ljust(qubit_space)
        testdraw += "\n"
        testdraw += initial_qubit_space*" "
        for i in range(self.num_qubit):
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
                for qubit_idx in range(self.num_qubit):
                    if self.gate_list[qubit_idx][check_idx][0] == "BARRIER":
                        testdraw += " ".ljust(qubit_space)
                    elif self.gate_list[qubit_idx][check_idx][0] == "U3":
                        param = self.gate_list[qubit_idx][check_idx][1]
                        gate_str = f"U({param[0]:.1f},{param[1]:.1f},{param[2]:.1f})"
                        testdraw += gate_str + (qubit_space-(len(gate_str)))*" "
                    elif self.gate_list[qubit_idx][check_idx][0] == "H":
                        gate_str = "H"
                        testdraw += gate_str + (qubit_space-(len(gate_str)))*" "
                    elif self.gate_list[qubit_idx][check_idx][0] == "RZ":
                        param = self.gate_list[qubit_idx][check_idx][1]
                        gate_str = f"RZ({param[0]:.1f})"
                        testdraw += gate_str + (qubit_space-(len(gate_str)))*" "
                    elif self.gate_list[qubit_idx][check_idx][0] == "RX":
                        param = self.gate_list[qubit_idx][check_idx][1]
                        gate_str = f"RX({param[0]:.1f})"
                        testdraw += gate_str + (qubit_space-(len(gate_str)))*" "
                    elif self.gate_list[qubit_idx][check_idx][0] == "CNOT":
                        # check whether the qubit is a target qubit or a control qubit
                        if self.gate_list[qubit_idx][check_idx][1][0] == 1:
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
                for i in range(self.num_qubit):
                    testdraw += "|".ljust(qubit_space)
                testdraw += "\n"
                check_idx += 1
        print(testdraw)
    
    def visualize(self, mode="testdraw"):
        if mode == "testdraw":
            self.test_draw_circuit_from_list()
        elif mode == "mindspore":
            print(self.name)
            print(self.circuit)
        else:
            print("Invalid mode")


class QAOAMaxCut_QuantumCircuit_RC:
    def __init__(self, num_qubit, graph, num_trials, gamma_list, beta_list):
        self.num_qubit = num_qubit
        self.p = len(beta_list)
        self.graph = graph
        self.num_trials = num_trials
        self.ideal_circuit = None
        self.noisy_circuit = None
        self.RC_circuits = []
        self.gamma_list = gamma_list
        self.beta_list = beta_list
        self.ideal_prob_list = [0 for i in range(2**self.num_qubit)]
        self.noisy_no_RC_prob_list = [0 for i in range(2**self.num_qubit)]
        self.noisy_with_RC_prob_list = [0 for i in range(2**self.num_qubit)]
    
    # The function to generate the RC trials
    def RC_trials_gen(self):
        for i in range(self.num_trials):
            qc = QAOAMaxCut_QuantumCircuit.from_graph("RC_trial_"+str(i), self.graph, self.num_qubit, self.beta_list, self.gamma_list, noise=True, RC=True)
            for qubit in range(self.num_qubit):
                qc.circuit += Measure(f"q{qubit}").on(qubit)
            self.RC_circuits.append(qc)
    
    # The function to generate the ideal probabilities of each basis
    def ideal_prob_gen(self):
        self.ideal_circuit.apply_circuit() 
        state_vector = self.ideal_circuit.sim.get_pure_state_vector()
        # update the probability list for the ideal circuit
        for i in range(len(state_vector)):
            self.ideal_prob_list[i] = (state_vector[i].real)**2+(state_vector[i].imag)**2
    
    # The function to generate the probability list of the noisy ideal circuit
    def ideal_with_noise_prob_gen(self, simulator="mqvector"):
        for qubit in range(self.num_qubit):
            self.noisy_circuit.circuit += Measure(f"q{qubit}").on(qubit)
        # sampling the circuit
        sim = Simulator(simulator, self.num_qubit)
        result_dict_ideal = sim.sampling(self.noisy_circuit.circuit, shots=self.num_trials).bit_string_data
        for bitstring in result_dict_ideal:
            idx = utils.bitstring_to_decimal_idx(bitstring)
            self.noisy_no_RC_prob_list[idx] = result_dict_ideal[bitstring]
        # update the probability list for the noise version of ideal circuit
        for i in range(len(self.noisy_no_RC_prob_list)):
            self.noisy_no_RC_prob_list[i] = self.noisy_no_RC_prob_list[i]/self.num_trials
        return
    
    # The function to generate the probability list of the RC trials
    def RC_with_noise_prob_gen(self, simulator="mqvector"):
        sim = Simulator(simulator, self.num_qubit)
        for trial in range(self.num_trials):
            # sampling the circuit
            result_dict = sim.sampling(self.RC_circuits[trial].circuit, shots=1).bit_string_data
            # update the count list
            for bitstring in result_dict:
                idx = utils.bitstring_to_decimal_idx(bitstring)
                self.noisy_with_RC_prob_list[idx] += result_dict[bitstring]
        # update the probability list
        for i in range(len(self.noisy_with_RC_prob_list)):
            self.noisy_with_RC_prob_list[i] = self.noisy_with_RC_prob_list[i]/self.num_trials
        return
    
    # The function for testing from graph and randomly generating the beta and gamma lists based on the specified p
    @staticmethod
    def test_from_graph(graph, num_qubit, p, num_trials):
        beta_list = [random.uniform(0, 2*math.pi) for i in range(p)]
        gamma_list = [random.uniform(0, 2*math.pi) for i in range(p)]
        qc = QAOAMaxCut_QuantumCircuit_RC(num_qubit, graph, num_trials, gamma_list, beta_list)
        # generate the ideal circuit
        qc.ideal_circuit = QAOAMaxCut_QuantumCircuit.from_graph("ideal", graph, num_qubit, beta_list, gamma_list, noise=False, RC=False)
        # generate the noisy ideal circuit
        qc.noisy_circuit = QAOAMaxCut_QuantumCircuit.from_graph("noisy_ideal", graph, num_qubit, beta_list, gamma_list, noise=True, RC=False)
        # generate the RC trials
        qc.RC_trials_gen()
        # generate the probability lists 
        qc.ideal_prob_gen()
        qc.ideal_with_noise_prob_gen()
        qc.RC_with_noise_prob_gen()
        return qc
    
    # The following function is for visualizations
    # The function to visualize the circuits
    def visualize_circuits(self, mode="mindspore"):
        print("Ideal Circuit:")
        self.ideal_circuit.visualize(mode)
        print("Noisy Ideal Circuit:")
        self.noisy_circuit.visualize(mode)
        print("RC Trials:")
        for trial in self.RC_circuits:
            trial.visualize(mode)

    # The function to visualize the probability lists
    def visualize_probabilities(self):
        utils.print_colored("Ideal Probability:", "blue")
        print(self.ideal_prob_list)
        utils.print_colored("Noisy Ideal Probability:", "blue")
        print(self.noisy_no_RC_prob_list)
        utils.print_colored("RC with Noise Probability:", "blue")
        print(self.noisy_with_RC_prob_list)
        no_RC_fidelity, RC_fidelity = RandomCompiling.RandomCompile.Fidelity_Evaluation(self.ideal_prob_list, self.noisy_no_RC_prob_list, self.noisy_with_RC_prob_list)
        utils.print_colored("Error Rate check:", "blue")
        print(f"No RC Error Rate: {no_RC_fidelity}", f"RC Error Rate: {RC_fidelity}")
    



graph = {
    'edges': [(0, 1), (1, 2), (2, 3), (3, 0)],
    'weights': [1, 1, 1, 1]
}

# graph = {
#     'edges': [(0, 1), (1, 2), (2, 3), (3, 4), (0, 4), (0, 2)],
#     'weights': [1, 1, 1, 1, 1, 1]
# }
# beta_list = [0.1, 0.2, 0.3]
# gamma_list = [1, 2, 3]

# beta_list = [0.1]
# gamma_list = [1]
# qc = QAOAMaxCut_QuantumCircuit.from_graph("test", graph, 4, beta_list, gamma_list, noise=True, RC=False)
# # qc.test_gate_list()
# # qc.test_draw_circuit_from_list()
# qc.visualize("mindspore")
 
 
qc = QAOAMaxCut_QuantumCircuit_RC.test_from_graph(graph, 4, 1, 1000)
# qc.visualize_circuits()
qc.visualize_probabilities()
