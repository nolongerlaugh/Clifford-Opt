from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Operator
from qiskit.providers.aer import AerSimulator
from qiskit.quantum_info import state_fidelity

import numpy as np
from collections import defaultdict
from tqdm import tqdm
import pandas as pd

np.random.seed(239)
native_gates = ['rz', 'cz', 'rx']

def get_duration_prob(name, params, T2=2000, T_RX=100, T_RZ=30, T_CZ=15): #mks
    if (name == 'rx' or name == 'ry'):
        theta = params[0] % (2 * np.pi)
        return theta / np.pi * 5, 1 - np.exp(-theta / T_RX / np.pi)
    if (name == 'rz'):
        theta = params[0] % (2 * np.pi)
        return theta / np.pi * 0.5, 1 - np.exp(-theta / T_RZ / np.pi)
    if (name == 'cz'):
        return 2, 1 - np.exp(-2. / T_CZ)
    if (name == 'id'):
        if len(params):
            t = params[0]
            return t, 1 - np.exp(-t/T2)
        return 0, 1
    return None

def make_native_circuit(t_circ : QuantumCircuit, opt_level=0, T2=2000, T_RX=100, T_RZ=30, T_CZ=15) -> (QuantumCircuit, dict):
    circuit = transpile(t_circ, basis_gates=native_gates, optimization_level=opt_level)
    time = 0.0 #mks
    new_circuit = QuantumCircuit(*circuit.qregs, *circuit.cregs)
    new_circuit.name = t_circ.name
    data = []
    times = np.zeros(circuit.num_qubits)
    for instr, qargs, cargs in circuit.data:
        if (instr.name in native_gates):
            for qarg in qargs:
                index = circuit.find_bit(qarg).index     
                # id gate:
                if times[index] != time:
                    new_circuit.id(qarg)
                    duration_id, prob_id = get_duration_prob('id', [time - times[index]], T2=T2)
                    data.append(prob_id)
                    times[index] = time
            # gate:
            duration, prob = get_duration_prob(instr.name, instr.params, T2=T2, T_RX=T_RX, T_RZ=T_RZ, T_CZ=T_CZ)
            new_circuit.append(instr, qargs, cargs)
            data.append(prob)
            time = time + duration
            for qarg in qargs:
                times[index] = time
    for qreg in new_circuit.qregs:
        for qarg in qreg:
            index = new_circuit.find_bit(qarg).index
            if times[index] != time:
                new_circuit.id(qarg)
                duration, prob = get_duration_prob('id', [time - times[index]], T2=T2)
                data.append(prob)        
    return new_circuit, data

#to check (measuring -> measure = True)
def fidelity_density_matrix_orig(circuit :QuantumCircuit, T2=2000, T_RX=100, T_RZ=30, T_CZ=15, measure=True):
    circ, d = make_native_circuit(circuit, opt_level=0, T2=T2, T_RX=T_RX, T_RZ=T_RZ, T_CZ=T_CZ)
    noiseless_circ = circ.copy()
    noiseless_circ.save_density_matrix()
    g = AerSimulator(method='density_matrix')

    rho1 = np.array(g.run(noiseless_circ, shots=1, noise_model=None, optimization_level=0).result().data()['density_matrix'])
    rho2 = np.zeros_like(rho1)
    rho2[0, 0] = 1
    for i, (instr, qargs, cargs) in enumerate(circ.data):
        qc = QuantumCircuit(*circ.qregs, *circ.cregs)
        qc.append(instr, qargs, cargs)
        U = Operator(qc).to_matrix()
        qc = QuantumCircuit(*circ.qregs, *circ.cregs)
        qc.z(qargs)
        Z = Operator(qc).to_matrix()
        rho2 = U @ rho2 @ U.T.conj()
        prob = d[i]
        #rho2 = (1 - prob) * rho2 + prob * np.eye(2 ** circ.num_qubits) / (2 ** circ.num_qubits)
        rho2 = (1 - prob) * rho2 + prob * Z @ rho2 @ Z
    if measure:
        return state_fidelity(np.diag(rho1.diagonal()), np.diag(rho2.diagonal()))
    else:
        return state_fidelity(rho1, rho2)

#CLIFFORD

def normalize_angle(angle):
    normalized_angle = angle % (2 * np.pi)
    if normalized_angle < 0:
        normalized_angle += 2 * np.pi   
    return normalized_angle
    
def round_to_pi2(angle):
    normalized_angle = normalize_angle(angle)
    rounded_angle = (normalized_angle // (np.pi / 2)) * (np.pi / 2)
    return rounded_angle

def round_angles(circuit):
    new_circuit = QuantumCircuit(*circuit.qregs, *circuit.cregs)
    angle = 0
    for instr, qargs, cargs in circuit.data:
        if (instr.name == 'rz'):
            angle = round_to_pi2(instr.params[0])
            new_circuit.rz(angle, qargs) if angle else 0
        elif(instr.name == 'rx'):
            angle = round_to_pi2(instr.params[0])
            new_circuit.rx(angle, qargs) if angle else 0
        elif(instr.name == 'ry'):
            angle = round_to_pi2(instr.params[0])
            new_circuit.ry(angle, qargs) if angle else 0
        else:
            new_circuit.append(instr, qargs, cargs)
    return new_circuit



def make_clifford_data(circ : QuantumCircuit, data : list = None, round_angle : bool = False):
    if not round_angle or not data:
        c, data = make_native_circuit(round_angles(circ), opt_level=0)
    else:
        c = circ
    new_circuit = QuantumCircuit(*c.qregs, *c.cregs)
    new_data = []
    for i, (instr, qargs, cargs) in enumerate(c.data):
        if (instr.name == 'rz'):
            angle = instr.params[0]
            if (angle == np.pi / 2):
                new_circuit.s(qargs)
                new_data.append(data[i])
            elif (angle == np.pi):
                new_circuit.z(qargs)
                new_data.append(data[i])
            elif (angle == 3 * np.pi / 2):
                new_circuit.s(qargs)
                new_circuit.z(qargs)
                new_data += [0, data[i]]
        elif (instr.name == 'rx'):
            angle = instr.params[0]
            if (angle == np.pi / 2):
                new_circuit.h(qargs)
                new_circuit.s(qargs)
                new_circuit.h(qargs)
                new_data += [0, 0, data[i]]
            elif (angle == np.pi):
                new_circuit.x(qargs)
                new_data.append(data[i])
            elif (angle == 3 * np.pi / 2):
                new_circuit.h(qargs)
                new_circuit.s(qargs)
                new_circuit.h(qargs)
                new_circuit.x(qargs)
                new_data += [0, 0, 0, data[i]]
        else:
            new_circuit.append(instr, qargs, cargs)
            new_data.append(data[i])
    return new_circuit, new_data


def noisy_circuit(circuit, data):
    new_circuit = QuantumCircuit(*circuit.qregs, *circuit.cregs)
    assert(len(circuit.data) == len(data))
    for i, (instr, qargs, cargs) in enumerate(circuit.data):
        prob_error = data[i]
        choice = np.random.choice(['Z', 'I'], p=[prob_error, 1 - prob_error])
        new_circuit.append(instr, qargs, cargs)
        if (choice == 'Z'):
            new_circuit.z(qargs)
    return new_circuit


#to check clifford
def fidelity_density_matrix_clifford(circuit :QuantumCircuit, T2=2000, T_RX=100, T_RZ=30, T_CZ=15, measure=True):
    circ, d = make_clifford_data(circuit)
    noiseless_circ = circ.copy()
    noiseless_circ.save_density_matrix()
    g = AerSimulator(method='density_matrix')
    rho1 = np.array(g.run(noiseless_circ, shots=1, noise_model=None, optimization_level=0).result().data()['density_matrix'])
    rho2 = np.zeros_like(rho1)
    rho2[0, 0] = 1
    for i, (instr, qargs, cargs) in enumerate(circ.data):
        qc = QuantumCircuit(*circ.qregs, *circ.cregs)
        qc.append(instr, qargs, cargs)
        U = Operator(qc).to_matrix()
        qc = QuantumCircuit(*circ.qregs, *circ.cregs)
        qc.z(qargs)
        Z = Operator(qc).to_matrix()
        rho2 = U @ rho2 @ U.T.conj()
        prob = d[i]
        #rho2 = (1 - prob) * rho2 + prob * np.eye(2 ** circ.num_qubits) / (2 ** circ.num_qubits)
        rho2 = (1 - prob) * rho2 + prob * Z @ rho2 @ Z
    if measure:
        return state_fidelity(np.diag(np.abs(rho1.diagonal())), np.diag(np.abs(rho2.diagonal())))
    else:
        return state_fidelity(rho1, rho2)

def fidelity_clifford_measuring(circuit : QuantumCircuit, shots=1000, use_tqdm=True):
    c, d = make_clifford_data(round_angles(circuit))
    noiseless = c.copy()
    noiseless.measure_all()
    g = AerSimulator(method='stabilizer')
    result1 = dict(g.run(noiseless, noise_model=None, shots=shots, optimization_level=0).result().data())['counts']
    noisies = []
    result2 = defaultdict(int)
    iterator = tqdm(range(shots)) if use_tqdm else range(shots)
    for _ in iterator:
        noisy = noisy_circuit(c, d)
        noisy.measure_all()
        noisies.append(noisy)
    results = g.run(noisies, noise_model=None, shots=1, optimization_level=0).result()
    for i in range(shots):
        res = list(results.data(i)['counts'])[0]
        result2[res] += results.data(i)['counts'][res]
    df = pd.DataFrame([result1, result2]).T
    df = df.fillna(0).astype(int)
    state1 = np.array(np.sqrt(df[0] / shots))
    state2 = np.array(np.sqrt(df[1] / shots))
    return state_fidelity(state1, state2)

def log_prob(circuit : QuantumCircuit):
    _, d = make_native_circuit(circuit, opt_level=0)
    res = 0
    for error_prob in d:
        if not error_prob is None:
            res += np.log(1 - error_prob)
    return res