from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Operator
from qiskit.providers.aer import AerSimulator
from qiskit_aer import aerbackend
from qiskit.providers.aer.noise import pauli_error
from qiskit.quantum_info import state_fidelity

import numpy as np
import pandas as pd

np.random.seed(239)
native_gates = ['rz', 'cz', 'rx']
noise_model = aerbackend.NoiseModel()
p_gate1 = 0.05
error_gate1 = pauli_error([('Z',p_gate1), ('I', 1 - p_gate1)])
error_gate2 = error_gate1.tensor(error_gate1)
noise_model.add_all_qubit_quantum_error(error_gate1, ["rx", "rz"])
noise_model.add_all_qubit_quantum_error(error_gate2, ["cz"])

#to check (measuring -> measure = True)
def Qfidelity_density_matrix_orig(circuit :QuantumCircuit, measure=False, shots=None, use_tqdm=None):
    circ = circuit.copy()
    circ.save_density_matrix()
    g = AerSimulator(method='density_matrix')

    rho1 = np.array(g.run(circ, shots=1, noise_model=None, optimization_level=0).result().data()['density_matrix'])
    rho2 = np.array(g.run(circ, shots=1, noise_model=noise_model, optimization_level=0).result().data()['density_matrix'])
    if measure:
        return state_fidelity(np.diag(rho1.diagonal()), np.diag(rho2.diagonal()))
    else:
        return state_fidelity(rho1, rho2)

#CLIFFORD

from .fidelity import round_angles

#to check clifford
def Qfidelity_density_matrix_clifford(circuit :QuantumCircuit, measure=False, shots=None, use_tqdm=None):
    circ = round_angles(circuit)
    circ.save_density_matrix()
    g = AerSimulator(method='density_matrix')

    rho1 = np.array(g.run(circ, shots=1, noise_model=None, optimization_level=0).result().data()['density_matrix'])
    rho2 = np.array(g.run(circ, shots=1, noise_model=noise_model, optimization_level=0).result().data()['density_matrix'])
    if measure:
        return state_fidelity(np.diag(rho1.diagonal()), np.diag(rho2.diagonal()))
    else:
        return state_fidelity(rho1, rho2)


def Qfidelity_clifford_measuring(circuit : QuantumCircuit, shots=1000, use_tqdm=True):
    c = round_angles(circuit)
    noiseless = c.copy()
    noiseless.measure_all()
    g = AerSimulator(method='stabilizer')
    result1 = dict(g.run(noiseless, noise_model=None, shots=shots, optimization_level=0).result().data())['counts']
    result2 = dict(g.run(noiseless, noise_model=noise_model, shots=shots, optimization_level=0).result().data())['counts']
    df = pd.DataFrame([result1, result2]).T
    df = df.fillna(0).astype(int)
    state1 = np.array(np.sqrt(df[0] / shots))
    state2 = np.array(np.sqrt(df[1] / shots))
    return state_fidelity(state1, state2)

def log_prob(circuit : QuantumCircuit):
    res = 0
    for instr, _, _ in circuit:
        if instr.num_qubits == 1:
            res += np.log(1 - p_gate1)
        else:
            res += np.log(1 - p_gate1**2 - 2* p_gate1*(1-p_gate1))
    return res