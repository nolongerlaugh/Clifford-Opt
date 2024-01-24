from .transformer import X_CZ_to_CZ_X_Z, CZ_X_to_X_Z_CZ, RZ_CZ_to_CZ_RZ, CZ_RZ_to_RZ_CZ, merge_cz, merge_unitary
from .fidelity import fidelity_clifford_measuring, log_prob

from numpy import random
from qiskit import QuantumCircuit

def transpiler(qc : QuantumCircuit, shots = 1000, n_iter=1000, inplace=False, use_tqdm=False, func = fidelity_clifford_measuring) -> QuantumCircuit:
    circuit = qc
    if not inplace:
        circuit = qc.copy()
    fidelity = func(circuit, shots, use_tqdm=use_tqdm)
    print(fidelity, log_prob(circuit))
    circuit = merge_cz(circuit)
    circuit = merge_unitary(circuit)
    fidelity = func(circuit, shots, use_tqdm=use_tqdm)
    print(fidelity, log_prob(circuit))
    transformers = [CZ_X_to_X_Z_CZ(), CZ_RZ_to_RZ_CZ(), X_CZ_to_CZ_X_Z(), RZ_CZ_to_CZ_RZ()]
    for _ in range(n_iter):
        tr = transformers.copy()
        random_transformer = random.choice(tr)
        indexes = None
        while len(tr) > 0:
            random_transformer = random.choice(tr)
            res = random_transformer.find(circuit)
            if len(res) == 0:
                i = tr.index(random_transformer)
                tr.pop(i)
            else:
                indexes = res[random.choice(range(len(res)))]
                break
        if len(tr) == 0:
            return circuit
        circuit2 = random_transformer.transform(circuit, indexes)
        circuit2 = merge_cz(circuit2)
        circuit2 = merge_unitary(circuit2)
        fidelity2 = func(circuit, shots, use_tqdm=use_tqdm)
        print(fidelity, log_prob(circuit2))
        if fidelity2 > fidelity:
            fidelity = fidelity2
            circuit = circuit2
    return circuit