from .transformer import RX_CZ_to_RX_CZ_X_Z, CZ_RX_to_X_Z_CZ_RX, RZ_CZ_to_CZ_RZ, CZ_RZ_to_RZ_CZ, RX_RZ_to_RX_RZ_X, RZ_RX_to_RX_RZ_X, merge_cz, merge_unitary
from .fidelity import fidelity_clifford_measuring, log_prob

from numpy import random, abs
from tqdm import tqdm
from qiskit import QuantumCircuit

def transpiler(qc : QuantumCircuit, shots = 1000, n_iter=1000, inplace=False, use_tqdm=False, func = fidelity_clifford_measuring, output=True, same=1000) -> QuantumCircuit:
    circuit = qc
    if not inplace:
        circuit = qc.copy()
    fidelity = func(circuit, shots, use_tqdm=use_tqdm)
    if output:
        print(fidelity, log_prob(circuit))
    circuit = merge_cz(circuit)
    circuit = merge_unitary(circuit)
    circuit = merge_cz(circuit)
    fidelity = func(circuit, shots, use_tqdm=use_tqdm)
    if output:
        print(fidelity, log_prob(circuit))
    fidelity_same = 0
    transformers = [CZ_RX_to_X_Z_CZ_RX(), CZ_RZ_to_RZ_CZ(), RX_CZ_to_RX_CZ_X_Z(), RZ_CZ_to_CZ_RZ(), RX_RZ_to_RX_RZ_X(), RZ_RX_to_RX_RZ_X()]
    iterator = tqdm(range(n_iter)) if use_tqdm else range(n_iter)
    for _ in iterator:
        fidelity_last = fidelity
        tr = transformers.copy()
        random_transformer = random.choice(tr)
        indexes = None
        i = -1
        res = []
        fidelity2 = 0
        while len(tr) > 0:
            random_transformer = random.choice(tr)
            res = random_transformer.find(circuit)
            i = tr.index(random_transformer)
            if len(res) == 0:
                tr.pop(i)
            else:
                while len(res) > 0:
                    j = random.choice(range(len(res)))
                    indexes = res[j]
                    circuit2 = random_transformer.transform(circuit, indexes)
                    circuit2 = merge_cz(circuit2)
                    circuit2 = merge_unitary(circuit2)
                    circuit2 = merge_cz(circuit2)
                    fidelity2 = func(circuit, shots, use_tqdm=use_tqdm)
                    if output:
                        print(_, fidelity2, fidelity, log_prob(circuit2))
                    if fidelity2 > fidelity + 1e-6:
                        print("popa")
                        fidelity = fidelity2
                        circuit = circuit2
                        res = []
                    else:
                        res.pop(j)
                break
        if len(tr) == 0:
            return circuit
        if abs(fidelity - fidelity_last) <= 1e-6:
            fidelity_same += 1
        else:
            fidelity_same = 0
        if fidelity_same == same:
            return circuit
    return circuit