from abc import ABC, abstractmethod
from .fidelity import normalize_angle
from qiskit import QuantumCircuit

import numpy as np

def merge_cz(qc : QuantumCircuit, inplace=False)->QuantumCircuit:
    """
    ───@──@──   ───
       |  |   ≡               
    ───@──@──   ───
    """
    circuit = qc
    if not inplace:
        circuit = qc.copy()
    for i, instr1 in enumerate(circuit):
        remove_index = None
        if (instr1.operation.name == 'cz'):
            q1, q2 = instr1.qubits
            for j, instr2 in enumerate(circuit[i + 1:]):
                if (instr2.operation.name == 'cz'):
                    g1, g2 = instr2.qubits
                    if (q1, q2) == (g1, g2) or (q2, q1) == (g1, g2): 
                        remove_index = i + j + 1
                        break
                if q1 in list(instr2.qubits) or q2 in list(instr2.qubits):
                    break
        if remove_index is not None:
            circuit.data.pop(i)
            circuit.data.pop(i + j)
    return circuit

def merge_unitary(qc : QuantumCircuit, inplace=False)->QuantumCircuit:
    """
    ───RX(a)──RX(b)── ≡ ─RX(a + b)──
    
    ───RX(2pi)──      ≡ ────────────
    """
    circuit = qc
    if not inplace:
        circuit = qc.copy()
    ops = {q: [] for q in list(circuit.qubits)}
    for i, instr in enumerate(circuit):
        for q in list(instr.qubits):
            ops[q].append(i)
    drops = []
    swaps = [] #swap
    for q in ops:
        buf_i = None
        buf_name = None
        buf_angle = 0
        for k, i in enumerate(ops[q]):
            op = circuit[i]
            if ((not buf_i is None) and buf_name == op.operation.name):
                drops.append(i)
                buf_angle += op.operation.params[0]
            else:
                if not (buf_i is None):
                    buf_angle = normalize_angle(buf_angle)
                    if buf_name == 'rx':
                        circuit.rx(buf_angle, q)
                    else:
                        circuit.rz(buf_angle, q)
                    swaps.append((buf_i, len(circuit) - 1))
                    if (np.abs(buf_angle) < 1e-6 or np.abs(buf_angle - 2 * np.pi) < 1e-6):
                        drops.append(buf_i)
                buf_i, buf_angle, buf_name = (i, op.operation.params[0], op.operation.name) if (
                    op.operation.name in ['rx', 'rz']) else (None, 0, None)
            if (k == len(ops[q]) - 1):
                if not (buf_i is None):
                    buf_angle = normalize_angle(buf_angle)
                    if buf_name == 'rx':
                        circuit.rx(buf_angle, q)
                    else:
                        circuit.rz(buf_angle, q)
                    swaps.append((buf_i, len(circuit) - 1))
                    if (np.abs(buf_angle) < 1e-6 or np.abs(buf_angle - 2 * np.pi) < 1e-6):
                        drops.append(buf_i)
    for a, b in swaps:
        circuit.data[a], circuit.data[b] = circuit.data[b], circuit.data[a]
        drops.append(b)
    drops = sorted(drops)[::-1]
    for d in drops:
        circuit.data.pop(d)
    return circuit

class Transformer(ABC):
    @abstractmethod
    def find(self, qc : QuantumCircuit) -> list:
        pass

    @abstractmethod
    def transform(self, qc : QuantumCircuit, indexes : tuple, inplace=False) ->QuantumCircuit:
        pass

class X_CZ_to_CZ_X_Z(Transformer):
    """
    ───X───@───      ───@──X────
           |     ≡      |       
    ───────@───      ───@────Z──
    """
    def find(self, qc : QuantumCircuit) -> list:
        circuit = qc
        result = []
        for i, instr1 in enumerate(circuit):
            indexes = None
            if (instr1.operation.name == 'rx' and np.abs(instr1.operation.params[0] - np.pi) < 1e-6):
                q_x = instr1.qubits[0]
                for j, instr2 in enumerate(circuit[i + 1:]):
                    if q_x in list(instr2.qubits):
                        if (instr2.operation.name == 'cz'):
                            indexes = i, i + j + 1
                        break
            if indexes is not None:
                result.append(indexes)
        return result

    def transform(self, qc : QuantumCircuit, indexes : tuple, inplace=False) -> QuantumCircuit:
        circuit = qc
        if not inplace:
            circuit = qc.copy()
        index_x, index_cz = indexes
        assert(circuit[index_x].operation.name == 'rx')
        assert(circuit[index_cz].operation.name == 'cz')
        q_x = circuit[index_x].qubits[0]
        qs = circuit[index_cz].qubits
        assert (q_x in list(qs))
        q_z = qs[1] if qs[0] == q_x else qs[0]
        circuit.rx(np.pi, q_x)
        circuit.rz(np.pi, q_z)
        circuit.data.insert(index_cz + 1, circuit[-2])
        circuit.data.insert(index_cz + 1, circuit[-1])
        circuit.data.pop(index_x)
        circuit.data = circuit.data[:-2]
        return circuit

class CZ_X_to_X_Z_CZ(X_CZ_to_CZ_X_Z):
    """
    ────@─────X─      ───X───@───────
        |         ≡          |       
    ────@───────      ─────Z─@──────
    """
    def find_cz_x(self, qc : QuantumCircuit) -> list:
        circuit = qc.copy()
        circuit.data = circuit.data[::-1]
        result = super().find(circuit)
        d = len(qc)
        for i in range(len(result)):
            index_x, index_cz = result[i]
            result[i] = d - 1 - index_x, d - 1 - index_cz
        return result

    def change_cz_x_to_x_z_cz(self, qc : QuantumCircuit, indexes : tuple, inplace=False) ->QuantumCircuit:
        circuit = qc
        if not inplace:
            circuit = qc.copy()
        index_x, index_cz = indexes
        circuit2 = qc.copy()
        circuit2.data = circuit2.data[::-1]
        d = len(circuit2)
        circuit2 = super().transform(circuit2, (d - 1 - index_x, d - 1 - index_cz), inplace=True)
        circuit.data = circuit2.data[::-1]
        return circuit
    
class RZ_CZ_to_CZ_RZ(Transformer):
    """
    ───RZ───@───      ───@───RZ───
            |     ≡      |       
    ────────@───      ───@──────
    """
    def find(self, qc : QuantumCircuit) -> list:
        circuit = qc
        result = []
        for i, instr1 in enumerate(circuit):
            indexes = None
            if (instr1.operation.name == 'rz'):
                q_z = instr1.qubits[0]
                for j, instr2 in enumerate(circuit[i + 1:]):
                    if q_z in list(instr2.qubits):
                        if (instr2.operation.name == 'cz'):
                            indexes = i, i + j + 1
                        break
            if indexes is not None:
                result.append(indexes)
        return result

    def transform(self, qc : QuantumCircuit, indexes : tuple, inplace=False) -> QuantumCircuit:
        circuit = qc
        if not inplace:
            circuit = qc.copy()
        index_z, index_cz = indexes
        assert(circuit[index_z].operation.name == 'rz')
        assert(circuit[index_cz].operation.name == 'cz')
        q_z = circuit[index_z].qubits[0]
        qs = circuit[index_cz].qubits
        assert (q_z in list(qs))
        circuit.rz(np.pi, q_z)
        circuit.data.insert(index_cz + 1, circuit[-1])
        circuit.data.pop(index_z)
        circuit.data = circuit.data[:-1]
        return circuit

class CZ_RZ_to_RZ_CZ(RZ_CZ_to_CZ_RZ):
    """
    ────@─────Z─      ───Z───@───────
        |         ≡          |       
    ────@───────      ───────@──────
    """
    def find(self, qc : QuantumCircuit) -> list:
        circuit = qc.copy()
        circuit.data = circuit.data[::-1]
        result = super().find(circuit)
        d = len(qc)
        for i in range(len(result)):
            index_z, index_cz = result[i]
            result[i] = d - 1 - index_z, d - 1 - index_cz
        return result

    def transform(self, qc : QuantumCircuit, indexes : tuple, inplace=False) ->QuantumCircuit:
        circuit = qc
        if not inplace:
            circuit = qc.copy()
        index_z, index_cz = indexes
        circuit2 = qc.copy()
        circuit2.data = circuit2.data[::-1]
        d = len(circuit2)
        circuit2 = super().transform(circuit2, (d - 1 - index_z, d - 1 - index_cz), inplace=True)
        circuit.data = circuit2.data[::-1]
        return circuit
    
"""
────X─────RZ(a)─   ≡   ───RZ(-a)───X───

────RX(a + pi)──── ≡ ────RX(a)────RX(pi)

──── U ────────    ≡ ────RZ────RX────RZ────

────X──Z──         ≡ ────Z──X──

"""
