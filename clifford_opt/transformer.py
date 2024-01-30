from abc import ABC, abstractmethod
from .fidelity import normalize_angle
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Operator
from qiskit.circuit.library import RZGate
from tqdm import tqdm
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

def merge_unitary(qc : QuantumCircuit, inplace=False, u=False)->QuantumCircuit:
    """
    ─R_─R_─R_─R_─@─R_─    ≡     ─RZ─RX─@─RZ─RX─
    			 |					   |
    ─────────────@───     ≡     ───────@──────
    """
    circuit = qc
    if not inplace:
        circuit = qc.copy()
    new_circuit = QuantumCircuit(*circuit.qregs, *circuit.cregs)
    buf = {q: np.eye(2) for q in list(circuit.qubits)}
    it = tqdm(circuit.data) if u else circuit.data
    for instr, qargs, cargs in it:
        if instr.name in ['rx', 'rz']:
            buf[qargs[0]] = Operator(instr).to_matrix() @ buf[qargs[0]]
        else:
            u = {qarg: [] for qarg in qargs}
            for qarg in qargs:
                c = QuantumCircuit(1)
                c.unitary(buf[qarg], 0)
                c = transpile(c, basis_gates=['rz', 'rx'], optimization_level=0)
                for instr1, _, _ in c:
                    u[qarg].append((instr1.name, instr1.params[0]))
                buf[qarg] = np.eye(2)
            if instr.name == 'cz':
                for qarg in qargs:
                    if len(u[qarg]) > 0 and u[qarg][-1][0] == 'rz':
                        buf[qarg] = RZGate(u[qarg][-1][1]).to_matrix()
                        u[qarg].pop()
            for qarg in qargs:
                for op in u[qarg]:
                    if op[0] == 'rx':
                        new_circuit.rx(normalize_angle(op[1]), qarg)
                    else:
                        new_circuit.rz(normalize_angle(op[1]), qarg)
            new_circuit.append(instr, qargs, cargs)
    for qarg in list(circuit.qubits):
        c = QuantumCircuit(1)
        c.unitary(buf[qarg], 0)
        c = transpile(c, basis_gates=['rz', 'rx'], optimization_level=0)
        for instr1, _, _ in c:
            if instr1.name == 'rx':
                new_circuit.rx(normalize_angle(instr1.params[0]), qarg)
            else:
                new_circuit.rz(normalize_angle(instr1.params[0]), qarg)
        buf[qarg] = np.eye(2) 
    return new_circuit

class Transformer(ABC):
    @abstractmethod
    def find(self, qc : QuantumCircuit) -> list:
        pass

    @abstractmethod
    def transform(self, qc : QuantumCircuit, indexes : tuple, inplace=False) ->QuantumCircuit:
        pass

class RX_CZ_to_RX_CZ_X_Z(Transformer):
    """
    ───RX(a + pi)───@───      ─RX(a)──@──X────
                    |     ≡           |       
    ────────────────@───      ────────@────Z──
    """
    def find(self, qc : QuantumCircuit) -> list:
        circuit = qc
        result = []
        for i, instr1 in enumerate(circuit):
            indexes = None
            if (instr1.operation.name == 'rx'):
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
        a = normalize_angle(circuit[index_x].operation.params[0] - np.pi)
        circuit.rx(a, q_x)
        circuit.rx(np.pi, q_x)
        circuit.rz(np.pi, q_z)
        circuit.data.insert(index_cz + 1, circuit[-2])
        circuit.data.insert(index_cz + 1, circuit[-1])
        circuit.data[index_x] = circuit.data[-3]
        circuit.data = circuit.data[:-3]
        return circuit

class CZ_RX_to_X_Z_CZ_RX(RX_CZ_to_RX_CZ_X_Z):
    """
    ────@────RX(a + pi)─      ───X───@────RX(a)───
        |                 ≡          |       
    ────@──────────────       ─────Z─@────────────
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
        circuit.rz(circuit[index_z].operation.params[0], q_z)
        circuit.data.insert(index_cz + 1, circuit[-1])
        circuit.data.pop(index_z)
        circuit.data = circuit.data[:-1]
        return circuit

class CZ_RZ_to_RZ_CZ(RZ_CZ_to_CZ_RZ):
    """
    ────@─────RZ─      ───RZ───@───────
        |         ≡            |       
    ────@───────      ─────────@──────
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

class RX_RZ_to_RX_RZ_X(Transformer):
    """
    ────RX(pi + b)─────RZ(a)─   ≡   ─RX(b)──RZ(-a)───X───
    """
    def find(self, qc : QuantumCircuit) -> list:
        circuit = qc
        result = []
        for i, instr1 in enumerate(circuit):
            indexes = None
            if (instr1.operation.name == 'rx'):
                q_x = instr1.qubits[0]
                for j, instr2 in enumerate(circuit[i + 1:]):
                    if q_x in list(instr2.qubits):
                        if (instr2.operation.name == 'rz'):
                            indexes = i, i + j + 1
                        break
            if indexes is not None:
                result.append(indexes)
        return result

    def transform(self, qc : QuantumCircuit, indexes : tuple, inplace=False) -> QuantumCircuit:
        circuit = qc
        if not inplace:
            circuit = qc.copy()
        index_x, index_rz = indexes
        assert(circuit[index_x].operation.name == 'rx')
        assert(circuit[index_rz].operation.name == 'rz')
        q_x = circuit[index_x].qubits[0]
        qs = circuit[index_rz].qubits
        assert (q_x in list(qs))
        a = normalize_angle(circuit[index_x].operation.params[0] - np.pi)
        circuit.rx(a, q_x)
        circuit.rz(normalize_angle(-circuit[index_rz].operation.params[0]), q_x)
        circuit.rx(np.pi, q_x)
        circuit.data.insert(index_rz + 1, circuit[-1])
        circuit.data.insert(index_rz + 1, circuit[-2])
        circuit.data.insert(index_rz + 1, circuit[-3])
        circuit.data.pop(index_rz)
        circuit.data.pop(index_x)
        circuit.data = circuit.data[:-3]
        return circuit

class RZ_RX_to_RX_RZ_X(RX_RZ_to_RX_RZ_X):
    """
    ───RZ(a)───RX(pi + b)─── ≡ ────X─────RZ(-a)─────RX(b)────        
    """
    def find(self, qc : QuantumCircuit) -> list:
        circuit = qc.copy()
        circuit.data = circuit.data[::-1]
        result = super().find(circuit)
        d = len(qc)
        for i in range(len(result)):
            index_x, index_rz = result[i]
            result[i] = d - 1 - index_x, d - 1 - index_rz
        return result

    def transform(self, qc : QuantumCircuit, indexes : tuple, inplace=False) ->QuantumCircuit:
        circuit = qc
        if not inplace:
            circuit = qc.copy()
        index_x, index_rz = indexes
        circuit2 = qc.copy()
        circuit2.data = circuit2.data[::-1]
        d = len(circuit2)
        circuit2 = super().transform(circuit2, (d - 1 - index_x, d - 1 - index_rz), inplace=True)
        circuit.data = circuit2.data[::-1]
        return circuit