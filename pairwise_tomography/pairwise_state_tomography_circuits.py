"""
Pairwise tomography circuit generation
"""
# pylint: disable=invalid-name
import numpy as np

from qiskit import ClassicalRegister
from qiskit.ignis.verification.tomography.basis.circuits import _format_registers

def pairwise_state_tomography_circuits(circuit, measured_qubits):
    """
    Generates a minimal set of circuits for pairwise state tomography.

    This performs measurement in the Pauli-basis resulting in
    circuits for an n-qubit state tomography experiment.

    Args:
        circuit (QuantumCircuit): the state preparation circuit to be
                                  tomographed.
        measured_qubits (QuantumRegister): the qubits to be measured.
            This can also be a list of whole QuantumRegisters or
            individual QuantumRegister qubit tuples.

    Returns:
        A list of QuantumCircuit objects containing the original circuit
        with state tomography measurements appended at the end.
    """

    ### Initialisation stuff
    if isinstance(measured_qubits, list):
        # Unroll list of registers
        meas_qubits = _format_registers(*measured_qubits)
    else:
        meas_qubits = _format_registers(measured_qubits)

    N = len(meas_qubits)

    cr = ClassicalRegister(len(meas_qubits))

    ### Uniform measurement settings
    X = circuit.copy(name=str(('X',)*N))
    Y = circuit.copy(name=str(('Y',)*N))
    Z = circuit.copy(name=str(('Z',)*N))

    X.add_register(cr)
    Y.add_register(cr)
    Z.add_register(cr)

    for bit_index, qubit in enumerate(meas_qubits):
        X.h(qubit)
        Y.sdg(qubit)
        Y.h(qubit)

        X.measure(qubit, cr[bit_index])
        Y.measure(qubit, cr[bit_index])
        Z.measure(qubit, cr[bit_index])

    output_circuit_list = [X, Y, Z]

    ### Heterogeneous measurement settings
    # Generation of six possible sequences
    sequences = []
    meas_bases = ['X', 'Y', 'Z']
    for i in range(3):
        for j in range(2):
            meas_bases_copy = meas_bases[:]
            sequence = [meas_bases_copy[i]]
            meas_bases_copy.remove(meas_bases_copy[i])
            sequence.append(meas_bases_copy[j])
            meas_bases_copy.remove(meas_bases_copy[j])
            sequence.append(meas_bases_copy[0])
            sequences.append(sequence)

    # Qubit colouring
    nlayers = int(np.ceil(np.log(float(N))/np.log(3.0)))

    for layout in range(nlayers):
        for sequence in sequences:
            meas_layout = circuit.copy()
            meas_layout.add_register(cr)
            meas_layout.name = ()
            for bit_index, qubit in enumerate(meas_qubits):
                local_basis = sequence[int(float(bit_index)/float(3**layout))%3]
                if local_basis == 'Y':
                    meas_layout.sdg(qubit)
                if local_basis != 'Z':
                    meas_layout.h(qubit)
                meas_layout.measure(qubit, cr[bit_index])
                meas_layout.name += (local_basis,)
            meas_layout.name = str(meas_layout.name)
            output_circuit_list.append(meas_layout)

    return output_circuit_list

def pairwise_mitigation_circuits(circuit, meas_qubits=None):
    
    # if no `meas_qubits` is supplied, mitigation will be
    # for all qubits in the first (or only) qubit register
    if not meas_qubits:
        meas_qubits = circuit.qregs[0]
    cr = ClassicalRegister(len(meas_qubits))
    
    # create a blank version of the circuit
    blank_circuit = circuit.copy()
    blank_circuit.data = []

    # determine all circuit names
    
    N = len(meas_qubits)
    n = int(np.ceil(np.log(N)/np.log(2)))+1
    
    strings = []
    for j in range(N):
        strings.append( bin(j)[2::].zfill(n))

    circuit_names = []
    for k in range(n):
        circuit_0 = []
        circuit_1 = []
        for j in range(N):
            circuit_0.append(strings[j][k])
            circuit_1.append(str((int(strings[j][k])+1)%2))
        for circuit in [circuit_0,circuit_1]:
            circuit_names.append(tuple(circuit))

    # create the circuits
            
    output_circuit_list = []
    
    for circuit_name in circuit_names:
        meas_layout = blank_circuit.copy(name=str(circuit_name))
        meas_layout.add_register(cr)
        
        for bit_index, qubit in enumerate(meas_qubits):
            if circuit_name[bit_index]=='1':
                meas_layout.x(qubit)
            meas_layout.measure(qubit,cr[bit_index])
        
        output_circuit_list.append( meas_layout )
        
    return output_circuit_list
