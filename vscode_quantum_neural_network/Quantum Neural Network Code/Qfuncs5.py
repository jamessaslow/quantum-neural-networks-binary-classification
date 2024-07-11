'''
Qfuncs5.py
James Saslow
3/9/2024

Requires Qiskit version 1.0.2 or later

A Complimentary custom package for Qiskit that improves user access to statevector retrieval,
readability, and measurement. 

Public use functions in this code are:

-  QiskitVersion()
        - Returns version of Qiskit installed on local device
-  ReturnPsi(qc, **kwargs)
        - Returns psi in big endian
-  Measure(qc, c,shots, **kwargs)
        - Measures specified qubits and retuns statistics
-  ProbPlot(qc, q, shots, **kwargs)
        - Plots histogram of measurement statsitics

'''


#===================================================================================================

# Importing Packages
import numpy as np
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit.quantum_info import Statevector
import qiskit
from qiskit.primitives import BackendSampler
from qiskit.providers import BackendV1
from qiskit.primitives import Sampler

#===================================================================================================

# Defining Intermediate / Private use Functions

def _get_statevec(qc):
    '''
    - Returns numpy statevector in little endian
    - Not intended for public use
    '''

    statevec = np.array(Statevector.from_instruction(qc))
    return statevec

def _dual_sort(key, values):
    '''
    Sorts an array 'values' and simultaneously orders a list 'key'
    '''
    k=1
    while k!=0:
        k=0
        for i in range(len(key)-1):
            a = key[i]
            b = key[i+1]

            c = values[i]
            d = values[i+1]

            if b<a:
                k+=1
                key[i] = b
                key[i+1] = a

                values[i] = d
                values[i+1] = c

    return values

def _bin_gen(number, num_qubits):
    '''
    Generates a string binary number given a base 10 'number' 
    and the number of qubits 'num_qubits'
    '''
    bin1 = bin(number)[2:]
    L = num_qubits - len(bin1)
    bin2 = L*'0' + bin1
    return bin2
    
def _rev_bin_gen(number, num_qubits):
    '''
    Reverses the ordering of generated binary number given base 10 'number'
    and the number of qubits 'num_qubits'
    '''
    return _bin_gen(number, num_qubits)[::-1]
    
def _binto10(number):
    '''
    Converts a binary string to a base 10 number
    '''
    num_qubits = len(number)
    num_bases  = int(2**num_qubits)
    reg_order = []
    for i in range(num_bases):
        reg_order.append(_bin_gen(i, num_qubits))
            
    reg_order = np.array(reg_order)
    return np.where(reg_order == number)[0][0]

#===================================================================================================

# Defining Public Use Functions


def QiskitVersion():
    '''
    Prints current version of Qiskit
    '''
    print(qiskit.__version__)

def ReturnPsi(qc, **kwargs):
    '''
    Returns the wavefunction psi in a numpy array in little endian
    
    **kwargs:
    - (braket = True) => Displays states in braket notation
    - (zeros = True)  => Displays zero amplitude states
    - (polar = True)  => Displays states in polar form
    
    '''

    statevec = _get_statevec(qc)
    dec = 5
    if 'precision' in kwargs:
        dec = int( kwargs['precision'] )
           
    # Rounding off states
    statevec = np.round(statevec,dec) 
    num_bases = len(statevec)
    num_qubits = int(np.log2(num_bases))
    
    key = []
    for i in range(num_bases):
        key.append(_binto10(_rev_bin_gen(i, num_qubits)))
        
    psi = _dual_sort(key, statevec) # Ordering the Wavefunction in Little Endian
    
    braket_bool = kwargs.get('braket', False)
    zeros_bool  = kwargs.get('zeros', False)
    if braket_bool == True:
        
        # Polar Form
        polar_bool = kwargs.get('polar', False)
        printed_psi = []
        if polar_bool == True:
            # Storing wavefunction in polar format
            for i in range(num_bases):
                r = np.round(np.abs(psi[i]),dec )
                theta = np.round(np.angle(psi[i]),dec)
                if r == 0:
                    printed_psi.append(0)
                else:
                    printed_psi.append(str(r) + ' exp(1j*' + str(theta) + ')')
            psi = printed_psi
            
        
        # Generating bases labels i.e. |x>
        bases = []
        for i in range(num_bases):
            bases.append('|'+_bin_gen(i, num_qubits) + '>')
  
        # Determining number of spaces to have alligned formatting
        char_arr = []
        for i in range(num_bases):
            char_arr.append(len(str(psi[i])))
        max_char = max(char_arr)
        spaces = max_char - np.array(char_arr)


        for i in range(num_bases):
            
            # Not Displaying Zero Amplitude States Unless zero_bool = True
            if type(psi[i]) != str:       
                if (zeros_bool == False) and (np.abs(psi[i]) == 0):
                    continue
            print(psi[i],(spaces[i])*' ' ,bases[i])
    else:
        return psi
    




def Measure(qc, c,shots, **kwargs):
    '''
    Retuns probability amplitudes and their assosciated basis states
    
    '''
    # ========================================================================
    # Taking the Measurement
    sampler = Sampler()
    result = sampler.run([qc], shots=shots)
    counts_rev =  (result.result().quasi_dists)[0] # Counts in reverse binary order
    keys = np.array(list(counts_rev.keys()))
    values = np.array(list(counts_rev.values()))

    # ========================================================================
    # Filling in 0 probabilities

    # num_qubits = len(q)
    # num_bases  = int(2**num_qubits)

    num_qubits = len(c)
    num_bases  = int(2**num_qubits)

    new_keys   = []
    new_values = []

    for i in range(num_bases):
        if i in keys:
            index = np.where(keys == i)[0][0]
            x = values[index]
            new_keys.append(i)
            new_values.append(x)
        else:
            new_keys.append(i)
            new_values.append(0)

    # ========================================================================
    # Reformating in Big Endian

    keys_rev = []
    for i in range(num_bases):
        num1 = _rev_bin_gen(i, num_qubits)
        num2 = _binto10(num1)
        keys_rev.append(num2)

    # List of Probabilities in Big Endian
    ans = _dual_sort(keys_rev,new_values)

    # ========================================================================
    # Generating Basis State Labels
     
    bases = []
    for i in range(int(2**num_qubits)):
        bases.append(str(_bin_gen(i, num_qubits)))

    # ========================================================================
    # **kwargs stuff
    
    count_bool = kwargs.get('counts', False)
    if count_bool == True:
        ans = np.array(ans)*shots

    return bases, ans




def ProbPlot(qc, q, shots, **kwargs):
    '''
    Plots a histogram of measurement statistics
    '''
    ylabel = 'probability'
    size = (5,3)

    bases, ans = Measure(qc,q,shots = shots)

    plt.figure(figsize=size)
    plt.bar(bases, ans)
    plt.xlabel('Basis States')
    plt.ylabel(ylabel)
    plt.show()

    return bases, ans

