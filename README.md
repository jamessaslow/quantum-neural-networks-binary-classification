# Solving Binary Classification Problems using Quantum Neural Networks

### Contributors: James Saslow
### 7/11/2024

<b> Return to [Full Project Portfolio](https://github.com/jamessaslow/portfolio) </b>

A showcase of quantum neural network PQC's trained to perform binary classification. 

Quantum computing offers promising avenues in
computational parallelism, and leveraging these quantum capabilities is an active area of research for machine learning.
In this study, we explore the applications of Quantum Neural
Networks (QNNs) in binary classification tasks, aiming to develop
efficient models with minimal complexity. We present a simple
QNN model for binary classification as a Parameterized Quantum
Circuit (PQC) to encode input data and perform computations.
Through experiments on 3 benchmark datasets - Iris, Breast
Cancer Wisconsin, and MNIST, we demonstrate the effectiveness
of our QNN approach. Despite the inherent challenges posed
by non-linearly separable data, QNN models exhibit exceptional
performance. Our findings highlight the promise of QNN in
solving real-world classification tasks.

<h2> Table of Contents</h2>

This project contains tutorial modules for training quantum neural networks to perform binary classification on Iris Data, Breast Cancer Wisconsin Data, and MNIST dataset.


## 1) [Engineering a Parameterized Quantum Circuit](https://github.com/jamessaslow/quantum-neural-networks-binary-classification/blob/main/vscode_quantum_neural_network/Quantum%20Neural%20Network%20Code/1.%20generate_pqc.ipynb)
   - An Introduction to Solving Linear Unconstrained Binary Optimization Problems
   - A simple LUBO is formulated as an arbitrary math problem & solved both analytically and experimentally with DWave solvers
## 2) [Fetching Iris Dataset](https://github.com/jamessaslow/quantum-neural-networks-binary-classification/blob/main/vscode_quantum_neural_network/Quantum%20Neural%20Network%20Code/2.1%20fetching_iris_dataset.ipynb)
   - The Subset Sum Problem (SSP) is formally introduced as a practical application of a LUBO problem
   - SSP is modified from its usual form to find the subset that extremizes the target sum amount
   - Solved with brute-force search & QUBO implementation in DWave's hybrid solver
