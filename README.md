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


For a full report, please see [report](https://github.com/jamessaslow/quantum-neural-networks-binary-classification/blob/main/vscode_quantum_neural_network/Report/Solving_Binary_Classification_Problems_with_Quantum_Neural_Networks.pdf)

<h2> Table of Contents</h2>

This project contains tutorial modules for training quantum neural networks to perform binary classification on Iris Data, Breast Cancer Wisconsin Data, and MNIST dataset.


## 1) [Engineering a Parameterized Quantum Circuit](https://github.com/jamessaslow/quantum-neural-networks-binary-classification/blob/main/vscode_quantum_neural_network/Quantum%20Neural%20Network%20Code/1.%20generate_pqc.ipynb)
   - Designing a 3-qubit parameterized quantum circuit to perform quantum machine learning
## 2.1) [Fetching Iris Dataset](https://github.com/jamessaslow/quantum-neural-networks-binary-classification/blob/main/vscode_quantum_neural_network/Quantum%20Neural%20Network%20Code/2.1%20fetching_iris_dataset.ipynb)
   - Scraping Iris Dataset from UCI ML REPO and performing data pre-processing
## 2.2) [Fetching Breast Cancer Dataset](https://github.com/jamessaslow/quantum-neural-networks-binary-classification/blob/main/vscode_quantum_neural_network/Quantum%20Neural%20Network%20Code/2.2%20fetching_breast_cancer_dataset.ipynb)
   - Scraping Breast Cancer Wisconsin Dataset from UCI ML REPO and performing data pre-processing
## 2.3) [Fetching MNIST Dataset](https://github.com/jamessaslow/quantum-neural-networks-binary-classification/blob/main/vscode_quantum_neural_network/Quantum%20Neural%20Network%20Code/2.3%20fetching_MNIST_dataset.ipynb)
   - Scraping MNIST Dataset from the Keras Library and performing data pre-processing

## 3.1) [QNN for Binary Classification of Iris Data](https://github.com/jamessaslow/quantum-neural-networks-binary-classification/blob/main/vscode_quantum_neural_network/Quantum%20Neural%20Network%20Code/3.1%20QNN_iris.ipynb)
   - Training the PQC via gradient descent to perform binary classification on the Iris dataset.

## 3.2) [QNN for Binary Classification of Breast Cancer Data](https://github.com/jamessaslow/quantum-neural-networks-binary-classification/blob/main/vscode_quantum_neural_network/Quantum%20Neural%20Network%20Code/3.2%20QNN_breast_cancer.ipynb)
   - Training the PQC via gradient descent to perform binary classification on the Breast Cancer dataset.

## 3.3) [QNN for Binary Classification of MNIST Data](https://github.com/jamessaslow/quantum-neural-networks-binary-classification/blob/main/vscode_quantum_neural_network/Quantum%20Neural%20Network%20Code/3.3%20QNN_MNIST.ipynb)
   - Training the PQC via gradient descent to perform binary classification on the MNIST dataset.
