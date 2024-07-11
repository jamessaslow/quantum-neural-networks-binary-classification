'''
QuantumNeuralNetwork.py
James Saslow
4/27/2024

Requires Qiskit version 1.0.2 or later
Requires Qfuncs5.py

A .py file that holds a class object 'QuantumNeuralNetwork' used for training and assessing performance
of a 2-input Parameterized Quantum Circuit with 4 trainable model parameters.

Public Use Functions in this code are:

 - Class: QuantumNeuralNetwork
    - QuantumNeuralNetwork.train()
            - Trains QNN and prints training performance on training data per epoch

    - QuantumNeuralNetwork.get()
            - Returns model parameters at any training instance

    - QuantumNeuralNetwork.confusion_matrix()
            - Returns confusion matrix statistics of model performance on test data

    - QuantumNeuralNetwork.learning_graph()
            - Returns a plot of cost function as a function of epoch

    
'''

#===================================================================================================
#Importing Packages
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import display

from qiskit.circuit import Parameter
from qiskit.quantum_info import Operator
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import UnitaryGate
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile, qpy

import Qfuncs5 as qf # My own custom Qiskit Package

#===================================================================================================

# Defining a Quantum Neural Network Class for training on our PQC
class QuantumNeuralNetwork:
    '''
    Class Description:
    An object used to train a 2 input PQC with 4 trainable parameters. Particularly developed to 
    handle parameterized_qc.qpy which can be retrieved from 1. generat_pqc.ipynb. 

    Public Use functions are:
    - QuantumNeuralNetwork.train()
            - Trains QNN and prints training performance on training data per epoch

    - QuantumNeuralNetwork.get()
            - Returns model parameters at any training instance

    - QuantumNeuralNetwork.confusion_matrix()
            - Returns confusion matrix statistics of model performance on test data

    - QuantumNeuralNetwork.learning_graph()
            - Returns a plot of cost function as a function of epoch
    '''

    def __init__(self,qc,x_train, binary_targets_train, x_test, binary_targets_test, **kwargs):
        '''
        Function Description:
        Initializing Variables in Class

        Params:
            - qc                    = QuantumCircuit Qiskit Object - A PQC with 2 inputs and 4 
                                     trainable parameters
            - x_train               = (len(x_train) x 2) dimensional vector of training data
            - binary_targets_train  = Array containing 0's and 1's of classification training data
            - x_test                = (len(x_test) x 2) dimensional vector of testing data
            - binary_targets_test   = Array containing 0's and 1's of classification test data

        **kwargs:
            - epoch = number of epochs in QNN training (Default = 30)
            - lr    = the learning rate (Default = 1)
            - w_vec = Setting the model parameters, a numpy array of length 4
                      (Default: randomly generated array of length 4 which each
                       element taking on a value between 0 and 2*pi)
            - shots = The number of shots the PQC runs to gather measurement statistics
                      (Default: Shots = 500_000)
    
        '''

        self.qc     = qc

        x0_train, x1_train = np.transpose(x_train)
        self.x0_train = x0_train
        self.x1_train = x1_train

        self.binary_targets_train = binary_targets_train
        
        x0_test, x1_test = np.transpose(x_test)
        self.x0_test = x0_test
        self.x1_test = x1_test
        
        self.binary_targets_test = binary_targets_test

        # ==============================================
        # Programming kwargs to __init__

        # Epoch
        if 'epoch' in kwargs:
            epoch = kwargs['epoch']
        else:
            epoch = 30 
        self.epoch = epoch

        # lr (learning rate)
        if 'lr' in kwargs:
            lr = kwargs['lr']
        else:
            lr = 1
        self.lr = lr
    
        # w_vec
        if 'w_vec' in kwargs:
            w_vec =  kwargs['w_vec']
        else:
            w_vec = 2*np.pi*np.random.random(4)
        self.w_vec = w_vec

        # shots
        if 'shots' in kwargs:
            shots = kwargs['shots']
        else:
            shots = 500_000
        self.shots = shots

        # ==============================================
        # Other Private Variables
        cost_per_epoch = np.zeros((self.epoch))
        self.cost_per_epoch = cost_per_epoch


    def get(self):
        '''
        Getter Function for the model Parameters
        '''
        w_vec = self.w_vec
        return w_vec


    def _QNN_output(self,x_vec, omega_vec):
        '''
        Function Description:
        Returns the probability of measuring the |1> state of the classifier qubit

        Params:
            - x_vec     = An array of length 2 containing the input features for
                      a given training point x_{i}
            - omega_vec = An array of length 4 containing the model parameters

        '''
        qc = self.qc
        shots = self.shots

        # Calling the classical register
        c = qc.cregs[0]

        # Calling the Parameters
        parameters = qc.parameters

        # Creating a Dictionary of the parameters and their values
        param_values = np.hstack((omega_vec, x_vec))
        parameter_values = {param: val for param, val in zip(parameters, param_values)}

        # Assign the parameter values to the circuit
        bound_qc = qc.assign_parameters(parameter_values)

        # Extracting Probabilities via Measurement
        bases, probs =  qf.Measure(bound_qc,c, shots = shots)
        p0,p1 = probs
        
        # Returning the Probability of measuring the |1> state
        return p1

    
    def _cost_function(self,p1,d_class):
        '''
        Function Description:
        Computes the cost of a single class assigment 'd_class' with prediction 'p1'

        Params:
            - p1        = The probability of measuring the |1> state - The output of
                          the QNN_output() function
            - d_class   = The true class assignment of the data point x_{i}
        
        '''
        return 0.5*(d_class - p1)**2 

    def _live_misclassification_detection(self, p1, d_class):
        '''
        Function Description:
        Computes Whether or not QNN output 'p1' is a valid prediction for the 'binary target'
        - Returns 'False' if there is not a misclassification
        - Returns 'True' if there is a misclassication

        Params:
            - p1        = The Probability of measuring the |1> state - The output of
                          the QNN_output() function
            - d_class   = The true class assignment of the data point x_{i}
        '''
        prediction  = np.round(p1)
        if prediction == d_class:
            return False
        else:
            return True


    def _cost_function_gradient(self,x_vec, omega_vec, d_class):
        '''
        Function Description:
        Computes the gradient of the cost function with respect
        the the weights vector 'omega' in only one QNN run. 
        In addition, since it computes a QNN already, it also 
        returns the cost function of the particular x_{i} training
        point, its misclassification status, and the dC_dw derivative
        to update the model parameters.

        Params:
            - x_vec     = An array of length 2 containing the input features for
                      a given training point x_{i}
            - omega_vec = An array of length 4 containing the model parameters
            - d_class   = The true class assignment of the data point x_{i}

        '''
        qc = self.qc
        
        x0,x1 = x_vec
        w0,w1,w2,w3 = omega_vec

        # Computing the Phases of the Eigenvalues of U
        alpha0 = x0*w0 + x1*w2
        alpha1 = x0*w0 + x1*w3
        alpha2 = x0*w1 + x1*w2
        alpha3 = x0*w1 + x1*w3

        # Derivative of prob(|1>) wrt to omega_k
        dP_dw0 = (x0/8)*(np.sin(alpha0) + np.sin(alpha1))
        dP_dw1 = (x0/8)*(np.sin(alpha2) + np.sin(alpha3))
        dP_dw2 = (x1/8)*(np.sin(alpha0) + np.sin(alpha2))
        dP_dw3 = (x1/8)*(np.sin(alpha1) + np.sin(alpha3))

        # Calculating prob(|1>)
        p1 = self._QNN_output(x_vec,omega_vec)

        # Calculating Cost Function
        cost = self._cost_function(p1,d_class)

        misclassification = self._live_misclassification_detection(p1, d_class)

        # Derivative of the cost function with respect to prob(|1>)
        dC_dP = -(d_class - p1)

        # Derivative of the cost function with respect to omega_k
        dC_dw = dC_dP* np.array([dP_dw0, dP_dw1, dP_dw2, dP_dw3]) # NO SCALING BY s HERE! Include that when you import data

        return cost , misclassification , dC_dw
    

    

    def train(self):
        '''
        Function Description:
        Iteratively performs gradient descent 'epoch' number of times, prints
        QNN training status, objective function, and model error all in real
        time. 

        '''
        
        qc                   = self.qc
        x0_train             = self.x0_train
        x1_train             = self.x1_train
        binary_targets_train = self.binary_targets_train
        w_vec                = self.w_vec        
        lr                   = self.lr
        epoch                = self.epoch
        cost_per_epoch       = self.cost_per_epoch

        s = len(x0_train)  # Number of training samples




        # Training Our Model
        for k in range(epoch):
            total_cost = 0 # Recording model cost per epoch
            num_misclassifications = 0 # Recording Number of misclassifications per epoch
            for i in range(s):
                d_class = binary_targets_train[i]           # True Class Assignment
                x_vec   = [ x0_train[i], x1_train[i] ]      # Model Inputs

                cost, misclassification, dC_dw= self._cost_function_gradient(x_vec, w_vec, d_class)

                w_vec -= lr/s * dC_dw # Gradient Descent
                total_cost += cost/ s        # Adding cost per data sample to self
                if misclassification == True:
                    num_misclassifications +=1
       
                # if i > 5:
                #     my_string= str(np.round(100 * num_misclassifications / (i + 1),4))
                #     print('\033[KModel Error: ' + my_string + '%', end='\r') 
            


            cost_per_epoch[k] = total_cost

            
            # Keep these print statements as they pop up
            print('============================================================')
            print('Epoch ' + str(k+1))
            print('Cost Function : ', total_cost)
            print('Model Error   : ',100*num_misclassifications/s , '%')
            print('omega_vec = ', w_vec)
            print('============================================================')
            print(' ')

        # Updating Model Parameters
        self.w_vec = w_vec


    def confusion_matrix(self, **kwargs):
        '''
        Function Description:
            1) Returns a histogram plot that assesses performance of the binary classifier by
               displaying the number of True Positives (Lime), True Negatives (Red), 
               False Positives (Green), and False Negatives (Orange) in test data.
               Each catagory is color coded to be assessed in the next plot
            2) Returns a scatter plot of test data with each point color coded to correspond to 
               being detected by the classifier as a True Positive, True Negative, False
               Positive, or False Negative. 

        **kwargs:
            - xlabel = string xlabel on scatter plot (Default = 'x')
            - ylabel = string ylabel on scatter plot (Default = 'y')
            - vertical = bool ~ True => Vertical Plotting (Default => Horizontal Plotting)
        
        '''
        # xlabel
        if 'xlabel' in kwargs:
            xlabel = kwargs['xlabel']
        else:
            xlabel = 'x'

        if 'ylabel' in kwargs:
            ylabel = kwargs['ylabel']
        else:
            ylabel = 'y'

        # Vertical Plotting
        vertical_bool = kwargs.get('vertical', False)



        w_vec   = self.w_vec # Calling our model
        x0_test = self.x0_test
        x1_test = self.x1_test
        binary_targets_test = self.binary_targets_test

        num_data = len(x0_test) # Number of test data points

        TP = 0 # True Positive
        TN = 0 # True Negative
        FP = 0 # False Positive
        FN = 0 # False Negative

        prediction_record = []

        # Assessing Classification performance of the model on the test data via confusion matrix
        for i in range(num_data):
            prob1 = self._QNN_output([x0_test[i],x1_test[i]], w_vec)
            prediction = np.round(prob1)       
            # prediction_record.append(prob1)
            d_class = binary_targets_test[i]

            if prediction == d_class:
                if prediction == 0:
                    # True Negative
                    TN += 1
                    prediction_record.append('red')
                elif prediction == 1:
                    # True Positive
                    TP += 1
                    prediction_record.append('lime')
            if prediction != d_class:
                if prediction == 0:
                    # False Negative
                    FN +=1
                    prediction_record.append('orange')
                elif prediction == 1:
                    # False Positive
                    FP +=1
                    prediction_record.append('green')


        # Making a Bar Chart of the Confusion Matrix Outputs

        confusion_labels = ["True Positive", "False Positive", "False Negative", "True Negative"]
        confusion_outputs = [TP, FP, FN, TN]


        if vertical_bool == True:
            # Plotting the Figures Vertically
            fig, axs = plt.subplots(2, 1, figsize=(6, 10))
        else:
            # Plotting the Figure Horizontally
            fig, axs = plt.subplots(1, 2, figsize=(12, 5))

        # Plot 1: Confusion Matrix on Test Data
        axs[0].set_title('Confusion Matrix on Test Data')
        axs[0].tick_params(axis='x', rotation=35)
        axs[0].set_ylabel('Occurrences')
        bars = axs[0].bar(confusion_labels, confusion_outputs, color=['lime', 'green', 'orange', 'red'])
        for bar, count in zip(bars, confusion_outputs):
            axs[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height(), str(count), ha='center', va='bottom')

        # Plot 2: Test Data Assignment
        axs[1].set_title('Test Data Assignment')
        axs[1].scatter(x0_test, x1_test, marker='o', edgecolor='black', c=prediction_record)
        axs[1].set_xlabel(xlabel)
        axs[1].set_ylabel(ylabel)
        plt.tight_layout()
        plt.show()


        # Calculating Other Binary Classifier Performance Measures
        accuracy = (TP + TN)/num_data
        precision = TP/(TP + FP)
        TPR = TP/(TP + FN)
        specificity = TN / (TN + FP)
        FPR = 1- specificity


        data = {
            "Accuracy" : accuracy,
            "Precision": precision,
            "True Positive Rate": TPR,
            "Specificity" : specificity,
            "False Positive Rate": FPR 
        }

        df = pd.DataFrame(data, index = ["metrics"])
        display(df)



    def learning_graph(self):
        '''
        Function Description:
        Creates a plot of objective function as a function of epoch

        '''
        epoch = self.epoch
        cost_per_epoch = self.cost_per_epoch
        lr = self.lr

        epoch_array = np.arange(1,epoch+1,1)


        plt.figure(figsize = (5,4))
        plt.title('Cost Function vs Epoch for lr = '+str(lr))
        plt.xlabel('Epoch')
        plt.ylabel('Cost Function')
        plt.plot(epoch_array, cost_per_epoch)
        plt.show()

