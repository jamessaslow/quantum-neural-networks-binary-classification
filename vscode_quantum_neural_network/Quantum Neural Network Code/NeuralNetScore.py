'''
NeuralNetScore.py
James Saslow
4/30/2024


A .py file that does returns a Confusion Matrix and other binary classification 
metrics for a neural network

'''

# Importing Packages
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import display


# Prototyping a function that assesses binary classifier performance on test data
def nn_confusion(model_prediction, binary_targets_test, x_test):
    '''
    Assesses the performance of a neural net 
    classifier on binary test data

    '''
    
    num_data = len(binary_targets_test) # Number of data in test dataset


    # Applying a threshold function to the model predictions
    prediction = np.round(model_prediction)

    num_misclassifications = 0

    TP = 0 # True Positives
    TN = 0 # True Negatives
    FP = 0 # False Positives
    FN = 0 # False Negatives


    prediction_record = []

    for i in range(num_data):
        p = prediction[i]
        d_class = binary_targets_test[i]

        if p == d_class:
            if p == 0:
                # True Negative
                TN +=1
                prediction_record.append("red")
            elif p == 1:
                # True Positive
                TP +=1
                prediction_record.append("lime")
        if p != d_class:
            if p == 0:
                # False Negative
                FN += 1
                prediction_record.append("orange")
            elif p == 1:
                # False Positive
                FP += 1
                prediction_record.append("green")


    confusion_labels = ["True Positive", "False Positive", "False Negative", "True Negative"]
    confusion_outputs = [TP, FP, FN, TN]
        

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
    axs[1].scatter(x_test[:,0],x_test[:,1] , marker='o', edgecolor='black', c=prediction_record)
    axs[1].set_xlabel('x')
    axs[1].set_ylabel('y')
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


    pass

