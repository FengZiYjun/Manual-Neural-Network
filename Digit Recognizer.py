# -*- coding: utf-8 -*-

# 1. read data and store in matrixs
# 2. train neural network
# 3. predict

#import csv
import numpy as np
import pandas as pd
#import neurolab as nl

#import matplotlib as plt
import neuralNetwork as NN

# read the file and form a DataFrame
#dfobj = pd.read_csv("D:/Data Science Experiment/Kaggle/Kaggle Digit Recognizer/train.csv")
dfobj = pd.read_csv("D:/Data Science Experiment/Kaggle/Kaggle Digit Recognizer/small train.csv")


# transfer DataFrame into numpy matrix
features = list(dfobj.columns)
features = features[1:]
X = np.matrix(dfobj[features])  # this must be matrix because 
# there is difference between operation * of matrix and array 
y = np.array(dfobj['label'])

# number of labels to classify
num_labels = len(set(y))

# deal with y
num_examp = np.size(X,0)
tmp = np.zeros([num_labels,num_examp])
p = list(y)
for i in range(num_examp):
    tmp[p[i]-1][i] = 1

y = tmp


# X, y, num_labels, num_hidden_unit, _lambda, step
neuralNetwork = NN.neuralNetwork(X,y,num_labels,500,10,100)

# learning_rate,0<motion_factor<1
neuralNetwork.train(0.8,0)


# read the predict files and examine the correctness rate
#dfobj = pd.read_csv("D:/Data Science Experiment/Kaggle/Kaggle Digit Recognizer/small predict.csv")
matrix = np.array(dfobj)
ask = matrix[:,1:]
ans = matrix[:,0]


pred,res = neuralNetwork.predict(ask)
print(pred)
neuralNetwork.show_cost_values()
'''
para1 = neuralNetwork.theta1
para2 = neuralNetwork.theta2
para3 = neuralNetwork.theta3
para4 = neuralNetwork.theta4
'''
print("finished!")

# why there are 'nan' output when computing J ? 
# how to improve?





