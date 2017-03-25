# -*- coding: utf-8 -*-

import numpy as np
import activationFunc as af
import matplotlib.pyplot as plt

class neuralNetwork(object):
  def __init__(self,X,y,num_labels,num_hidden_unit,_lambda,step=0):
    self.inputMatrix = X
    self.result = y  # o*m
    self._lambda = _lambda
    
    self.num_labels = num_labels
    self.num_feature = np.size(X,1)
    self.num_examp = np.size(X,0)
    
    # 3 hidden layers  100 hidden units
    self.hidden_unit1 = num_hidden_unit
    self.hidden_unit2 = num_hidden_unit - step;
    self.hidden_unit3 = num_hidden_unit - step -step;
    
    # mapping from inout to the first hidden layer
    self.theta1 = np.zeros([self.hidden_unit1,self.num_feature+1]) # h1*(n+1)
    # from 1st to 2nd layer
    self.theta2 = np.zeros([self.hidden_unit2,self.hidden_unit1+1]) # h2*(h1+1)
    # from 2nd to 3rd layer
    self.theta3 = np.zeros([self.hidden_unit3,self.hidden_unit2+1]) # h3*(h2+1)
    # from 3rd layer to output layer
    self.theta4 = np.zeros([self.num_labels,self.hidden_unit3+1]) # o*(h3+1)
    
    # the final predicted output of input 
    self.info_output = np.zeros([self.num_labels,self.num_examp])
    
    # record all J
    self.cost_values = []
    
    
  # create a random Theta matrix to start iteraton  
  def randomInit(self,row,col):
    randomTheta = np.random.rand(row,col)
    return randomTheta
    
  def forwardPropogation(self,input_matrix):
    a1_bar =  np.append(np.ones([len(input_matrix),1]),input_matrix,1)
    z2 = a1_bar * (self.theta1.T)
    a2 = af.activationFunc(z2)  # the first hidden layer
    
    a2_bar = np.append(np.ones([len(a2),1]),a2,1)
    z3 = a2_bar * (self.theta2.T)
    a3 = af.activationFunc(z3) # the 2nd hidden layer
    
    a3_bar = np.append(np.ones([len(a3),1]),a3,1)
    z4 = a3_bar * (self.theta3.T)
    a4 = af.activationFunc(z4) # the 3rd hidden layer
    
    a4_bar = np.append(np.ones([len(a4),1]),a4,1)
    z5 = a4_bar * (self.theta4.T)
    a5 = af.activationFunc(z5) # the final output layer
    
    return z2,z3,z4, a1_bar,a2_bar,a3_bar,a4_bar, a5.T
    
  def backwardPropogation(self,input_matrix):
    # add a bias unit to input data
    
    hid_layer1,hid_layer2,hid_layer3, a1_bar,a2_bar,a3_bar,a4_bar, self.info_output = self.forwardPropogation(input_matrix)
    
    
    # forward propogation ends-----------------------------------
    # start back propogation ------------------------------------
    
    delta5 = self.info_output - self.result # o*m
    
    theta4_bar = (self.theta4.T)[1:] # leave out the first line  h*o
    delta4 = theta4_bar*delta5 # h*m
    delta4 = np.multiply(delta4,af.activationFunc_derivative(hid_layer3.T))
    
    theta3_bar = (self.theta3.T)[1:] # h*h
    delta3 = theta3_bar*delta4 # h*m
    delta3 = np.multiply(delta3,af.activationFunc_derivative(hid_layer2.T))
    
    theta2_bar = (self.theta2.T)[1:] # h*h
    delta2 = theta2_bar*delta3 # h*m
    delta2 = np.multiply(delta2,af.activationFunc_derivative(hid_layer1.T))
    
  
    D1 = delta2*a1_bar # h*(n+1)
    D2 = delta3*a2_bar
    D3 = delta4*a3_bar
    D4 = delta5*a4_bar
    
    _lambda = self._lambda
    Delta1 = (D1 + _lambda*self.theta1) / self.num_examp
    Delta2 = (D2 + _lambda*self.theta2) / self.num_examp
    Delta3 = (D3 + _lambda*self.theta3) / self.num_examp
    Delta4 = (D4 + _lambda*self.theta4) / self.num_examp

    return Delta1,Delta2,Delta3,Delta4
    
  # compute regulazation 
  def theta_square(self):
    tmp = np.sum(np.square(self.theta1))
    tmp = tmp + np.sum(np.square(self.theta2))
    tmp = tmp + np.sum(np.square(self.theta3))
    tmp = tmp + np.sum(np.square(self.theta4))
    return tmp
    
  # theta is compressed into a big matrix
  def constFunction(self,a5,y,_lambda):
    tmp = np.multiply(y,np.log(a5)) + np.multiply(1-y,np.log(1-a5))
    J = np.sum(tmp)/self.num_examp
    J = -J + _lambda/(2*self.num_examp)*self.theta_square()
    return J
    
  # calculate Theta
  def train(self,learning_rate,motion_factor):
    # random initialize all theta matrices
    self.theta1 = self.randomInit(self.hidden_unit1,self.num_feature+1)
    self.theta2 = self.randomInit(self.hidden_unit2,self.hidden_unit1+1)
    self.theta3 = self.randomInit(self.hidden_unit3,self.hidden_unit2+1)
    self.theta4 = self.randomInit(self.num_labels,self.hidden_unit3+1)
    
    alpha = learning_rate
    sigma = motion_factor
        
    old_delta1 = 0
    old_delta2 = 0
    old_delta3 = 0
    old_delta4 = 0
    old_J = 0
    
    for i in range(500):
      try:
        print("The "+str(i)+" iteration: ")
      # perform backward error propogation
        Delta1,Delta2,Delta3,Delta4 = self.backwardPropogation(self.inputMatrix)
      except(ValueError):
        print('error occurs in'+str(i))
        break
      # show cost value
      J = self.constFunction(self.info_output,self.result,self._lambda)
      print(J)
      self.cost_values.append(J)
      
      if np.abs(old_J-J) <= 1e-3:
        break
      
      # review all thetas 
  
      self.theta1 = self.theta1 - (Delta1*alpha*(1-sigma) + old_delta1*sigma)
      self.theta2 = self.theta2 - (Delta2*alpha*(1-sigma) + old_delta2*sigma)
      self.theta3 = self.theta3 - (Delta3*alpha*(1-sigma) + old_delta3*sigma)
      self.theta4 = self.theta4 - (Delta4*alpha*(1-sigma) + old_delta4*sigma)
      
      old_delta1 = Delta1
      old_delta2 = Delta2
      old_delta3 = Delta3
      old_delta4 = Delta4
      
  def _predict(self,asked_matrix):
    a,b,c,d,e,f,g,h = self.forwardPropogation(asked_matrix)
    return h    

  def predict(self,asked_matrix):
    res = self._predict(asked_matrix)
    return res.argmax(axis=0),res
  

  def show_cost_values(self):
    size = np.size(self.cost_values)
    X = list(range(size))
    fig = plt.figure()
    pl = fig.add_subplot(111)
    pl.scatter(X,self.cost_values,c='red',marker='o')
    pl.plot(X,self.cost_values,'r')
    # plt.show()

    
    
    
    
    
    
    