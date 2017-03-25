# -*- coding: utf-8 -*-

import numpy as np


def activationFunc(z):
  return 1/(1+np.exp(-z)) # element-wise 
   
def activationFunc_derivative(z):
  return np.multiply(activationFunc(z),(1-activationFunc(z))) # element-wise