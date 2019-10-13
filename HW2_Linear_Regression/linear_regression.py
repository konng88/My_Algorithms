"""
Do not change the input and output format.
If our script cannot run your code or the format is improper, your code will not be graded.

The only functions you need to implement in this template is linear_regression_noreg, linear_regression_invertible，regularized_linear_regression,
tune_lambda, test_error and mapping_data.
"""

import numpy as np
import pandas as pd

###### Q1.1 ######
def mean_absolute_error(w, X, y):
    """
    Compute the mean absolute error on test set given X, y, and model parameter w.
    Inputs:
    - X: A numpy array of shape (num_samples, D) containing test feature.
    - y: A numpy array of shape (num_samples, ) containing test label
    - w: a numpy array of shape (D, )
    Returns:
    - err: the mean absolute error
    """
    #####################################################
    # TODO 1: Fill in your code here #
    #####################################################
    N = len(X)
    err = np.sum(abs(np.dot(w,X.T) - y))
    err = err / N
    return err

###### Q1.2 ######
def linear_regression_noreg(X, y):
  """
  Compute the weight parameter given X and y.
  Inputs:
  - X: A numpy array of shape (num_samples, D) containing feature.
  - y: A numpy array of shape (num_samples, ) containing label
  Returns:
  - w: a numpy array of shape (D, )
  """
  #####################################################
  #	TODO 2: Fill in your code here #
  #####################################################
  w = np.dot(np.dot(np.linalg.inv(np.dot(X.T,X)),X.T),y)
  return w

###### Q1.3 ######
def linear_regression_invertible(X, y):
    """
    Compute the weight parameter given X and y.
    Inputs:
    - X: A numpy array of shape (num_samples, D) containing feature.
    - y: A numpy array of shape (num_samples, ) containing label
    Returns:
    - w: a numpy array of shape (D, )
    """
    #####################################################
    # TODO 3: Fill in your code here #
    #####################################################
    D = len(X[0])
    M = np.dot(X.T,X)
    adjustment = np.eye(D) * 0.1
    eig_values,eig_vectors = np.linalg.eig(M)
    min_eig_val = np.min(abs(eig_values))
    while min_eig_val < 10 ** (-5):
        M = M + adjustment
        eig_values,eig_vectors = np.linalg.eig(M)
        min_eig_val = np.min(abs(eig_values))
    w = np.dot(np.dot(np.linalg.inv(M),X.T),y)
    return w


###### Q1.4 ######
def regularized_linear_regression(X, y, lambd):
    """
    Compute the weight parameter given X, y and lambda.
    Inputs:
    - X: A numpy array of shape (num_samples, D) containing feature.
    - y: A numpy array of shape (num_samples, ) containing label
    - lambd: a float number containing regularization strength
    Returns:
    - w: a numpy array of shape (D, )
    """
  #####################################################
  # TODO 4: Fill in your code here #
  #####################################################
    D = len(X[0])
    M = np.dot(X.T,X)
    adjustment = np.eye(D) * lambd
    M = M + adjustment
    w = np.dot(np.dot(np.linalg.inv(M),X.T),y)
    return w

###### Q1.5 ######
def tune_lambda(Xtrain, ytrain, Xval, yval):
    """
    Find the best lambda value.
    Inputs:
    - Xtrain: A numpy array of shape (num_training_samples, D) containing training feature.
    - ytrain: A numpy array of shape (num_training_samples, ) containing training label
    - Xval: A numpy array of shape (num_val_samples, D) containing validation feature.
    - yval: A numpy array of shape (num_val_samples, ) containing validation label
    Returns:
    - bestlambda: the best lambda you find in lambds
    """
    #####################################################
    # TODO 5: Fill in your code here #
    #####################################################

    lambds = []

    for i in range(-18,20):
        lambds.append(10 ** i)
    bestlambda = 10 ** (-19)
    best_w = regularized_linear_regression(Xtrain, ytrain, bestlambda)
    best_MAE = mean_absolute_error(best_w, Xval, yval)
    for lambd in lambds:
        w = regularized_linear_regression(Xtrain, ytrain, lambd)
        MAE = mean_absolute_error(w, Xval,yval)
        if MAE < best_MAE:
            bestlambda = lambd
            best_w = w
            best_MAE = MAE
    return bestlambda


###### Q1.6 ######
def mapping_data(X, power):
    """
    Mapping the data.
    Inputs:
    - X: A numpy array of shape (num_training_samples, D) containing training feature.
    - power: A integer that indicate the power in polynomial regression
    Returns:
    - X: mapped_X, You can manully calculate the size of X based on the power and original size of X
    """
    #####################################################
    # TODO 6: Fill in your code here #
    #####################################################
    mapped_X = X

    # print(X.shape)
    # print(X1.shape)
    for i in range(2,power+1):
        polyX = X ** i
        mapped_X = np.hstack((mapped_X,polyX))
    return mapped_X
