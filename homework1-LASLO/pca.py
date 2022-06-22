import scipy as sp
import scipy.linalg as linalg
import pylab as pl
import numpy as np
import pandas as pd

from utils import plot_color

'''############################'''
'''Principle Component Analyses'''
'''############################'''

'''
Compute Covariance Matrix
Input: Matrix of size #samples x #features
Output: Covariance Matrix of size #features x #features
Note: Do not use scipy or numpy cov. Implement the function yourself.
      You can of course add an assert to check your covariance function
      with those implemented in scipy/numpy.
'''
def computeCov(X=None):
    


    means = np.mean(X, axis = 0)
    mat_wo_means = np.subtract(X, means)
    cov = np.dot(mat_wo_means.T, mat_wo_means)/X.shape[0]


    return cov

'''
Compute PCA
Input: Covariance Matrix
Output: [eigen_values,eigen_vectors] sorted in such a why that eigen_vectors[:,0] is the first principle component
        eigen_vectors[:,1] the second principle component etc...
Note: Do not use an already implemented PCA algorithm. However, you are allowed to use an implemented solver 
      to solve the eigenvalue problem!
'''
def computePCA(matrix=None):
    #computing the eigenvalues and eigen vectors (not ordered)
    eigenvalues, eigenvectors = np.linalg.eig(matrix)

    #making sure they are sorted
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:,idx]
    return eigenvalues, eigenvectors

'''
Transform Input Data Onto New Subspace
Input: pcs: matrix containing the first x principle components
       data: input data which should be transformed onto the new subspace
Output: transformed input data. Should now have the dimensions #samples x #components_used_for_transformation
'''
def transformData(pcs=None,data=None):
    
    #matrix of size principal_comp * features is pcs
    #in pcs v[:,i] is the eigenvector corresponding to the ith largest eigen value
    #to transform the data I need to apply those two vectors to my data
    tr_data = np.zeros((data.shape[0], pcs.shape[1]))
    for i in range(pcs.shape[1]):
        tr_data[:,i] = np.dot(data, pcs[:,i])
    
    return tr_data

'''
Compute Variance Explaiend
Input: eigen_values
Output: return vector with varianced explained values. Hint: values should be between 0 and 1 and should sum up to 1.
'''
def computeVarianceExplained(evals=None):

    total_var = sum(evals)
    explained_var = evals/total_var

    return explained_var


'''############################'''
'''Different Plotting Functions'''
'''############################'''

'''
Plot Transformed Data
Input: transformed: data matrix (#samples x 2)
       labels: target labels, class labels for the samples in data matrix
       filename: filename to store the plot
'''
def plotTransformedData(transformed=None,labels=None,filename="exercise1.pdf"):
    pl.figure()
    pl.scatter (transformed[:,0], transformed[:,1], c = labels)
    pl.xlabel ("PC_1")
    pl.ylabel ("PC_2")
    pl.title("Visualization of transformed data \n using the first two principal components")
    # You can use plot_color[] to obtain different colors for your plots
    # Save File
    pl.savefig(filename)

'''
Plot Cumulative Explained Variance
Input: var: variance explained vector
       filename: filename to store the file
'''
def plotCumSumVariance(var=None,filename="cumsum.pdf"):
    pl.figure()
    pl.plot(sp.arange(var.shape[0]),sp.cumsum(var)*100)
    pl.xlabel("Principle Component")
    pl.ylabel("Cumulative Variance Explained in %")
    pl.grid(True)
    # Save file
    pl.savefig(filename)



'''############################'''
'''Data Preprocessing Functions'''
'''############################'''

'''
Exercise 2 Part 2:
Data Normalisation (Zero Mean, Unit Variance)
'''
def dataNormalisation(X=None):

    mean = np.mean(X, axis = 0)
    st_dev = np.std(X, axis = 0 )
    return (X- mean)/st_dev
