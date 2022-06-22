"""
Course  : Data Mining II (636-0019-00L)
"""
from matplotlib.transforms import Transform
from utils import *
from pca import *
from impute import *

import scipy.misc as misc
import scipy.ndimage as sn
import numpy as np 
from sklearn.datasets import make_moons
import matplotlib
import matplotlib.pyplot as plt
import seaborn_image as isns
'''
Main Function
'''
if __name__ in "__main__":
    
    # font size in plots:
    fs = 10
    matplotlib.rcParams['font.size'] = fs
    
    #################
    #Exercise 1:
    ranks = np.arange(1, 31) #30
    
    # get image data:
    img = misc.ascent() 
    X = sn.rotate(img, 180) #we rotate it for displaying with seaborn
    
    #generate data matrix with 70% missing values
    X_missing = randomMissingValues(X, per=0.70)
    
    #plot data for comparison
    plt.gray()
    fig, ax = plt.subplots(1, 2)
    isns.imgplot(X, ax=ax[0], cbar=False)

    isns.imgplot(X_missing, ax=ax[1], cbar=False)
    ax[0].set_title('Original')
    ax[1].set_title(f'70% missing data')
    plt.savefig("exercise1_1.pdf")
     
    #Impute data with optimal rank r
    #TODO Implement svd_imputation_optimised with mean imputation
    [X_imputed,r,testing_errors] = svd_imputation_optimised(
        X=X_missing,
        ranks=ranks,
        test_size=0.2
    )

    #plot data for comparison
    plt.gray()
    fig, ax = plt.subplots(1, 3)
    isns.imgplot(X, ax=ax[0], cbar=False)
    isns.imgplot(X_missing, ax=ax[1], cbar=False)
    isns.imgplot(X_imputed, ax=ax[2], cbar=False)
    ax[0].set_title('Original', fontsize=fs)
    ax[1].set_title(f'70% missing data', fontsize=fs)
    ax[2].set_title('Imputed', fontsize=fs)
    plt.savefig("exercise1_2.pdf")


    #TODO Plot testing_errors and highlight optimal rank r (mean imputation)
    #Plot testing_error and highlight optimial rank r
    fig, ax = plt.subplots(1, 1)
    ax.scatter(ranks, testing_errors)
    ax.scatter(ranks[np.where(testing_errors == min(testing_errors))],min(testing_errors))
    ax.set_title("Testing errors for different ranks")
    ax.set_xlabel("Rank")
    ax.set_ylabel("Testing error (MSE)")
    plt.savefig('exercise1_3.pdf')


    

    #Exercise 2
    #load data
    [X, y] = make_moons(n_samples=300,noise=None)
    
    #TODO: Perform a PCA
    #1. Compute covariance matrix
    covariance = computeCov(X)
    #2. Compute PCA by computing eigen values and eigen vectors
    eigen = computePCA(covariance)
    #3. Transform your input data onto a 2-dimensional subspace using the first two PCs
    transformed_data  = transformData(eigen[1], X)
    tr_data_2dim = transformed_data[:,:2]
    #4. Plot your transformed data and highlight the sample classes. 
    plotTransformedData(tr_data_2dim, y, filename="exercise_2_1.pdf")
    #5. How much variance can be explained with each principle component?
    var = np.array(computeVarianceExplained(eigen[0]))
    print("Variance Explained Exercise 2a: ")
    for i in range(var.shape[0]):
        print("PC %d: %.2f"%(i+1,var[i]))



    #TODO:
    #1. Perform Kernel PCA
    #2. Plot your transformed data and highlight the sample classes
    #3. Repeat the previous 2 steps for gammas [1,5,10,50] and compare the results. 

    for i in [1,5,10,50]:
        transformed_data = RBFKernelPCA(X, gamma = i, n_components=2).T

        #plotting transformed data 
        filename = "Kernel_PCA_gamma_{}".format(i)
        plotTransformedData(transformed_data, y, filename)
