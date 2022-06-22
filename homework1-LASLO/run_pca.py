"""
Homework: Principal Component Analysis
Course  : Data Mining II (636-0019-00L)
"""

# Import all necessary functions
from utils import *
from pca import *

'''
Main Function
'''
if __name__ in "__main__":
    # Initialise plotting defaults
    initPlotLib()

    ####################################
    # Exercise 2:
    
    # Simulate Data
    data = simulateData()
    # Perform a PCA
    # 1. Compute covariance matrix
    
    #extract data 
    # print(data)

    X = data['data']
    labels = data['target']
    covariance = computeCov(X)
    # 2. Compute PCA by computing eigen values and eigen vectors
    eigenvalues, eigenvectors = computePCA(covariance)
    # 3. Transform your input data onto a 2-dimensional subspace using the first two PCs
    transformed_data  = transformData(eigenvectors, X)
    #using only the first two PCA
    tr_data_2dim = transformed_data[:,:2]
    # 4. Plot your transformed data and highlight the four different sample classes
    plotTransformedData(tr_data_2dim, labels, filename="exercise_2_1.pdf")
    # 5. How much variance can be explained with each principle component?
    var = np.array(computeVarianceExplained(eigenvalues)) # Compute Variance Explained
    np.set_printoptions(precision=2)
    print("Variance Explained Exercise 2.1: ")
    # Uncomment the following 3 lines!
    for i in range(15):
        print("PC %d: %.2f"%(i+1,var[i]))
    # 6. Plot cumulative variance explained per PC
    plotCumSumVariance(var,filename="cumvar_2_1.pdf")
    ####################################
    # Exercise 2 Part 2:
    
    # 1. Normalise data
    normalised_data = dataNormalisation(X)
    # 2. Compute covariance matrix
    covariance_norm = computeCov(normalised_data)
    # 3. Compute PCA
    eigenvalues_norm, eigenvectors_norm = computePCA(covariance_norm)
    # 4. Transform your input data inot a 2-dimensional subspace using the first two PCs
    transformed_data_norm  = transformData(eigenvectors_norm, normalised_data)
    #using only the first two PCA
    tr_data_norm= transformed_data_norm[:,:2]
    # 5. Plot your transformed data
    plotTransformedData(tr_data_norm, labels, filename="exercise_2_2.pdf")
    # 6. Compute Variance Explained
    var_norm = np.array(computeVarianceExplained(eigenvalues_norm)) # Compute Variance Explained
    np.set_printoptions(precision=2)
    print("Variance Explained Exercise 2.2: ")
    # Uncomment the following 3 lines!
    for i in range(15):
        print("PC %d: %.2f"%(i+1,var_norm[i]))
    # 7. Plot Cumulative Variance
    plotCumSumVariance(var_norm,filename="cumvar_2_2.pdf")