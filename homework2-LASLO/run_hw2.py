#import all necessary functions
from utils import *
from pca import *
from pinv import *
from noise_reduction import *

'''
Main Function
'''
if __name__ in "__main__":
    # Initialise plotting defaults
    initPlotLib()

    ####################################
    # Exercise 1:

    # Get Iris Data
    data = loadIrisData()


  # Perform a PCA
    # 1. Compute covariance matrix

    X = data['data']
    labels = data['target']
    covariance = computeCov(X)
    # 2. Compute PCA by computing eigen values and eigen vectors
    eigen = computePCA(covariance)
    print(eigen[0].shape)
    # 3. Transform your input data onto a 2-dimensional subspace using the first two PCs
    transformed_data  = transformData(eigen[1], X)
    #using only the first two PCA
    tr_data_2dim = transformed_data[:,:2]
    # 4. Plot your transformed data and highlight the four different sample classes
    plotTransformedData(tr_data_2dim, labels, filename="exercise_2_1.pdf")
    # 5. How much variance can be explained with each principle component?
    var = np.array(computeVarianceExplained(eigen[0])) # Compute Variance Explained
    np.set_printoptions(precision=2)
    print("Variance Explained Exercise 2.1: ")
    # Uncomment the following 3 lines!
    for i in range(var.shape[0]):
        print("PC %d: %.2f"%(i+1,var[i]))
    # 6. Plot cumulative variance explained per PC
    plotCumSumVariance(var,filename="cumvar_2_1.pdf")

    # Perform a PCA using SVD
    # 1. Normalise data
    normalised_data = zeroMean(X)
    # 2. Compute covariance matrix
    covariance_norm = computeCov(normalised_data)
    # 3. Compute PCA
    eigen_norm = computePCA(covariance_norm)
    # 4. Transform your input data inot a 2-dimensional subspace using the first two PCs
    transformed_data_norm  = transformData(eigen_norm[1], normalised_data)
    #using only the first two PCA
    tr_data_norm= transformed_data_norm[:,:2]
    # 5. Plot your transformed data
    plotTransformedData(tr_data_norm, labels, filename="exercise_2_2.pdf")
    # 6. Compute Variance Explained
    var_norm = np.array(computeVarianceExplained(eigen_norm[0])) # Compute Variance Explained
    np.set_printoptions(precision=2)
    print("Variance Explained Exercise 2.2: ")
    # Uncomment the following 3 lines!
    for i in range(var_norm.shape[0]):
       print("PC %d: %.2f"%(i+1,var_norm[i]))
    # 7. Plot Cumulative Variance
    plotCumSumVariance(var_norm,filename="cumvar_2_2.pdf")

    ####################################
    # Exercise 2:
    # 1. Compute the Moore-Penrose Pseudo-Inverse on the Iris data
    ps_inv = compute_pinv(X)


    # 2. Check Properties

    print("\nChecking status exercise 3:")
    status = False

    if np.allclose(np.dot(X,np.dot(ps_inv,X)) , X): 
        status = True
    print(f"X X^+ X = X is {status}")


    status = False
    if np.allclose(np.dot(ps_inv,np.dot(X,ps_inv)), ps_inv): 
        status = True
    print(f"X^+ X X^+ = X^+ is {status}")



    # Exercise 3
    ####################################
    # 1. Loading the images
    img_sp = imageio.imread('images/greece_s&p.jpg', as_gray = True)
    
    # 2.  Perform noise reduction via SVD, i.e. call the function NoiseReduction

    img = NoiseReduction(img_sp , 1000)

    # 3. Save the denoised images using the command: 
    imageio.imwrite('images/red_1000.jpg', img)

    img = NoiseReduction(img_sp , 3000)

    # 3. Save the denoised images using the command: 
    imageio.imwrite('images/red_3000.jpg', img)


    img = NoiseReduction(img_sp , 500)

    # 3. Save the denoised images using the command: 
    imageio.imwrite('images/red_500.jpg', img)
