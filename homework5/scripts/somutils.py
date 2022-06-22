"""
Homework: Self-organizing maps
Course  : Data Mining II (636-0019-00L)

Auxiliary functions to help in the implementation of an online version
of the self-organizing map (SOM) algorithm.
"""
# Author: Dean Bodenham, May 2016
# Modified by: Damian Roqueiro, May 2017
# Modified by: Christian Bock, April 2021

from cmath import nan
from sklearn import datasets
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.patches as mpatches


"""
A function to create the S curve
"""
def makeSCurve():
    n_points = 1000
    noise = 0.2
    X, color = datasets.make_s_curve(n_points, noise=noise, random_state=0)
    Y = np.array([X[:,0], X[:,2]])
    Y = Y.T
    # Stretch in all directions
    Y = Y * 2
    
    # Now add some background noise
    xMin = np.min(Y[:,0])
    xMax = np.max(Y[:,0])
    yMin = np.min(Y[:,1])
    yMax = np.max(Y[:,1])
    
    n_bg = n_points//10
    Ybg = np.zeros(shape=(n_bg,2))
    Ybg[:,0] = np.random.uniform(low=xMin, high=xMax, size=n_bg)
    Ybg[:,1] = np.random.uniform(low=yMin, high=yMax, size=n_bg)
    
    Y = np.concatenate((Y, Ybg))
    return Y


"""
Plot the data and SOM for the S-curve
  data: 2 dimensional dataset (first two dimensions are plotted)
  buttons: N x 2 array of N buttons in 2D
  fileName: full path to the output file (figure) saved as .pdf or .png
"""
def plotDataAndSOM(data, buttons, fileName):
    fig = plt.figure(figsize=(8, 8))
    # Plot the data in grey
    plt.scatter(data[:,0], data[:,1], c='grey')
    # Plot the buttons in large red dots
    plt.plot(buttons[:,0], buttons[:,1], 'ro', markersize=10)
    # Label axes and figure
    plt.title('S curve dataset, with buttons in red')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.savefig(fileName)


# Important note:
# 
# Most of the functions below are currently just headers. Provide a function
# body for each of them. 
#
# In case you want to create your own functions with their own interfaces, adjust
# the rest of the code appropriately.

"""
Create a grid of points, dim p x q, and save grid in a (p*q, 2) array
  first column: x-coordinate
  second column: y-coordinate
"""
def createGrid(p, q):
    index = 0
    grid = np.zeros(shape=(p*q, 2))
    for i in range(p):
        for j in range(q):
            index = i*q + j
            grid[index, 0] = i
            grid[index, 1] = j
    return grid


"""
A function to plot the crabs results
It applies a SOM previously computed (parameters grid and buttons) to a given
dataset (parameters data)

Parameters
 X : is the original data that was used to compute the SOM.
     Rows are samples and columns are features.
 idInfo : contains the information (sp and sex for the crab dataset) about
          each data point in X.
          The rows in idInfo match one-to-one to rows in X.
 grid, buttons : obtained from computing the SOM on X.
 fileName : full path to the output file (figure) saved as .pdf or .png
"""
def plotSOMCrabs(X, p, q, idInfo, grid, buttons, fileName):
    # Use the following colors for samples of each pair [species, sex]
    # Blue male:     dark blue #0038ff
    # Blue female:   cyan      #00eefd
    # Orange male:   orange    #ffa22f
    # Orange female: yellow    #e9e824

    # TODO replace statement below with function body

    #defining colors

    colors = ["#0038ff", "#00eefd", "#ffa22f", "#e9e824"]


    #first define the circles by the coordinates and store them in a list 

    circle_coords = []
    for i in range(0, p):
        for j in range(0, q):
            circle_coords.append([i,j])


    labels = []
    for i in range(X.shape[0]):

        x_i = X[i, :]
        closest_button  = findNearestButtonIndex(x_i, buttons)
        labels.append(closest_button)

    fig, ax = plt.subplots(1, figsize = (10,10))

    for i, label in enumerate(labels):
        info = idInfo[i,:]
        color_crab = info[0]
        gender = info[1]

        if (color_crab =="B" and gender == "M"): color_point = colors[0]
        if (color_crab =="B" and gender == "F"): color_point = colors[1]
        if (color_crab =="O" and gender == "M"): color_point = colors[2]
        if (color_crab =="O" and gender == "F"): color_point = colors[3]

        # sns.scatterplot(x= label_coords[i,0], y = label_coords[i][1], x_jitter=0.2, y_jitter=0.2, palette = color_point)
        sns.regplot(x=np.array([grid[label,0]]), y=np.array([grid[label,1]]), fit_reg=False, x_jitter=0.2, y_jitter=0.2, color=color_point, ax=ax)
    c1 = mpatches.Circle((0,0),color=colors[0], label="Blue Male")
    c2 = mpatches.Circle((0,0),color=colors[1], label="Blue Female")
    c3 = mpatches.Circle((0,0),color=colors[2], label="Orange Male")
    c4 = mpatches.Circle((0,0),color=colors[3], label="Orange Female")

    plt.legend(handles = [c1,c2,c3,c4], loc = "lower right", bbox_to_anchor=(1,1)) 
    
    for circle in circle_coords:
        plt.scatter(circle[0], circle[1], s=2200, facecolors = 'none', edgecolors="black")
    
    plt.title('A SOM applied to crab data')
    plt.xlabel('p')
    plt.ylabel('q')
    plt.savefig(fileName)


"""
Function for computing distance in grid space.
Use Euclidean distance.
"""
def getGridDist(z0, z1):
    # TODO replace statement below with function body
    dist = np.linalg.norm(z0-z1)

    return dist


"""
Function for computing distance in feature space.
Use Euclidean distance.
"""
def getFeatureDist(z0, z1):
    
    dist = np.linalg.norm(z0-z1)

    return dist



"""
Create distance matrix between points numbered 1,2,...,K=p*q from grid
"""
def createGridDistMatrix(grid):
   
    dist_matrix =np.zeros((grid.shape[0], grid.shape[0]))
    for i in range(grid.shape[0]):
        for j in range(grid.shape[0]):
            dist_matrix [i, j] = getGridDist(grid[i,:], grid[j,:])
    
    return dist_matrix



"""
Create array for epsilon. Values in the array decrease to 1.
"""
def createEpsilonArray(epsilon_max, N):
    
    epsilons = np.linspace(epsilon_max, 1, N)

    return epsilons


"""
Create array for alpha. Values in the array decrease from 
alpha_max according to the equation in the homework sheet.
"""
def createAlphaArray(alpha_max, lambda_, N):
    # TODO replace statement below with function body
    alphas = np.zeros(N)

    for i in range(N):
        alphas[i] = alpha_max - alpha_max*lambda_*i + 0.5*(alpha_max** 2)*(lambda_**2)*(i**2)
    
    return alphas
    


"""
X is whole data set, K is number of buttons to choose
"""
def initButtons(X, K):

    samples  = np.random.choice(X.shape[0], K, replace = False)

    # return X[samples, :]
    return np.take(X, samples, 0)


"""
x is one data point, buttons is the grid in FEATURE SPACE
"""
def findNearestButtonIndex(x, buttons):
    min_dist = np.inf
    index = 0

    for i in range(len(buttons)):
        dist = getFeatureDist(x, buttons[i])
        if dist < min_dist:
            min_dist = dist
            index = i
    
    return index



"""
Find all buttons within a neighborhood of epsilon of index IN GRID SPACE 
(return a boolean vector)
"""
def findButtonsInNhd(index, epsilon, buttonDist):
    
    distances_to_button = buttonDist[index]
    in_nhd = []
    for i in range(len(distances_to_button)):
        if distances_to_button[i]<epsilon:
            in_nhd.append(True)
        else:
            in_nhd.append(False)
    
    return in_nhd
    # distances = buttonDist[index]
    # buttons = [i for i in range(len(distances)) if distances[i] < epsilon]
    # return buttons      

"""
Do gradient descent step, update each button position IN FEATURE SPACE
"""
def updateButtonPosition(button, x, alpha):
    updated_button = button + alpha*(x-button)

    return updated_button


"""
Compute the squared distance between data points and their nearest button
"""
def computeError(data, buttons):
    
    error = 0

    for i in range(data.shape[0]):
        nearest_button = findNearestButtonIndex(data[i,:], buttons)
        error += np.power(getFeatureDist(data[i,:], buttons[nearest_button,:]),2)

    return error



"""
Function to plot the errors obtained using the computeError function on the self organising maps. 
"""
def plotErrors (errors, fileName):

    iteration_no = np.arange(1, len(errors)+1)
    fig = plt.figure(figsize=(8, 8))
    plt.plot(iteration_no, errors)
    plt.xlabel("Iteration number")
    plt.ylabel("Reconstruction error")
    plt.title("Errors across iterations of the algorithm")
    plt.savefig(fileName)

    

"""
Implementation of the self-organizing map (SOM)

Parameters
 X : data, rows are samples and columns are features
 p, q : dimensions of the grid
 N : number of iterations
 alpha_max : upper limit for learning rate
 epsilon_max : upper limit for radius
 compute_error : boolean flag to determine if the error is computed.
                 The computation of the error is time-consuming and may
                 not be necessary every time the function is called.
 lambda_ : decay constant for learning rate
                 
Returns
 buttons, grid : the buttons and grid of the newly created SOM
 error : a vector with error values. This vector will contain zeros if 
         compute_error is False

TODO: Complete the missing parts in this function following the pseudocode
      in the homework sheet
"""
def SOM(X, p, q, N, alpha_max, epsilon_max, compute_error=False, lambda_=0.01):
    # 1. Create grid and compute pairwise distances
    grid = createGrid(p, q)
    gridDistMatrix = createGridDistMatrix(grid)
    
    # 2. Randomly select K out of d data points as initial positions
    #    of the buttons
    K = p * q
    d = X.shape[0]
    buttons = initButtons(X, K)
    
    # 3. Create a vector of size N for learning rate alpha
    alphas = createAlphaArray(alpha_max, lambda_, N)
    
    # 4. Create a vector of size N for epsilon
    epsilons = createEpsilonArray(epsilon_max, N)
    
    # Initialize a vector with N zeros for the error
    # This vector may be returned empty if compute_error is False
    error = np.zeros(N)

    # 5. Iterate N times
    for i in range(N):
        # 6. Initialize/update alpha and epsilon
        alpha = alphas[i]
        epsilon = epsilons[i]
        
        # 7. Choose a random index t in {1, 2, ..., d}
        t = np.random.choice(a= X.shape[0], size = 1)
        
        # 8. Find button m_star that is nearest to x_t in F 
        x_t = X[t,:]
        index = findNearestButtonIndex(x_t, buttons)

        # 9. Find all grid points in epsilon-nhd of m_star in GRID SPACE
        eps_nhd = findButtonsInNhd(index, epsilon, gridDistMatrix)

        # 10. Update position (in FEATURE SPACE) of all buttons m_j
        #     in epsilon-nhd of m_star, including m_star
        buttons[eps_nhd] = updateButtonPosition(buttons[eps_nhd], x_t, alpha)

        # Compute the error 
        # Note: The computation takes place only if compute_error is True
        #       Replace the statement below with your code
        if compute_error:
            error[i] = computeError(X, buttons)

    # 11. Return buttons, grid and error
    return (buttons, grid, error)