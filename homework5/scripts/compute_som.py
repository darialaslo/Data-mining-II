


#loading the required libraries 
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from somutils import plotDataAndSOM
import somutils

#defining the inline arguments to be passed 

parser = argparse.ArgumentParser(description="Run exercises")
parser.add_argument("--exercise", required=True, 
                    help="Which exercise of the homework to solve.")

parser.add_argument("--outdir", required=False,
                    default='./results',
                    help="Where to create the new directory for the model.")

parser.add_argument("--p", required=False,
                    default=99,
                    help="The number of rows in the grid.")

parser.add_argument("--q", required=False,
                    default=99,
                    help="The number of columns in the grid.")

parser.add_argument("--N", required=False,
                    default=99,
                    help="The number of iterations.")

parser.add_argument("--alpha_max", required=False,
                    default=99,
                    help="Upper limit for learning rate.")
parser.add_argument("--epsilon_max", required=False,
                    default=99,
                    help="Upper limit for radius.")
parser.add_argument("--lamb", required=False,
                    default=0.99,
                    help="Decay constant for learning rate decay.")
                    
parser.add_argument("--file", required=True, 
                    default = '../crabs.csv',
                    help="Which file should be used as data source.")
args = parser.parse_args()




##################### MAIN PROGRAM ##############################

if os.path.isdir(args.outdir):
        print("Directory already exists. Files will be overwritten.")
else:
    os.makedirs(args.outdir)


np.random.seed(1)
#setting params to int 
p = int(args.p)
q = int(args.q)
N = int(args.N)
alpha_max = float(args.alpha_max)
epsilon_max = float(args.epsilon_max)
lamb = float(args.lamb)


################ exrcise 1 ###################
if int(args.exercise) == 1:

    print("Solving exercise 1.")
    #exercise 1.b.
    #create data
    X = somutils.makeSCurve()
    #create self organizing maps
    buttons, grid, error = somutils.SOM(X, p, q, N, alpha_max, epsilon_max, True, lamb )

    #create filename
    filename = "{}/exercise_1b.pdf".format(args.outdir)
    #plot the obtained data
    somutils.plotDataAndSOM(X, buttons, filename)


    #exercise 1.c.
    filename_1c = "{}/exercise_1c.pdf".format(args.outdir)
    #plot the errors
    somutils.plotErrors(error, filename_1c)



################ exrcise 2 ###################


if int(args.exercise) == 2:
    print("Solving exercise 2.")


    #load data
    crabs = pd.read_csv(args.file)

    #exercise 2.a.
    X = (crabs.iloc[:,-5:]).to_numpy()


    labels = []
    buttons, grid, error = somutils.SOM(X, p, q, N, alpha_max, epsilon_max,  False, lamb)
    file_out = "{}/output_som_crabs.txt".format(args.outdir)

    with open(file_out, 'w') as f_out:
        #write first row
        f_out.write("sp\tsex\tindex\tlabel\n")

        for i in range(X.shape[0]):

            x_i = X[i, :]
            closest_button  = somutils.findNearestButtonIndex(x_i, buttons)
            f_out.write('{}\t{}\t{}\t{}\n'.format(
                crabs.iloc[i,0],
                crabs.iloc[i,1],
                crabs.iloc[i,2],
                int(closest_button)
            ))
    # f_out.close()
    
    #exercise 2.b.
    idInfo = (crabs.iloc[:, :2]).to_numpy()
    filename = '{}/exercise_2b.pdf'.format(args.outdir)
    somutils.plotSOMCrabs(X, p, q, idInfo, grid, buttons, filename)



















