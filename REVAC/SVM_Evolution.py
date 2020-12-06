import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from sklearn.model_selection import cross_val_score, TimeSeriesSplit
from scipy import stats
from math import sqrt
from numpy import concatenate
from matplotlib import pyplot
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from sklearn.metrics import mean_squared_error
from pandas import Series
import random
from sklearn.model_selection import ParameterGrid
from sklearn.preprocessing import StandardScaler
from mpl_toolkits.axes_grid1 import make_axes_locatable
#import numpy as np

#Initilaize population
def init_poplulation(numberOfParents):
    C_Value = np.empty([numberOfParents, 1])
    gammaValue = np.empty([numberOfParents, 1])
    
    for i in range(numberOfParents):
        #print(i)
        C_Value[i] = round(random.uniform(0, 100),3)
        gammaValue[i] = round(random.uniform(0.01, 1),3)
        
    population = np.concatenate((C_Value, gammaValue), axis= 1)
    #print (population)
    return population

# the fitness score to be minimized
def root_mean_square_error(y_true, y_pred):
        
    score=np.sqrt(((y_true - y_pred) ** 2).mean())
    score=round(score,3)
    return score

#Perform train-test split with respect to time series structure.

def timeseries_split(X, y, test_size):
    test_index = int(len(X) * (1 - test_size))
    X_train = X[:test_index]
    X_test = X[test_index:]
    y_train = y[:test_index]
    y_test = y[test_index:]
    
    return X_train, X_test, y_train, y_test


#train the data annd find fitness score

def Calculate_Fitness(population, X, y, test_size=0.3):
    
    X_train, X_test, y_train, y_test = timeseries_split(X, y, test_size=0.3)
    sc_X = StandardScaler()
    X_train = sc_X.fit_transform(X_train)
    X_test = sc_X.transform(X_test)
    y_train = y_train.reshape((len(y_train), 1)) 
    y_train = sc_X.fit_transform(y_train)
    y_train = y_train.ravel()
    
    Score = [] 
    
    for i in range(population.shape[0]):
        params = { 'kernel':'rbf',
              'C': population[i][0],
              'gamma': population[i][1] }
        
        model=SVR(**params)
        model.fit(X_train, y_train)
        prediction = model.predict(X_test)
        prediction = sc_X.inverse_transform(prediction) 
    
        Score.append(root_mean_square_error(prediction, y_test))
    return Score

#select n number of parents for mating
def new_parents_selection(population, fitness, numParents):
    #fitness=fitnessvalue array
    #population=population array
    bestparentsIndex=np.argpartition(fitness,numParents)[:numParents]
    bestparents=population[bestparentsIndex]
    return bestparents

# Perform uniform crossover on parents to create n children
def crossover_uniform(parents, numberOfParameters, NumChild):
    
    child=np.empty([NumChild,numberOfParameters])
    Dim0Index=parents.shape[0]-1
    rows=child.shape[0]
    cols=child.shape[1]
    for x in range (rows):
         for y in range(0, cols):
            child[x][y]= parents[(random.randint(0,Dim0Index))][y]
        
    return child

# Single Mutation changes a single gene in each offspring randomly.
def Single_mutation(crossover, numberOfParameters):
    
    #choose a parameter to be mutated
    parameterSelect = np.random.randint(0, numberOfParameters, 1)
    
    #define the possible values for each parameter and #randomly select a value for each parameter
    P1=round(np.random.uniform(0.01, 100),3)
    P2=round(np.random.uniform(0.001, 0.1),3)
    
    print("_________Here are the new PSSSSSS_____________", parameterSelect)
    print(P1,P2)
    
    if parameterSelect == 0: #min/max C Value
        crossover[0][0] = P1 #Parameter1
        
    if parameterSelect == 1: #min/max Gamma Value
        crossover[0][1] = P2 #Parameter2

    print("-------------------This is the new child to replace parent----------------")
    print(crossover)
    
    return crossover



# Mutation changes every gene in each offspring randomly.

def mutation(crossover):
    #Define minimum and maximum values allowed for each parameter
    
   # print (crossover)
    
    P1=round(np.random.uniform(0.01, 100),3)
    P2=round(np.random.uniform(0.001, 0.1),3)
               
    crossover[0][0] = P1 #Parameter 1:min/max C Value
    crossover[0][1] = P2 #Parameter 2:min/max Gamma Value

    return crossover


def parents_replacement(population, fitness, numParents, Children):#creates new population
    WorstparentsIndex=np.argpartition(fitness,-numParents)[-numParents:] #returns an arry of index
    print("The worst parent index for this iteration is :")
    print(WorstparentsIndex)
    population[WorstparentsIndex]= Children
    #print(fitness[WorstparentsIndex])
    return population, WorstparentsIndex


def UpdateFitnessValues(population, childIndex, fitnessValue, X, y, test_size=0.3):

    Newchild=population[childIndex]
    
    X_train, X_test, y_train, y_test = timeseries_split(X, y, test_size=0.3)
    sc_X = StandardScaler()
    X_train = sc_X.fit_transform(X_train)
    X_test = sc_X.transform(X_test)
    y_train = y_train.reshape((len(y_train), 1)) 
    y_train = sc_X.fit_transform(y_train)
    y_train = y_train.ravel()
    
    Score = [] 
    
    for i in range(Newchild.shape[0]):
        
        params = { 'kernel':'rbf',
              'C': Newchild[i][0],
              'gamma': Newchild[i][1] }
        
        model=SVR(**params)
        model.fit(X_train, y_train)
        prediction = model.predict(X_test)
        prediction = sc_X.inverse_transform(prediction) 
    
        Score.append(root_mean_square_error(prediction, y_test))
        

        fitnessValue[childIndex[0]]=Score[0]
        
        print('this is the child rmse', fitnessValue[childIndex[0]])
        
    return fitnessValue

'''
This function will allow us to genrate the heatmap for various parameters and fitness to visualize 
how each parameter and fitness changes with each generation
'''

def plot_parameters(numberOfGenerations, numberOfParents, parameter, parameterName):
    #inspired from https://matplotlib.org/gallery/images_contours_and_fields/image_annotated_heatmap.html
    generationList = ["Gen {}".format(i) for i in range(numberOfGenerations+1)]
    populationList = ["Parent {}".format(i) for i in range(numberOfParents)]
    
    fig, ax = plt.subplots(figsize=(50,50)) #15,20 looks great formely 10,15
    im = ax.imshow(parameter, cmap=plt.get_cmap('YlGn'))
    
    # show ticks
    ax.set_xticks(np.arange(len(populationList)))
    ax.set_yticks(np.arange(len(generationList)))
    
    # show labels
    ax.set_xticklabels(populationList, size='30')
    ax.set_yticklabels(generationList, size='30')
    
    # set ticks at 45 degrees and rotate around anchor
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")
    
    
    # insert the value of the parameter in each cell
    #for i in range(len(generationList)):
    #    for j in range(len(populationList)):
   #         text = ax.text(j, i, parameter[i, j],
    #                       ha="center", va="center", color="k")
    
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="2.5%", pad=0.5)
   

    cbar=plt.colorbar(im, cax=cax)
    cbar.ax.tick_params(labelsize=30)
    cbar.set_label(parameterName, rotation=90, size='30')
    
    ax.set_title("Change in the value of " + parameterName, size='30')
    fig.tight_layout()
    plt.show()


#returns best parameter in a given population
def Best_Utility(population, fitness):
    #fitness=fitnessvalue array
    #population=population array
    bestIndex=np.argpartition(fitness,1)[:1]
    print(bestIndex)
    bestparameter=population[bestIndex]
    bestfitness=fitness[bestIndex]
    print(bestparameter)
    return bestfitness