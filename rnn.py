import numpy as np
import pandas as pd

from deap import base, creator, tools, algorithms
from scipy.stats import bernoulli
from bitstring import BitArray

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split as split

from keras.layers import LSTM, Input, Dense
from keras.models import Model

np.random.seed(1120)

data = pd.read_csv('/home/sunandini/Project_codes/GeneticAlgo/all/train.csv')
data = np.reshape(np.array(data['wp1']),(len(data['wp1']),1))
print(data[:10])
train_data = data[0:18000]
test_data = data[18000:]

def prepare_dataset(data, window_size):
    X, Y = np.empty((0,window_size)), np.empty((0))
    for i in range(len(data)-window_size-1):
        X = np.vstack([X,data[i:(i + window_size),0]])
        Y = np.append(Y,data[i + window_size,0])   
    X = np.reshape(X,(len(X),window_size,1))
    Y = np.reshape(Y,(len(Y),1))
    return X, Y

X_train,y_train = prepare_dataset(train_data,3)
print(X_train)
print(y_train)


def train_evaluate(ga_individual_solution):   
    # Decode GA solution to integer for window_size and num_units
    window_size_bits = BitArray(ga_individual_solution[0:6])
    num_units_bits = BitArray(ga_individual_solution[6:]) 
    window_size = window_size_bits.uint
    num_units = num_units_bits.uint
    print('\nWindow Size: ', window_size, ', Num of Units: ', num_units)
    
    
    # Segment the train_data based on new window_size; split into train and validation (80/20)
    X,Y = prepare_dataset(train_data,window_size)
    X_train, X_val, y_train, y_val = split(X, Y, test_size = 0.20, random_state = 1120)
    
    # Train LSTM model and predict on validation set
    inputs = Input(shape=(window_size,1))
    x = LSTM(num_units, input_shape=(window_size,1))(inputs)
    predictions = Dense(1, activation='linear')(x)
    model = Model(inputs=inputs, outputs=predictions)
    model.compile(optimizer='adam',loss='mean_squared_error')
    model.fit(X_train, y_train, epochs=5, batch_size=10,shuffle=True)
    y_pred = model.predict(X_val)
    
    # Calculate the RMSE score as fitness score for GA
    rmse = np.sqrt(mean_squared_error(y_val, y_pred))
    print('Validation RMSE: ', rmse,'\n')
    
    return rmse,


def train_GA():
    population_size = 5
    num_generations = 5
    gene_length = 10

    # As we are trying to minimize the RMSE score, that's why using -1.0. 
    # In case, when you want to maximize accuracy for instance, use 1.0
    creator.create('FitnessMax', base.Fitness, weights = (-1.0,))
    creator.create('Individual', list , fitness = creator.FitnessMax)

    toolbox = base.Toolbox()
    toolbox.register('binary', bernoulli.rvs, 0.5)
    toolbox.register('individual', tools.initRepeat, creator.Individual, toolbox.binary, n = gene_length)
    toolbox.register('population', tools.initRepeat, list , toolbox.individual)

    toolbox.register('mate', tools.cxOrdered)
    toolbox.register('mutate', tools.mutShuffleIndexes, indpb = 0.6)
    toolbox.register('select', tools.selRoulette)
    toolbox.register('evaluate', train_evaluate)

    population = toolbox.population(n = population_size)
    r = algorithms.eaSimple(population, toolbox, cxpb = 0.4, mutpb = 0.1, ngen = num_generations, verbose = False)
    return population


population = train_GA()
best_individuals = tools.selBest(population,k = 1)
best_window_size = None
best_num_units = None

for bi in best_individuals:
    window_size_bits = BitArray(bi[0:6])
    num_units_bits = BitArray(bi[6:]) 
    best_window_size = window_size_bits.uint
    best_num_units = num_units_bits.uint
    print('\nWindow Size: ', best_window_size, ', Num of Units: ', best_num_units)


#train model using best values obtained

def train_best(best_window_size, best_num_units):
    X_train,y_train = prepare_dataset(train_data,best_window_size)
    X_test, y_test = prepare_dataset(test_data,best_window_size)

    inputs = Input(shape=(best_window_size,1))
    x = LSTM(best_num_units, input_shape=(best_window_size,1))(inputs)
    predictions = Dense(1, activation='linear')(x)
    model = Model(inputs = inputs, outputs = predictions)
    model.compile(optimizer='adam',loss='mean_squared_error')
    model.fit(X_train, y_train, epochs=5, batch_size=10,shuffle=True)
    y_pred = model.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print('Test RMSE: ', rmse)

train_best(best_window_size, best_num_units)


