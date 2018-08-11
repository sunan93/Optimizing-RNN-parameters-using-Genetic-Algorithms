# Optimizing LSTM model parameters using Genetic Algorithms

### Introduction:-
The aim of the project is to optimize the parameters of a recurrent neural network to obtain the best configuration of parameters. Genetic algorithm has been used to fine tune parameters used to train an RNN for wind power forecasting. The best number of LSTM units and ther optimal window size to be used for prediction have been found. For deep learning, Keras library is used and for Genetic Algorithms, DEAP library has been used. 

The following paragraphs explain the application of genetic algorithms to this project.

### Theory behind Genetic Algoithms:-
Genetic algorithm is an adaptive heuristic search algorithm based on the ideas of evolution. It leads to an exploration as well as exploitation of the search space. It exploits the past information to get better and better subsequently. The techniques are inspired from the principles of Charles Darwin of "survival of the fittest". In nature also, we observe that the weaker individuals get dominated by the stronger ones. 

The solution representation for the parameters used in this project are as follows:-
A solution to our problem is a 10 bit integer where the first six bits represent the window size and the next four bits represent the number of units of LSTM.

The basic operators that consttitue a Genetic Algorithm are as follows:-

#### 1. Selection Operator:-
The process of selection gives more preference to better individuals. The fitness of each individual solution is calculated using a fitness function. In this implementation, roulette wheel selection has been used where a wheel is divided into n pies for n individuals and each individual gets a portion of the circle which is proportional to its fitness value.

#### 2. Crossover Operator:-
Two individuals are chosen from a the population and a crossover site is chosen. Then these two values are exhanged accross the crossover site to get new solutions. The two new solutions created are passed onto the next generation. In this implementation, ordered cross-over has been used. Here, we select two random crossover points and copy the contents of first parent into the offspring. Then, starting from the second crossover point, copy the unused numbers of the second parent into the offsrping.

#### 3. Mutation Operator:-
The purpose of mutation is to maintain diversity in the population and also inhibit premature convergence. It uses the idea of random walk in the search space. In this implementation, shuffle mutation has been used where the attributes of the solution are shuffeled randomly to get a new solution.

### RNN Implementation:-
The wind power forecasting data contains the wind power measurements of seven wind farms. But only, column 'wp1' has been used for experimentation.
A basic LSTM cell in keras is used to create a chain of LSTM cells. The root mean square error on the validation set has been calculated and returned as a fitness score to the genetic algorithm solution. 

### Results:-
The optimal window size has been found to be 47 and the optimal number of LSTM units is 9.

The outputs of 5 epochs for training are shown below:-
Epoch 1/5

17209/17209 [==============================] - 21s 1ms/step - loss: 0.0190

Epoch 2/5

17209/17209 [==============================] - 18s 1ms/step - loss: 0.0078

Epoch 3/5

17209/17209 [==============================] - 18s 1ms/step - loss: 0.0062

Epoch 4/5

17209/17209 [==============================] - 19s 1ms/step - loss: 0.0060

Epoch 5/5

17209/17209 [==============================] - 19s 1ms/step - loss: 0.0060

Test RMSE:  0.09710418381525192

### References:-
1. This project is an implementation of this [blog](http://aqibsaeed.github.io/2017-08-11-genetic-algorithm-for-optimizing-rnn/). The code has been referred from here and the dataset can be downloaded from [here](https://www.kaggle.com/c/GEF2012-wind-forecasting/data).
2. DEAP Package[deap](http://deap.readthedocs.io/en/master/api/tools.html#deap.tools.cxOrdered)
3. Tutorial on [Genetic AlgorithmsGA](https://www.tutorialspoint.com/genetic_algorithms/genetic_algorithms_mutation.htm)
4. Blog on [GA](https://www.doc.ic.ac.uk/~nd/surprise_96/journal/vol1/hmw/article1.html)
