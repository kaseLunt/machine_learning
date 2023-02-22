# Loads necessary packages
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor

# Seed random number generator
rng = np.random.RandomState(4)

# Loads the taxicabsNY.csv dataset
taxicabsNY = pd.read_csv('taxicabsNY.csv')

# Loads predictor and target variables
X = taxicabsNY[['tip','total']].to_numpy() # converted to numpy type array
y = taxicabsNY[['distance']]

# Splits the data into training and test sets
XTrain, XTest, yTrain, yTest = train_test_split(X, np.ravel(y),random_state=rng)

# Initializes and trains a multilayer perceptron regressor model on the training and validation sets

multLayerPercModelTrain = MLPRegressor(random_state=1, max_iter=500000, hidden_layer_sizes=[1]).fit(XTrain, np.ravel(yTrain))
multLayerPercModelValidation = MLPRegressor(random_state=1, max_iter=500000, hidden_layer_sizes=[1]).fit(XTest, np.ravel(yTest))

# Predicts the distance of a taxi ride with a specific tip and total cost
print(multLayerPercModelTrain.predict([[20, 51]]))

# Prints the final weights, biases, and losses
weights = multLayerPercModelTrain.coefs_
biases = multLayerPercModelTrain.intercepts_
loss = multLayerPercModelTrain.loss_
print('{}\n{}\n{}'.format(weights, biases, loss))