import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

from layers.fullyconnected import FC
from losses.meansquarederror import MeanSquaredError
from model import Model
from activations import Activation, get_activation
from optimizers.gradientdescent import GD
from optimizers.adam import Adam

TRAIN_PATH = './datasets/california_houses_price/california_housing_train.csv'
TEST_PATH = './datasets/california_houses_price/california_housing_test.csv'

# read files
training_data = pd.read_csv(TRAIN_PATH)
test_data = pd.read_csv(TEST_PATH)
# create train and test datasets
X_train = np.array(training_data[training_data.columns[0:-1]]).T
Y_train = np.array(training_data[training_data.columns[-1]]).reshape(1,-1)
X_test = np.array(test_data[test_data.columns[0:-1]]).T
Y_test = np.array(test_data[test_data.columns[-1]]).reshape(1,-1)

#normalize
X_train_normalized = np.array([(x-np.mean(x))/(np.std(x)) for x in X_train])
Y_train_normalized = np.array([(y-np.mean(y))/(np.std(y)) for y in Y_train])
X_test_normalized = np.array([(x-np.mean(x))/(np.std(x)) for x in X_test])
Y_test_normalized = np.array([(y-np.mean(y))/(np.std(y)) for y in Y_test])

arch_model = {
    "FC1": FC(8, 12, "FC1"),
    "LINEAR1": get_activation("relu")(),
    "FC2": FC(12, 4, "FC2"),
    "LINEAR2": get_activation("relu")(),
    "FC3": FC(4, 1, "FC3"),
    "LINEAR3": get_activation("linear")(),
}

criterion = MeanSquaredError()
optimizer = Adam(layers_list = {key: arch_model[key] for key in ["FC1", "FC2", "FC3"]}, learning_rate=0.01)

myModel = Model(arch_model, criterion, optimizer)

costs = []
costs, val = myModel.train(X_train_normalized,Y_train_normalized,epochs=500,val=None,batch_size=32,shuffling=True,verbose=1,save_after='price_prediction')

plt.plot(costs)
plt.show()

Y = myModel.predict(X_test_normalized)
fig, ax = plt.subplots(1,2,figsize=(16,16))
ax[0].plot(Y_test_normalized.T)
ax[0].set_title('true')
ax[1].plot(Y.T)
ax[1].set_title('predict')
plt.show()