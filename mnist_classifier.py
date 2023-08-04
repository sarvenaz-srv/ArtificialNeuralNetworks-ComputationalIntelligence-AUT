from layers.fullyconnected import FC
from layers.convolution2d import Conv2D
from layers.maxpooling2d import MaxPool2D
from activations import Activation, get_activation
from optimizers.gradientdescent import GD
from optimizers.adam import Adam
from losses.binarycrossentropy import BinaryCrossEntropy
from model import Model
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from PIL import Image
from random import sample
import os
DIR = "./datasets/MNIST/"
DIR2 = DIR + "2/"
DIR5 = DIR + "5/"
FILES2 = [DIR2 + f for f in os.listdir(DIR2)]
FILES5 = [DIR5 + f for f in os.listdir(DIR5)]


def load_data(path, label):
    return  [np.expand_dims(np.array(Image.open(path)) / 255., axis=-1), label]

data = []
for f in FILES2:
    data.append(load_data(f, 0))
for f in FILES5:
    data.append(load_data(f, 1))

def shuffle(m, shuffling):
        order = list(range(m))
        if shuffling:
            return sample(order, m)
        return order


Batch_Size = 47
N = len(data)
TOT_STEP = N // Batch_Size

def prepare_data():
    X, y = [], []
    for xy in data:
        X.append(xy[0])
        y.append(xy[1])
    return np.array(X), np.array(y).reshape(1, -1)


arch_model = {
    "CONV1": Conv2D(1, 2, name="CONV1", kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
    "RELU1": get_activation("relu")(),
    "MAXPOOL1":MaxPool2D(kernel_size=(2, 2), stride=(2, 2)),
    "RELU2": get_activation("relu")(),
    "CONV2": Conv2D(2, 4, name="CONV2", kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
    "RELU3": get_activation("relu")(),
    "MAXPOOL2":MaxPool2D(kernel_size=(2, 2), stride=(2, 2)),
    "RELU4": get_activation("relu")(),
    "FC1": FC(49*4, 16, "FC1"),
    "SIGMOMID1": get_activation("sigmoid")(),
    "FC2": FC(16, 1, "FC2"),
    "SIGMOMID2": get_activation("sigmoid")(),
}

criterion = BinaryCrossEntropy()
optimizer = GD(arch_model, learning_rate=0.01)
# optimizer = Adam(layers_list = {key: arch_model[key] for key in ["CONV1", "CONV2", "FC1", "FC2"]}, learning_rate=0.01)
myModel = Model(arch_model, criterion, optimizer)

costs = []
X_train, Y_train = prepare_data()
costs, val = myModel.train(X_train,Y_train,epochs=25,val=None,batch_size=Batch_Size,shuffling=True,verbose=1,save_after='mnist_classifier')

plt.plot(costs)
plt.show()

order = shuffle(2000, True)[0:4]
X_test = X_train[order]
Y_test = Y_train[:,order]
Y = myModel.predict(X_test)
fig, ax = plt.subplots(2, 2, figsize=(8, 8))
ax[0, 0].imshow(X_test[0].reshape(28, 28), cmap="gray")
ax[0, 0].set_title(f"{Y[:, 0]} vs {Y_test[:, 0]}")
ax[0, 1].imshow(X_test[1].reshape(28, 28), cmap="gray")
ax[0, 1].set_title(f"{Y[:, 1]} vs {Y_test[:, 1]}")
ax[1, 0].imshow(X_test[2].reshape(28, 28), cmap="gray")
ax[1, 0].set_title(f"{Y[:, 2]} vs {Y_test[:, 2]}")
ax[1, 1].imshow(X_test[3].reshape(28, 28), cmap="gray")
ax[1, 1].set_title(f"{Y[:, 3]} vs {Y_test[:, 3]}")
plt.show()