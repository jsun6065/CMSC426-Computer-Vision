import numpy as np
import time
import matplotlib.pyplot as plt



train_images_np=np.load('./Project3_Data/MNIST_train_images.npy')
train_labels_np=np.load('./Project3_Data/MNIST_train_labels.npy')
val_images_np=np.load('./Project3_Data/MNIST_val_images.npy')
val_labels_np=np.load('./Project3_Data/MNIST_val_labels.npy')
test_images_np=np.load('./Project3_Data/MNIST_test_images.npy')
test_labels_np=np.load('./Project3_Data/MNIST_test_labels.npy')


##Template MLP code
def softmax(x):
    return np.exp(x)/np.sum(np.exp(x))

def sigmoid(x):
    return 1/(1+np.exp(-x))

def CrossEntropy(y_hat,y):
    return -np.dot(y,np.log(y_hat))

class MLP():

    def __init__(self):
        #Initialize all the parametres
        #Uncomment and complete the following lines
        # self.W1=
        # self.b1=
        # self.W2=
        # self.b2=
        self.reset_grad()

    def reset_grad(self):
        self.W2_grad = 0
        self.b2_grad = 0
        self.W1_grad = 0
        self.b1_grad = 0

    def forward(self, x):
        #Feed data through the network
        #Uncomment and complete the following lines
        # self.x=x
        # self.W1x=
        # self.a1=
        # self.f1=
        # self.W2x=
        # self.a2=
        # self.y_hat=
        return self.y_hat

    def update_grad(self,y):
        # Compute the gradients for the current observation y and add it to the gradient estimate over the entire batch
        # Uncomment and complete the following lines
        # dA2db2=
        # dA2dW2=
        # dA2dF1=
        # dF1dA1=
        # dA1db1=
        # dA1dW1=
        # dLdA2 =
        # dLdW2 =
        # dLdb2 =
        # dLdF1 =
        # dLdA1 =
        # dLdW1 =
        # dLdb1 =
        # self.W2_grad = self.W2_grad + dLdW2
        # self.b2_grad = self.b2_grad + dLdb2
        # self.W1_grad = self.W1_grad + dLdW1
        # self.b1_grad = self.b1_grad + dLdb1
        pass

    def update_params(self,learning_rate):
        self.W2 = self.W2 - learning_rate * self.W2_grad
        self.b2 = self.b2 - learning_rate * self.b2_grad.reshape(-1)
        self.W1 = self.W1 - learning_rate * self.W1_grad
        self.b1 = self.b1 - learning_rate * self.b1_grad.reshape(-1)

## Init the MLP
myNet=MLP()


learning_rate=1e-3
n_epochs=100

## Training code
for iter in range(n_epochs):
    #Code to train network goes here
    pass
    #Code to compute validation loss/accuracy goes here


## Template for ConvNet Code
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class ConvNet(nn.Module):
    #From https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x.view(-1,1,28,28))))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

#Your training and testing code goes here