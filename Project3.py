import numpy as np
import time
import matplotlib.pyplot as plt
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


##Template MLP code
def softmax(x):
    return np.exp(x)/np.sum(np.exp(x))

def sigmoid(x):
    return 1/(1+np.exp(-x))

def CrossEntropy(y_hat,y):
    return -np.dot(y,np.log(y_hat))


class MLP():

    def __init__(self):
        #2.2 Initialize all the parametres
        #Uncomment and complete the following lines
        self.W1=np.random.normal(size=(784,64),scale=0.1)# 784x64
        self.b1=np.zeros((1,64))
        self.W2=np.random.normal(size=(64,10),scale=0.1)# 64x10
        self.b2=np.zeros((1,10))
        self.reset_grad()

    def reset_grad(self): 
        self.W2_grad = 0
        self.b2_grad = 0
        self.W1_grad = 0
        self.b1_grad = 0

    def forward(self, x):
        #2.1 Feed data through the network
        #Uncomment and complete the following lines
        self.x=np.matrix(x)
        self.W1x=np.matmul(self.x,self.W1)
        self.a1=self.W1x + self.b1
        self.f1=sigmoid(self.a1)
        self.W2x=np.matmul(self.f1,self.W2)      
        self.a2=self.W2x + self.b2
        self.y_hat=softmax(self.a2)
        self.y_hat = np.array(self.y_hat)
        return self.y_hat

    def update_grad(self,y): 
        # 2.3 Compute the gradients for the current observation y and add it to the gradient estimate over the entire batch
        # Uncomment and complete the following lines
        dA2db2=1
        dA2dW2=self.f1
        dA2dF1=self.W2
        dF1dA1=np.multiply(sigmoid(self.a1),(1-sigmoid(self.a1)))
        dA1db1=1
        dA1dW1=self.x
        dLdA2 = self.y_hat-y
        dLdW2 = np.matmul(dLdA2.reshape(-1,1),dA2dW2)
        dLdb2 = dLdA2 
        dLdF1 = np.matmul(dLdA2,self.W2.T)
        dLdA1 = np.multiply(np.matmul(dLdA2,dA2dF1.T),dF1dA1)
        dLdW1 = np.matmul(np.multiply(np.matmul(dLdA2,dA2dF1.T),dF1dA1).T,dA1dW1)
        dLdb1 = np.multiply(np.matmul(dLdA2,dA2dF1.T),dF1dA1)
        self.W2_grad = self.W2_grad + dLdW2
        self.b2_grad = self.b2_grad + dLdb2
        self.W1_grad = self.W1_grad + dLdW1
        self.b1_grad = self.b1_grad + dLdb1
        pass

    def update_params(self,learning_rate):
        self.W2 = self.W2 - learning_rate * self.W2_grad.T
        self.b2 = self.b2 - learning_rate * self.b2_grad.reshape(-1)
        self.W1 = self.W1 - learning_rate * self.W1_grad.T
        self.b1 = self.b1 - learning_rate * self.b1_grad.reshape(-1)

        
##2.4 Training code SGD
def SGD(myNet,X,y,learning_rate,n_epochs, batch_size,n_images,val_images_np,val_labels_np):
    accuracies = []
    n_batches = int(np.ceil(n_images/256))
    X_val = X
    y_val = y
    v_acc = []
    tr_acc = []
    for iter in range(n_epochs):
        #Code to train network goes here
        #form minibatches 
        start = 0
        success = 0
        for ind in range(n_batches): # iterate for 256*n number of images
            indexes = np.arange(50000)
            np.random.shuffle(indexes)
            if ind == n_batches-1:
                end = end + (n_images%batch_size)
            else:               
                end = (ind + 1) * batch_size
            myNet.reset_grad()
            for i in range(start,end): # iterate in groups of batch size 256
                Y_hat = myNet.forward(X[indexes[i]])
                #create answer array
                Y = np.zeros(10)
                Y[y[indexes[i]]] = 1
                loss = CrossEntropy(Y_hat.T,Y)
                myNet.update_grad(Y)
                # check training set accuracy
                if (np.argmax(Y_hat) == np.argmax(Y)):
                    success += 1
            myNet.update_params(learning_rate)
            start = end
        #run test on validation set
        print(iter)
        v_acc.append(set_accuracy(myNet,val_images_np,val_labels_np,len(val_images_np)))
        tr_acc.append(success/n_images)
    return v_acc,tr_acc        
        
#2.8 plot 64 images of W1
def visualize_W1():
    a = np.load("W1.npy")
    a = a.T
    width=5
    height=5
    rows = 8
    cols = 8
    axes=[]
    fig=plt.figure(figsize = (10,10))
    for i in range(rows*cols):
        b = np.reshape(a[i],(28,28))
        axes.append( fig.add_subplot(rows, cols, i+1) )
        plt.imshow(b,cmap="gray")
    plt.show()
    
# compute training and validation set
def set_accuracy(myNet,X,Y,size):
    X_val = X
    y_val = Y
    success = 0
    for i in range(size):
        Y_hat = myNet.forward(X_val[i])
        Y = np.zeros(10)
        Y[y_val[i]] = 1
        if (np.argmax(Y_hat) == np.argmax(Y)):
            success += 1
    print(success/size)
    return success/size

# test MLP on testing set
def test_accuracy(myNet,test_images_np,test_labels_np):
    conf_matrix = np.zeros((10,10))
    X_val = test_images_np
    y_val = test_labels_np
    success = 0
    for i in range(5000):
        Y_hat = myNet.forward(X_val[i])
        Y = np.zeros(10)
        Y[y_val[i]] = 1
        if (np.argmax(Y_hat) == np.argmax(Y)):
            success += 1
        conf_matrix[np.argmax(Y_hat),np.argmax(Y)] += 1
    
    return success/5000,conf_matrix

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

# found at https://stackoverflow.com/questions/37511277/how-to-plot-confusion-matrix-with-gridding-without-color-using-matplotlib
def conf_matrix(conf_arr): 
    conf_arr = np.array(conf_arr)
    height, width = conf_arr.shape

    fig = plt.figure('confusion matrix')
    ax = fig.add_subplot(111, aspect='equal')
    for x in range(width):
        for y in range(height):
            ax.annotate(str(int(conf_arr[x][y])), xy=(y, x), ha='center', va='center')

    offset = .5  
    ax.set_xlim(-offset, width - offset)
    ax.set_ylim(-offset, height - offset)

    ax.hlines(y=np.arange(height+50)- offset, xmin=-offset, xmax=width-offset + 40)
    ax.vlines(x=np.arange(width+50) - offset, ymin=-offset, ymax=height-offset + 40)

    alphabet = '0123456789'
    plt.xticks(range(width), alphabet[:width])
    plt.yticks(range(height), alphabet[:height])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()
    
def CNN_set_accuracy(net,size,setloader):
    success = 0
    for i,data in enumerate(setloader):
        image,labels = data
        outputs = net(image)
        result = outputs.detach().numpy()
        labels_np = labels.detach().numpy()
        if (np.argmax(result[0]) == labels_np[0]):
            success += 1 
    print(success/size)
    return success/size


# 3.2 SGD for CNN
def SGD_CNN(net,X,n_images,n_epochs,trainloader,valloader,optimizer,criterion):
    n_batches = int(np.ceil(n_images/256))
    X_val = X
    v_acc = []
    tr_acc = []
    for ite in range(n_epochs):
        #Code to train network goes here
        #form minibatches 
        start = 0
        success = 0
        dataiter = iter(trainloader) # shuffles data
        for ind in range(n_batches): # iterate for 256*n number of images
            optimizer.zero_grad()            
            if ind == n_batches-1:
                end = end + (n_images%256)
            else:               
                end = (ind + 1) * 256
            for i in range(start,end):
                images,labels = dataiter.next()
                # forward + backward + optimize
                outputs = net(images)
                result = outputs.detach().numpy()
                labels_np = labels.detach().numpy()
                # check training set accuracy
                if (np.argmax(result[0]) == labels_np[0]):
                    success += 1 
                loss = criterion(outputs, labels)
                loss.backward()
            optimizer.step()
            start = end
            #run tests on validation set
        print(ite)
        print(success/n_images)
        tr_acc.append(success/n_images)
        v_acc.append(CNN_set_accuracy(net,10000,valloader))
    return tr_acc,v_acc    
    

def main():
    #load data
    train_images_np=np.load('./Project3_Data/MNIST_train_images.npy')
    train_labels_np=np.load('./Project3_Data/MNIST_train_labels.npy')
    val_images_np=np.load('./Project3_Data/MNIST_val_images.npy')
    val_labels_np=np.load('./Project3_Data/MNIST_val_labels.npy')
    test_images_np=np.load('./Project3_Data/MNIST_test_images.npy')
    test_labels_np=np.load('./Project3_Data/MNIST_test_labels.npy')
    
    ## Init the MLP
    myNet=MLP()    
    learning_rate=1e-3
    n_epochs=100
    batch_size = 256
    X = train_images_np
    y = train_labels_np
    
    debug = False
    #2.5 Overfitting
    # plot accuracies after training with 2000 images    
    v_acc,tr_acc = SGD(myNet,X,y,learning_rate,n_epochs,batch_size,2000,val_images_np,val_labels_np)    
    plt.plot(np.arange(100),tr_acc,label="Training Dataset")
    plt.plot(np.arange(100),v_acc,label="Validation Dataset")
    plt.xlabel("Number of Epochs")
    plt.ylabel("Accuracy")
    plt.legend(loc="lower right")
    plt.suptitle("Number of Epochs vs. Accuracy Trained with 2000 Images")
    plt.show()
   
    
    # plot accuracies after training with 50000 images
    V,TR = SGD(myNet,X,y,learning_rate,n_epochs,batch_size,50000,val_images_np,val_labels_np)
    np.save("W1.npy",myNet.W1)
    np.save("b1.npy",myNet.b1)
    np.save("W2.npy",myNet.W2)
    np.save("b2.npy",myNet.b2)    
    plt.plot(np.arange(100),TR,label="Training Dataset")
    plt.plot(np.arange(100),V,label="Validation Dataset")
    plt.xlabel("Number of Epochs")
    plt.ylabel("Accuracy")
    plt.legend(loc="lower right")
    plt.suptitle("Number of Epochs vs. Accuracy Trained with 50000 Images")
    plt.show()

    #2.6 Print Average Test Accuracy
    percent,c_matrix = test_accuracy(myNet,test_images_np,test_labels_np)        
    print("Average Accuracy Across Test Set")
    print(percent)
    
    #2.7 10x10 Confusion Matrix
    conf_matrix(c_matrix)
    
    #2.8 Visualize W1
    visualize_W1()
    
    #3.3 Overfitting
    net = ConvNet()
    learning_rate=1e-4
    n_epochs=20
    batch_size = 256
    transform = transforms.Compose(
            [transforms.ToTensor(),
            transforms.Normalize((0.5), (0.5))])
    X = torchvision.datasets.MNIST(root='./data', train=True,download=True, transform=transform)
    X_tr,X_val = torch.utils.data.random_split(X,[50000,10000])
    X_test = torchvision.datasets.MNIST(root='./data', train=False,download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(X_tr, batch_size=1,
                                                      shuffle=True, num_workers=2)
    valloader = torch.utils.data.DataLoader(X_val, batch_size=1,
                                                      shuffle=True, num_workers=2)
    testloader = torch.utils.data.DataLoader(X_test, batch_size=1,
                                                     shuffle=False, num_workers=2)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9)
    
    # Training and validation accuracy for 2000 images
    training_accuracy, validation_accuracy = SGD_CNN(net,X_tr,2000,20,trainloader,valloader,optimizer,criterion) 
    plt.plot(np.arange(20),training_accuracy,label="Training Dataset")
    plt.plot(np.arange(20),validation_accuracy,label="Validation Dataset")
    plt.xlabel("Number of Epochs")
    plt.ylabel("Accuracy")
    plt.legend(loc="lower right")
    plt.suptitle("Number of Epochs vs. Accuracy Trained with 2000 Images for CNN")
    plt.show()
    # Training and validation accuracy for 50000 images
    training_accuracy, validation_accuracy = SGD_CNN(net,X_tr,50000,20,trainloader,valloader,optimizer,criterion)
    plt.plot(np.arange(20),training_accuracy,label="Training Dataset")
    plt.plot(np.arange(20),validation_accuracy,label="Validation Dataset")
    plt.xlabel("Number of Epochs")
    plt.ylabel("Accuracy")
    plt.legend(loc="lower right")
    plt.suptitle("Number of Epochs vs. Accuracy Trained with 50000 Images for CNN")
    plt.show()
   
    # Average Accuracy on test set
    print("Average Accuracy on Test Set for CNN")
    CNN_set_accuracy(net,len(X_test),testloader)
    torch.save(net,"./CNN_torch")
if __name__ == '__main__':
    main()