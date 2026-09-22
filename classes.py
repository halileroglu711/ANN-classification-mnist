 #import libraries
import torch #pytorch library for tensor 
import torch.nn as nn #needed to define ann layer   
import torch.optim as optim #needed for backpropagation
import torchvision #includes image processing and pre-trained models 
import torchvision.transforms as transforms #needed for image pre-processing
import matplotlib.pyplot as plt #classic data visualization

#in order to build ann,first we need a class which inherits nn.Module.An instance of the NeuralNetwork class will be
#using every command in nn.module.
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        #flatten layer
        self.flatten=nn.Flatten()#flattens our image (28*28) into one straight vector.equals => 28*28=784
        
        #first fully connected layer.takes (input_feature,output_feature) input_feature must match the vector's length and output_feature is 
        #outputs which means neuron count.
        self.fc1=nn.Linear(28*28 ,128)
        
        #after our first fully connected layer,we are gonna have 128 different outputs.now we are gonna define a reLU function to use after the first layer
        
        self.relu=nn.ReLU()#activation function was created.
        
        self.fc2=nn.Linear(128,64)#second fully connected layer gave us 64 different outputs.(this layer contains 64 neurons as you can understand)
        #using another ReLU function is optional after fc2.
        
        #1 input layer (first layer) and 1 hidden layer(second layer) are fair enough for this project.now we need a output layer.(last layer before outputs(predicts)
        self.fc3=nn.Linear(64,10)#!!!-----ATTENTION------!!! this layer has to take last layer's output,its okay we know that but main problem 
        #here is that this layer's output_feature is fixed by the amount of our unique numbers in our dataset (0-9 for MNIST so we need 10 outputs in total)
    
    def forward(self,x): #will be the function that starts forward propagation.so it is going to include forward propagation processes.
        #it takes 2 parameters. self stands for current instance name and x stands for current image.
        #every function constantly returns x in order to continue ann
        
        x=self.flatten(x)# 1-image has been flattened.(728(length)vector.)
        x=self.fc1(x)# 2-our vector's every pixel has connected to every neuron(128)in the first layer and has returned 128 different outputs.
        x=self.relu(x) # 3- it alters first fully connected layers' outputs by replacing negative outputs with 0. Positive outputs stay the same.  
        x=self.fc2(x) # 4-second fully connected layer takes last 128 outputs as inputs and returns 64 outputs.
        x=self.relu(x) #5-relu here is optional.
        x=self.fc3(x) #6- output layer.takes 64 outputs from the second fully connected layer.
        # its the last phase before prediction.this layer is going to have 10 different outputs(scores) as it was determined.
        
        return x #7-function finally returns last 10 outputs (scores) for each image.