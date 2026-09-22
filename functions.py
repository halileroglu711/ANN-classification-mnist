#import libraries
import torch #pytorch library for tensor 
import torch.nn as nn #needed to define ann layer   
import torch.optim as optim #needed for backpropagation
import torchvision #includes image processing and pre-trained models 
import torchvision.transforms as transforms #needed for image pre-processing
import matplotlib.pyplot as plt #classic data visualization

device=torch.device("cuda"if torch.cuda.is_available() else "cpu")

#define a function to get train_loader and test_loader
def get_data_loaders(batch_size=64):#batch_size stands for the amount of data which will be processed in every iteration.
    
    transform=transforms.Compose([
        transforms.ToTensor(),#converts image to tensor and scales pixels between 0-1 (matris)
        transforms.Normalize((0.5,),(0.5,))#normalizes tensors according to mean and std between -1 and 1
    ])
    #download MNIST dataset and create train and test sets
    train_set=torchvision.datasets.MNIST(root="./data",train=True,download=True,transform=transform)
    test_set=torchvision.datasets.MNIST(root="./data",train=False,download=True,transform=transform)
    #create pytorch data loader(train and test loader)
    train_loader=torch.utils.data.DataLoader(train_set,batch_size=batch_size,shuffle=True)
    test_loader=torch.utils.data.DataLoader(test_set,batch_size=batch_size,shuffle=False)
    
    return train_loader,test_loader

#define a function to visualize some samples
def visualize_samples(loader,n):

    images,labels=next(iter(loader))#get images and labels from first batch (1 of 987)
    fig,axes=plt.subplots(1,n,figsize=(10,5))
    for i in range(n):
        axes[i].imshow(images[i].squeeze(),cmap="gray")
        axes[i].set_title(f"Label: {labels[i].item()}")
        axes[i].axis("off")
    plt.show()

#define a function to determine loss calculation criterion and optimizer for weight updates
define_loss_and_optimizer= lambda model:(
    nn.CrossEntropyLoss(),#loss function for multi class classification problems
    optim.Adam(model.parameters(),lr=0.001)#upgrade weights with adam
)

#define a function to start training process.
def train_model(model,train_loader,criterion,optimizer,epochs=10):
    
    #set model into train mode
    model.train()
    #create a list to keep loss values after every epoch
    train_losses=[]
    #start training for 10 epochs
    for epoch in range(epochs):
        #epoch means learning steps.10 is default.
        total_loss=0 #keeps total lass value after every epoch
        
        #iteration over all training data
        for images,labels in train_loader:
            images,labels=images.to(device),labels.to(device)
            
            #zero all gradiants
            optimizer.zero_grad()
            
            #apply model.(forward propagation)
            predictions=model(images)#returns a tensor [64,10] each 10 stands for 10 outputs of each image
            
            loss=criterion(predictions,labels)#compute loss => y_prediction and y_real returns loss value for each image,its a list. 
            
            #calculate gradiants with backward propagation
            loss.backward()
            
            #update weights (learning)
            optimizer.step()
            
            total_loss+=loss.item()#loss.item() stands for the loss of the current batch.we have more
            #than 900 batches so it takes more than 900 iterations.
            
        avg_loss=total_loss/len(train_loader) #computes average loss value of all train set(more than 900 batches)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss}")
        train_losses.append(avg_loss)
    torch.save(model.state_dict(),"weights/best.pt")
    print("Training process has finished. Best weights were saved as best.pt")
    #loss graph
    plt.figure()
    plt.plot(range(1,epochs+1),train_losses,marker="o",linestyle="-",label="Train Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.legend()
    plt.savefig("assets/training_loss_graph.png")
    plt.show()
    
    
#define a function to see the accuracy of the model.    
def test_model(model,test_loader):
    model.eval() #set model into evaluation mode
    correct=0 #variable to hold correct predictions
    total=0 #variable to hold total predictions
    
    with torch.no_grad():#do not compute gradians.there is no need to update weights on test phase.
        for images,labels in test_loader:
            images,labels=images.to(device),labels.to(device)
            predictions=model(images)
            __,predicted=torch.max(predictions,1)#pick the highest value for every 10 predictions.predicted variable holds 64 different values
            total+=labels.size(0)
            correct+=(predicted==labels).sum().item()
    print(f"Model Accuracy : {100*correct/total:.3f}%")