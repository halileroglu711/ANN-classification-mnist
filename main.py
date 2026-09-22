from functions import *
from classes import *
def train():
    #  ALL STEPS FOR MODEL TRAINING FROM 1-8

    # 1- Enable GPU for training.(set device to "cuda")
    device=torch.device("cuda"if torch.cuda.is_available() else "cpu")

    # 2 - Get train and test loaders.We can do both with one function
    train_loader,test_loader=get_data_loaders()

    # 3 - Visualize data samples
    visualize_samples(train_loader,5)

    # 4 - Create an instance from NeuralNetwork class.
    model=NeuralNetwork().to(device)

    # 5 - Define criterion for loss calculation and optimizer.
    criterion,optimizer=define_loss_and_optimizer(model)

    # 6 - Start training.
    train_model(model,train_loader,criterion,optimizer)

    # 7 - Test model's accuracy at the end of the process.
    test_model(model,test_loader)
if __name__=="__main__":
    train()