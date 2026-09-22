from classes import *
import cv2 as cv
# Model's final version can be tested after the training process.
#Images that were not previously seen by the model must be used during this test. 

#image pre-process phase
img=cv.imread("assets/testing_image.png",cv.IMREAD_GRAYSCALE)
img_resized=cv.resize(img,(28,28))
img_normalized=img_resized/255.0
img_tensor=torch.tensor(img_normalized, dtype=torch.float32)
img_vector=img_tensor.view(1,784)

#In order to start testing, first we need to initialize a new model.
tester_model=NeuralNetwork()

#Load the weights of best.pt to the new model.
tester_model.load_state_dict(torch.load(r"weights/best.pt"))

#Set model into evaluation mode. 
tester_model.eval()

#Get the model's prediction
with torch.no_grad():
    predictions=tester_model(img_vector)
    predicted_class=torch.argmax(predictions).item()
    display_img=img_vector.cpu().squeeze().numpy()
    display_img=display_img.reshape(28,28)
    plt.imshow(display_img,cmap='gray')
    plt.title(f"Predicted Class: {predicted_class}", fontsize=20, color= 'green')
    plt.axis('off')
    plt.savefig(r"assets/model-prediction.png")
    plt.show()
    