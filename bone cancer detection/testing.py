import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the pre-trained ResNet50 model and modify the output layer
model = models.resnet50(pretrained=False)
model.fc = nn.Linear(model.fc.in_features, 3)  # Adjust for your 3 classes
model.load_state_dict(torch.load('best_bone_cancer_model.pth', map_location=device))
model.to(device).eval()  # Set the model to evaluation mode

# Define the transformation to match training
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def predict_image(image_path):
    # Open, preprocess the image and add batch dimension
    image = transform(Image.open(image_path).convert('RGB')).unsqueeze(0).to(device)
    
    # Make predictions
    with torch.no_grad():
        predicted = model(image).argmax(dim=1).item()
    
    # Mapping the prediction to class labels
    class_names = ['Chondrosarcoma', 'Ewing Sarcoma', 'Osteosarcoma']
    return class_names[predicted]

# Test the function
image_path = r"C:\Users\pytorch\Desktop\New folder (2)\dataset\Ewing Sarcoma\png - 2024-10-01T194527.013.jpg"  # Replace with the actual path
prediction = predict_image(image_path)
print(f'Predicted Cancer Type: {prediction}')
