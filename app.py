import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.models as models
import pandas as pd
import seaborn as sns

# Define a VGG16 model
class VGG16(nn.Module):
    def __init__(self, num_classes=4):
        super(VGG16, self).__init__()
        self.vgg16 = models.vgg16(pretrained=True)
        # Freeze the pre-trained layers
        for param in self.vgg16.parameters():
            param.requires_grad = False
        # Replace the classifier with a new one, excluding the final fully connected layer
        self.vgg16.classifier = nn.Sequential(
            nn.Linear(25088, 4096),  # Adjust input size based on your image size
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes)
        )

    def forward(self, x):
        return self.vgg16(x)


# Define the function to load the model
def load_model(model_path):
    # Define the CNN architecture (assuming the same architecture as before)
    model = VGG16()
    # Load the trained weights
    model.load_state_dict(torch.load(model_path))
    # Set the model to evaluation mode
    model.eval()
    return model

# Define the function to predict on a new image
def predict_image(model, image):
    # Preprocess the image
    preprocess = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image_tensor = preprocess(image).unsqueeze(0)  # Add batch dimension

    # Predict the class probabilities
    with torch.no_grad():
        outputs = model(image_tensor)
        probs = torch.softmax(outputs, dim=1)

    # Get the predicted class label
    _, predicted = torch.max(outputs, 1)
    predicted_label = class_names[predicted.item()]

    # Convert tensor probabilities to a list
    probabilities = probs.squeeze().tolist()

    # Visualize the prediction
    visualize_prediction(image, predicted_label, class_names, probabilities)

# Define the function to visualize the prediction
def visualize_prediction(image, predicted_class, class_names, probabilities):
    st.image(image, caption='Uploaded Image', use_column_width=True)

    st.write(f'Predicted Class: {predicted_class}')
    st.write('Probabilities:')
    # Create a dictionary from class names and probabilities
    data = {'Class Name': class_names, 'Probability': probabilities}

    # Create a DataFrame from the dictionary
    df = pd.DataFrame(data)

    # Sort DataFrame by probabilities in descending order
    df = df.sort_values(by='Probability', ascending=False)

    # Plotting using Seaborn
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Probability', y='Class Name', data=df, palette='viridis')
    plt.xlabel('Probability')
    plt.ylabel('Class Name')
    plt.title('Probabilities for Each Class')
    st.pyplot(plt.gcf())

# Define some example descriptions for tea diseases (replace with your own content)
disease_descriptions = {
    "brown blight": "Brown blight is a fungal disease that affects tea leaves, causing brown patches. These patches can lead to reduced photosynthesis and overall plant health.",
    "bird eye spot": "Bird eye spot is a fungal disease characterized by small, dark spots on tea leaves. These spots resemble bird eyes and can impact leaf health.",
    "Anthracnose": "Anthracnose is a fungal disease causing dark, sunken lesions on tea leaves. These lesions weaken the leaves and affect overall plant vigor.",
}

treatment_methods = {
    "brown blight": "To treat brown blight, remove affected leaves and apply a fungicide specifically designed for brown blight control. Ensure proper plant nutrition and hygiene.",
    "bird eye spot": "To control bird eye spot, maintain proper hygiene in the tea garden, regularly inspect and remove affected leaves, and use fungicides as recommended by experts.",
    "Anthracnose": "To manage anthracnose, prune affected areas to remove infected tissue, improve air circulation within the tea plants, and apply fungicides to prevent further spread.",
}

def show_home_content():
    """Displays content for the home page"""
    st.title('Tea Leaf Disease Classifier')
    st.write("Welcome to the tea leaf disease classifier! This app helps identify potential diseases affecting your tea leaves. Before using the prediction tool, learn more about some common tea leaf diseases:")
    
    image_url = "https://live.staticflickr.com/4322/36147141665_7f022edf2b_b.jpg"  # Replace with the actual image URL or shared link
    st.image(image_url, caption="Tea Leaf", use_column_width=True)
    # Create a button to navigate to the prediction section
    # predict_button = st.button("Predict Now!")

    # # Check if the button is clicked
    # if predict_button:
    #     # Change page to "Predict Disease" by updating URL query parameter
    #     st.query_params(page="Predict Disease")

    # Loop through diseases and display descriptions
    for disease in disease_descriptions:
        st.header(disease)
        st.write(disease_descriptions[disease])
        st.write("Treatment Methods:")
        st.write(treatment_methods[disease])
        st.write("---")  # Add a separator between diseases

def show_prediction_content():
    # Load the saved model
    model_path = '/content/drive/My Drive/tea_vgg16_v1.pth'  # Update with your model path
    loaded_model = load_model(model_path)

    # Define the class names
    class_names = ["brown blight", "bird eye spot", "Anthracnose", "healthy"]  # Replace with your class names

    # Streamlit app for prediction
    st.title('Tea Leaf Disease Classifier')

    # Upload image
    uploaded_image = st.file_uploader('Upload an image', type=['jpg', 'jpeg', 'png'])

    if uploaded_image is not None:
        # ... Your existing code for handling image upload, prediction and visualization ...
        image = Image.open(uploaded_image).convert('RGB')
        # st.image(image, caption='Uploaded Image', use_column_width=True)

        # Predict the class and probabilities
        predict_image(loaded_model, image)

# Create the Streamlit app with navigation
st.sidebar.header("Navigation")
selected_page = st.sidebar.radio("Select a page", ("Home", "Predict Disease"))

if selected_page == "Home":
    show_home_content()
else:
    show_prediction_content()
