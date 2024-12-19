import torch
import streamlit as st
from torchvision import transforms, models
from PIL import Image
import torch.nn as nn
import os

# ChestXRayClassifier definition (updated according to your model)
class ChestXRayClassifier:
    def __init__(self, disease_type, device='cuda'):
        self.device = device
        self.disease_type = disease_type
        
        # Define possible diseases for binary classification
        if self.disease_type == 'COVID-19 vs Normal':
            self.classes = ['COVID-19', 'NORMAL']
        elif self.disease_type == 'Pneumonia vs Normal':
            self.classes = ['PNEUMONIA', 'NORMAL']
        elif self.disease_type == 'TB vs Normal':
            self.classes = ['TUBERCULOSIS', 'NORMAL']
        else:
            raise ValueError("Invalid disease type selected!")

        # Model setup (using EfficientNet as base)
        self.model = models.efficientnet_b0(weights='IMAGENET1K_V1')

        # Custom Classifier
        num_ftrs = self.model.classifier[1].in_features
        self.model.classifier = nn.Sequential(
            nn.Dropout(0.3),  # Increased dropout
            nn.Linear(num_ftrs, 1024),
            nn.ReLU(),
            nn.BatchNorm1d(1024),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.2),
            nn.Linear(512, 256),  # Additional layer
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.1),
            nn.Linear(256, len(self.classes))  # Adjusting output for binary classification (len(self.classes) = 2)
        )
        self.model = self.model.to(self.device)
        
        # Load the trained model weights (assumes the file is in the current directory)
        checkpoint = torch.load("best_model.pth", map_location=self.device, weights_only=True)
        
        # Load weights for feature extractor only (exclude classifier)
        state_dict = checkpoint['model_state_dict']
        state_dict = {k: v for k, v in state_dict.items() if 'classifier' not in k}  # Exclude classifier weights
        self.model.load_state_dict(state_dict, strict=False)
        
        self.model.eval()  # Set model to evaluation mode

    def predict(self, image):
        image = image.convert('RGB')

        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        image = transform(image).unsqueeze(0).to(self.device)  # Add batch dimension
        with torch.no_grad():
            outputs = self.model(image)
            _, predicted = outputs.max(1)
        return self.classes[predicted.item()]

# Streamlit interface
def main():
    st.set_page_config(page_title="LungScanAI", layout="wide")
    
    # Styling with background image, transparent black overlay, and content layer
    st.markdown(
        """
        <style>
        /* Background image */
        .stApp {
            background-image: url("https://img.freepik.com/premium-photo/essential-diagnostics-lung-health-medical-research-clinical-care-concept-pulmonary-function-tests-chest-xray-ct-scan-bronchoscopy-blood-gas-analysis_864588-72767.jpg?w=1380");
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
            z-index: -2;
            padding-top: 0 !important; /* Remove default Streamlit padding */
        }

        /* Black transparent overlay */
        .background-overlay {
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: rgba(0, 0, 0, 0.0); /* Black overlay with 60% opacity */
            z-index: -1; /* Ensure it stays behind the content */
        }

        /* Content header */
        .header {
            background-color: rgba(31, 119, 180, 0.8);
            color: white;
            padding: 10px;
            text-align: center;
            border-radius: 10px;
            font-size: 32px;
            margin-top: 0 !important; /* Remove space above the title */
            padding-top: 0 !important; /* Remove any padding at the top */
            margin-bottom: 20px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        
        /* Section cards for upload, image display, and prediction */
        .upload-section, .image-display, .prediction-box {
            background: rgba(255,255,255,0.8);
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }

        .prediction-box {
            text-align: center;
            font-size: 20px;
            font-weight: bold;
        }
        
        img {
            max-height: 400px;
            width: auto;
            margin: 0 auto;
            display: block;
        }

        .element-container {
            margin-bottom: 10px !important;
        }

        .stSelectbox {
            margin-bottom: 10px;
        }

        /* Set all form labels and input descriptions to white */
        .stMarkdown h3, 
        .stMarkdown label, 
        .stMarkdown span, 
        .stFileUploader label, 
        .stSelectbox label {
            color: white !important; 
        }

        </style>
        """, unsafe_allow_html=True
    )

    # Black transparent overlay
    st.markdown('<div class="background-overlay"></div>', unsafe_allow_html=True)

    # Header section
    st.markdown('<div class="header">LungScanAI : Chest X-Ray Disease Classifier</div>', unsafe_allow_html=True)

    # Create two columns
    col1, col2 = st.columns([1, 1.5])

    # Left column - Upload and Disease Selection
    with col1:
        st.markdown('<div class="upload-section">', unsafe_allow_html=True)
        uploaded_image = st.file_uploader("Upload an X-ray image", type=["jpg", "png", "jpeg"])
        
        disease_type = st.selectbox(
            "Select the disease to detect:",
            ["COVID-19 vs Normal", "Pneumonia vs Normal", "TB vs Normal"]
        )
        st.markdown('</div>', unsafe_allow_html=True)

    # Right column - Image Display and Prediction
    with col2:
        if uploaded_image:
            st.markdown('<div class="image-display">', unsafe_allow_html=True)
            image = Image.open(uploaded_image)
            st.image(image, use_container_width=True)
            
            # Initialize the classifier and make prediction
            classifier = ChestXRayClassifier(disease_type=disease_type, 
                                          device='cuda' if torch.cuda.is_available() else 'cpu')
            prediction = classifier.predict(image)

            # Display prediction
            st.markdown(f"""
                <div class="prediction-box">
                    Prediction: {prediction}
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()