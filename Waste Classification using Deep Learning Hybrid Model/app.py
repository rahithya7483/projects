from flask import Flask, request, render_template, jsonify
import torch
import torchvision.transforms as transforms
from PIL import Image
import torch.nn as nn
import torchvision.models as models

# Define RWC-Net Model (Same as Training)
class RWCNet(nn.Module):
    def __init__(self):
        super(RWCNet, self).__init__()
        self.densenet = models.densenet201(pretrained=False)
        self.mobilenet = models.mobilenet_v2(pretrained=False)

        # Remove the final classification layers
        self.densenet.classifier = nn.Identity()
        self.mobilenet.classifier = nn.Identity()

        # Fully connected layer for classification
        self.fc = nn.Sequential(
            nn.Linear(1920 + 1280, 512),  
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 6),  
            nn.LogSoftmax(dim=1)
        )

    def forward(self, x):
        x1 = self.densenet(x)
        x2 = self.mobilenet(x)
        x_combined = torch.cat((x1, x2), dim=1)
        return self.fc(x_combined)

# Load Model
device = "cuda" if torch.cuda.is_available() else "cpu"
model = RWCNet().to(device)
model.load_state_dict(torch.load("models/rwc_net.pth", map_location=device))
model.eval()

# Define Flask app
app = Flask(__name__)

# Define Transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Waste categories
categories = ["cardboard", "glass", "metal", "paper", "plastic", "litter"]

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['file']
    image = Image.open(file).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image)
        _, predicted_class = torch.max(outputs, 1)

    category = categories[predicted_class.item()]
    return render_template('index.html', category=category)

if __name__ == '__main__':
    app.run(debug=True)

