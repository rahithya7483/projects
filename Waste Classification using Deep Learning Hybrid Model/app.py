from flask import Flask, request, render_template
import torch
import torchvision.transforms as transforms
from PIL import Image
import torch.nn as nn
import torchvision.models as models
import os

# Define Flask app and specify static folder
app = Flask(__name__, static_folder="static")

# Define RWC-Net Model
class RWCNet(nn.Module):
    def __init__(self):
        super(RWCNet, self).__init__()
        self.densenet = models.densenet201(weights=None)
        self.mobilenet = models.mobilenet_v2(weights=None)

        # Remove final classification layers
        self.densenet.classifier = nn.Identity()
        self.mobilenet.classifier = nn.Identity()

        # Fully connected classification layer
        self.fc = nn.Sequential(
            nn.Linear(1920 + 1280, 512),  
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 9),  # 9 categories
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
model.load_state_dict(torch.load("rwc_net11.pth", map_location=device))
model.eval()

# Define Transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Waste categories
categories = ["battery", "biological", "brown-glass", "cardboard", "clothes", "metal", "paper", "plastic", "trash"]

# Waste Information (Recyclability, Uses, Problems)
waste_info = {
    "battery": {
        "recyclable": "No",
        "usage": "Must be disposed of in special recycling centers due to hazardous chemicals.",
        "problem": "Can leak toxic chemicals like lead and mercury, harming soil and water.",
        "image": "static/images/battery.jpg"
    },
    "biological": {
        "recyclable": "No",
        "usage": "Can be used for composting or biodegradable waste disposal.",
        "problem": "Improper disposal leads to disease spread and foul odor.",
        "image": "static/images/biological_waste.jpg"
    },
    "brown-glass": {
        "recyclable": "Yes",
        "usage": "Reused for making new glass bottles and decorative items.",
        "problem": "If broken and not recycled, it can cause injuries and landfill waste.",
        "image": "static/images/glass.jpg"
    },
    "cardboard": {
        "recyclable": "Yes",
        "usage": "Used for packaging, making recycled paper, or creative DIY crafts.",
        "problem": "If not recycled, it leads to deforestation and excess waste.",
        "image": "static/images/cardboard.jpeg"
    },
    "clothes": {
        "recyclable": "Yes",
        "usage": "Can be donated, upcycled into new fabrics, or used for insulation.",
        "problem": "Non-biodegradable synthetic fabrics contribute to microplastic pollution.",
        "image": "static/images/clothes.jpeg"
    },
    "metal": {
        "recyclable": "Yes",
        "usage": "Can be melted and reused for new metal products like cans and machinery.",
        "problem": "Takes years to degrade in landfills and releases toxins.",
        "image": "static/images/metal.jpeg"
    },
    "paper": {
        "recyclable": "Yes",
        "usage": "Recycled into newspapers, packaging, and notebooks.",
        "problem": "Cutting trees for paper leads to deforestation.",
        "image": "static/images/paper.jpg"
    },
    "plastic": {
        "recyclable": "Depends",
        "usage": "Some plastics can be recycled into new plastic products.",
        "problem": "Non-recyclable plastics pollute oceans and take hundreds of years to degrade.",
        "image": "static/images/plastic.jpg"
    },
    "trash": {
        "recyclable": "No",
        "usage": "General waste, should be minimized by proper segregation.",
        "problem": "Leads to landfill overflow, pollution, and harmful gases.",
        "image": "static/images/trash.jpg"
    },
}

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
    info = waste_info[category]

    return render_template('index.html', category=category, recyclable=info["recyclable"], usage=info["usage"], problem=info["problem"], image=info["image"])

if __name__ == '__main__':
    app.run(debug=True)
