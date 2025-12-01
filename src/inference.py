import torch
from torchvision import transforms
from torchvision.models import resnet18
from PIL import Image
import numpy as np

class CIFAR10Inference:
    def __init__(self, model_path):
        self.model = resnet18(num_classes=10)
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()
        self.transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

    def preprocess_image(self, image_path):
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)
        return image.unsqueeze(0)  # Add batch dimension

    def predict(self, image_path):
        image_tensor = self.preprocess_image(image_path)
        with torch.no_grad():
            outputs = self.model(image_tensor)
            _, predicted = torch.max(outputs, 1)
        return predicted.item()

if __name__ == "__main__":
    import sys
    model_path = 'path/to/your/model.pth'  # Update this path
    image_path = sys.argv[1]  # Get image path from command line argument
    classifier = CIFAR10Inference(model_path)
    prediction = classifier.predict(image_path)
    print(f'Predicted class: {prediction}')