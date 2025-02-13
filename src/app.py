import gradio as gr
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image
from model_module import SegmentationModel

# Load the trained model
model_checkpoint = "path/to/your/model_checkpoint.ckpt"
num_classes = 10  # Adjust based on your use case
model = SegmentationModel.load_from_checkpoint(model_checkpoint, num_classes=num_classes)
model.eval()

# Define preprocessing function
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])
    image = transform(image).unsqueeze(0)  # Add batch dimension
    return image

# Define inference function
def predict(image):
    image = preprocess_image(image)
    with torch.no_grad():
        output = model(image)
        prediction = torch.argmax(output, dim=1).squeeze().cpu().numpy()

    # Convert prediction to a colormap
    cmap = plt.get_cmap("viridis")
    colored_pred = cmap(prediction / num_classes)
    return colored_pred

# Define Gradio interface
interface = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil"),
    outputs=gr.Image(type="numpy"),
    title="Segmentation Model",
    description="Upload an image to see the model's segmentation prediction.",
)

# Run the app
if __name__ == "__main__":
    interface.launch()