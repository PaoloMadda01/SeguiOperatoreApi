from ultralytics import YOLO
import requests
from PIL import Image
from io import BytesIO

# Load the YOLOv8 model
model = YOLO('yolov8n.yaml')

# Load an image from a URL
url = 'https://ultralytics.com/images/zidane.jpg'
response = requests.get(url)
image = Image.open(BytesIO(response.content))

# Use the YOLOv8 model to detect objects in the image
results = model(image)

# Print the detected objects and their bounding boxes
for obj in results.xyxy[0]:
    x1, y1, x2, y2, conf, cls = obj
    print(f"Detected object: {results.names[int(cls)]} with confidence {conf:.2f} at [{x1:.0f}, {y1:.0f}, {x2:.0f}, {y2:.0f}]")

