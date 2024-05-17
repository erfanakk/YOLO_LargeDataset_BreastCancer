import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from ultralytics import YOLO
from sklearn.metrics import classification_report
from tqdm import tqdm

# Import necessary libraries

# Change this to the path of the best weights obtained from training
best_weight_pth = 'path for best weight of yolo ' #TODO chagne this 
# Path to the test folder
path_dataset = 'PATH folder' #TODO change this 
batch_size = 32

# Load the test dataset
test_dataset = datasets.ImageFolder(root=path_dataset, transform=None)
image_paths = [path for path, _ in test_dataset.imgs]
labels = [label for _, label in test_dataset.imgs]

# Define a custom dataset class to handle image paths and labels
class PathLabelDataset:
    def __init__(self, paths, labels, transform=None):
        self.paths = paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        image_path = self.paths[idx]
        label = self.labels[idx]
        return image_path, label

path_label_dataset = PathLabelDataset(image_paths, labels)

# Create a data loader for the test dataset
path_label_loader = DataLoader(path_label_dataset, batch_size=batch_size, shuffle=False)

# Initialize YOLO model with the best weights
model = YOLO(best_weight_pth)

# Set device to GPU if available, otherwise use CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

true_labels = []
predicted_labels = []

# Evaluate the model on the test dataset
with torch.no_grad():
    for images, labels in tqdm(path_label_loader, desc="Evaluating"):
        labels =  labels.to(device)

        # Perform inference using YOLO model
        output = model.predict(images, imgsz=224, verbose=False)
        outputs = [output[i].probs.top1 for i in range(len(output))]

        true_labels.extend(labels.cpu().numpy())
        predicted_labels.extend(outputs)

# Calculate confusion matrix
conf_matrix = confusion_matrix(true_labels, predicted_labels)

class_names = ['Benign', 'Malignant', 'Normal']

true_labels = []
predicted_labels = []

# Calculate true and predicted labels for classification report
for true_class in range(len(conf_matrix)):
    true_labels.extend([true_class] * conf_matrix[true_class].sum())
    predicted_labels.extend([i for i in range(len(conf_matrix)) for _ in range(conf_matrix[true_class][i])])

# Print classification report
print(classification_report(true_labels, predicted_labels, target_names=class_names, digits=4))

# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.show()
