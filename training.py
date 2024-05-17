# Importing YOLO from Ultralytics package
from ultralytics import YOLO

# Define training parameters
data_path = 'path dataset'  # TODO: Replace with your dataset path
weights = "yolov8x-cls.pt"  # Pre-trained weights, you can change the model (s, m, l, or x)
img_size = 224  # Image size for training
batch_size = 32  # Batch size
epochs = 150  # Number of training epochs
workers = 4  # Number of data loader workers 

lr0 = 0.01  # Initial learning rate
lrf = 0.1  # Final learning rate
momentum = 0.937  # Momentum value for optimization
weight_decay = 0.0005  # Weight decay for regularization
patience = 30  # Patience for early stopping

name = "yolov8X"  # Name for the training experience
optimizer = "Adam"  # Optimizer choice
seed = 0  # Seed for reproducibility

# Initializing YOLO model with pre-trained weights
model = YOLO(weights)

# Training the model
results = model.train(data=data_path, epochs=epochs, imgsz=img_size, batch=batch_size, patience=patience, name=name, 
                      optimizer=optimizer, seed=seed, lr0=lr0, lrf=lrf, momentum=momentum, weight_decay=weight_decay)
