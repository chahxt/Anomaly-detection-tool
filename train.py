import sys
import os

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Add the 'models' folder to the sys.path
sys.path.insert(0, os.path.join(root_dir, 'models'))

print("Updated sys.path:", sys.path)
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np

from models.autoencoder import Autoencoder  # Import Autoencoder from models

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the dataset for loading the frames
class FrameDataset(Dataset):
    def __init__(self, frames_folder, transform=None):
        self.frames_folder = frames_folder
        self.frame_files = [f for f in os.listdir(frames_folder) if f.endswith('.npy')]
        self.transform = transform

    def __len__(self):
        return len(self.frame_files)

    def __getitem__(self, idx):
        frame_path = os.path.join(self.frames_folder, self.frame_files[idx])
        frame = np.load(frame_path)  # Load the frame as a NumPy array
        frame = torch.tensor(frame, dtype=torch.float32).permute(2, 0, 1)  # Convert to CxHxW

        if self.transform:
            frame = self.transform(frame)

        return frame

# Hyperparameters
batch_size = 16
epochs = 10
learning_rate = 1e-3

# Data loading
frames_folder = "data/processed/subway_video_1"  # Folder with extracted frames
dataset = FrameDataset(frames_folder)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Model, loss function, and optimizer
model = Autoencoder().to(device)  # Move model to device (GPU/CPU)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for data in dataloader:
        data = data.to(device)  # Move data to the same device as the model
        
        # Zero gradients, perform a backward pass, and update the weights
        optimizer.zero_grad()

        # Forward pass
        output = model(data)

        # Compute the loss
        loss = criterion(output, data)

        # Backward pass
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(dataloader):.4f}")

# Save the model
torch.save(model.state_dict(), "output/model/autoencoder.pth")
print("Model saved!")
