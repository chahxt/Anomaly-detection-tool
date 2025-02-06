import os
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
from models.autoencoder import Autoencoder
import matplotlib.pyplot as plt
from PIL import Image

# Dataset for loading the frames (NumPy arrays)
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

# Load the trained model
def load_model(model_path):
    model = Autoencoder().cuda()  # Move model to GPU if available
    model.load_state_dict(torch.load(model_path))  # Load trained model weights
    model.eval()  # Set model to evaluation mode
    return model

# Compute reconstruction error (Mean Squared Error)
def compute_reconstruction_error(original, reconstructed):
    return torch.mean((original - reconstructed) ** 2, dim=(1, 2, 3))

# Detect anomalies in the frames
def detect_anomalies(model, dataloader, threshold=0.01):
    anomalies = []
    for data in dataloader:
        data = data.cuda()  # Move data to GPU if available
        with torch.no_grad():
            output = model(data)  # Get the model's reconstruction

        # Calculate reconstruction error
        reconstruction_error = compute_reconstruction_error(data, output)

        # Identify anomalies based on threshold
        for i, error in enumerate(reconstruction_error):
            if error > threshold:
                anomalies.append(i)  # Store the index of anomalous frame
    return anomalies

# Save the anomalous frames
def save_anomalous_frames(anomalies, dataloader, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    for anomaly_idx in anomalies:
        frame = dataloader.dataset[anomaly_idx].cpu().numpy().transpose(1, 2, 0)  # Convert back to HxWxC
        frame_image = Image.fromarray(np.uint8(frame * 255))  # Convert to Image format
        frame_image.save(os.path.join(output_folder, f"anomaly_{anomaly_idx}.png"))

# Visualize the anomalies
def visualize_anomalies(anomalies, dataloader):
    for anomaly_idx in anomalies:
        frame = dataloader.dataset[anomaly_idx].cpu().numpy().transpose(1, 2, 0)  # Convert back to HxWxC
        plt.imshow(frame)
        plt.title(f"Anomaly detected in frame {anomaly_idx}")
        plt.show()

# Main function
def main():
    # Paths
    model_path = 'output/model/autoencoder.pth'  # Path to the trained model
    frames_folder = 'data/processed/subway_video_1'  # Path to the extracted frames
    result_folder = 'output/results/anomalies'  # Folder where anomalous frames will be saved

    # Load dataset and dataloader
    dataset = FrameDataset(frames_folder)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)  # Set batch size to 1 for frame-by-frame evaluation

    # Load trained model
    model = load_model(model_path)

    # Detect anomalies
    anomalies = detect_anomalies(model, dataloader, threshold=0.01)

    # Print detected anomalies
    print(f"Anomalous frames detected: {anomalies}")
    
    # Visualize the anomalies
    visualize_anomalies(anomalies, dataloader)

    # Save anomalous frames
    save_anomalous_frames(anomalies, dataloader, result_folder)

if __name__ == "__main__":
    main()
