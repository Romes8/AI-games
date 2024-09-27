# %% import
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch import device
import numpy as np
import jax.numpy as jnp
import pickle
import os
import matplotlib.pyplot as plt
import torchvision.utils as vutils

# Define the autoencoder architecture
class SokobanAutoencoder(nn.Module):
    def __init__(self):
        super(SokobanAutoencoder, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.ReLU()
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

# Function to load and preprocess data
def load_data():
    # Load your 3D tensor data from the pickle file
    with open('sokoban_levels_combined.pkl', 'rb') as f:
        levels_loaded = pickle.load(f)

    # Convert to JAX array and normalize
    levels_loaded = jnp.array(levels_loaded).astype(jnp.float32) / 242.0

    # Print information about the dataset
    print("Dataset shape:", levels_loaded.shape)
    print("Min pixel value:", jnp.min(levels_loaded))
    print("Max pixel value:", jnp.max(levels_loaded))

    # Convert JAX array to PyTorch tensor and permute dimensions
    data = torch.from_numpy(np.array(levels_loaded)).permute(0, 3, 1, 2)
    return data

def visualize_reconstruction(model, inputs, outputs, epoch):
    # Move tensors to CPU
    inputs = inputs.cpu()
    outputs = outputs.detach().cpu()

    # Create a grid of original and reconstructed images
    comparison = torch.cat([inputs[:8], outputs[:8]])
    grid = vutils.make_grid(comparison, nrow=8, normalize=True, scale_each=True)
    
    # Convert to numpy for matplotlib
    grid = grid.numpy().transpose((1, 2, 0))

    # Plot
    plt.figure(figsize=(15, 6))
    plt.imshow(grid)
    plt.title(f"Original (top) vs Reconstructed (bottom) - Epoch {epoch+1}")
    plt.axis('off')
    
    # Save the figure
    plt.savefig(f"reconstruction_epoch_{epoch+1}.png")
    plt.close()

    print(f"Visualization saved for epoch {epoch+1}")

# Training function
def train_autoencoder(model, data, num_epochs=50, batch_size=32):

    device = next(model.parameters()).device  # Get the device from the model

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters())
    
    dataset = TensorDataset(data)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    for epoch in range(num_epochs):
        total_loss = 0
        for batch in dataloader:
            inputs = batch[0].to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, inputs)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/len(dataloader):.4f}")

    visualize_reconstruction(model, inputs, outputs, epoch)

# Generate new levels
def generate_level(model, num_levels=1):
    device = next(model.parameters()).device  # Get the device from the model
    model.eval()
    with torch.no_grad():
        latent_vector = torch.randn(num_levels, 256, 10, 10).to(device)
        generated_levels = model.decoder(latent_vector)
    return generated_levels

def save_model(model, path='sokoban_autoencoder.pth'):
    directory = os.path.dirname(path)
    if not os.path.exists(directory):
        os.makedirs(directory)
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")

def load_model(model, path='sokoban_autoencoder.pth'):
    if os.path.exists(path):
        model.load_state_dict(torch.load(path))
        print(f"Model loaded from {path}")
    else:
        print(f"No saved model found at {path}")
    return model

# Post-processing function to ensure playability
def post_process_level(level):
    # Convert the continuous values to discrete elements
    level = level.numpy()
    level = np.argmax(level, axis=0)
    
    # Ensure there's at least one player, one box, and one goal
    if 3 not in level:
        level[np.unravel_index(np.argmax(level == 0), level.shape)] = 3
    if 1 not in level:
        level[np.unravel_index(np.argmax(level == 0), level.shape)] = 1
    if 2 not in level:
        level[np.unravel_index(np.argmax(level == 0), level.shape)] = 2
    
    # Ensure the level is surrounded by walls
    level[0, :] = level[-1, :] = level[:, 0] = level[:, -1] = 0
    
    return level

def visualize_levels(levels, save_path=None):
    num_levels = len(levels)
    fig, axes = plt.subplots(1, num_levels, figsize=(5*num_levels, 5))
    
    # If only one level, axes will not be an array, so we convert it to a single-element list
    if num_levels == 1:
        axes = [axes]

    # Color mapping
    color_map = {
        0: [0, 0, 0],       # Wall: Black
        1: [1, 1, 1],       # Floor: White
        2: [1, 0, 0],       # Box: Red
        3: [0, 1, 0],       # Player: Green
        4: [0, 0, 1]        # Goal: Blue
    }

    for i, level in enumerate(levels):
        # Create RGB image
        rgb_level = np.zeros((*level.shape, 3))
        for value, color in color_map.items():
            rgb_level[level == value] = color

        axes[i].imshow(rgb_level)
        axes[i].set_title(f"Level {i+1}")
        axes[i].axis('off')

    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Visualization saved to {save_path}")
    else:
        plt.show()

    plt.close()

# Main execution
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load data
    data = load_data()
    data = data.to(device)  # Move data to the appropriate device

    model = SokobanAutoencoder().to(device)

    saved_model_path = 'C:\\Transfer\\Dokumenty\\DANSKO\\ITU\\Games\\3rdSemester\\AI\\AI-games\\prjs\\two\\sokoban_autoencoder.pth'
    if os.path.exists(saved_model_path):
        model = load_model(model, saved_model_path)
    else:
        # Train the model
        train_autoencoder(model, data)
        # Save the trained model
        save_model(model, saved_model_path)
    
    # Generate new levels
    new_levels = generate_level(model, num_levels=5)
    new_levels = new_levels.cpu()  # Move back to CPU for post-processing

    
    print("New levels generated. Shape:", new_levels.shape)
    post_processed_levels = [post_process_level(level) for level in new_levels]


    visualize_levels(post_processed_levels, save_path="generated_levels.png")

    print("Visualization of generated levels saved as 'generated_levels.png'")
# %%
