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


# Training function
def train_autoencoder(model, data, num_epochs=10, batch_size=64):

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

def visualize_levels(levels, save_path=None):
    num_levels = len(levels)
    fig, axes = plt.subplots(1, num_levels, figsize=(5*num_levels, 5))
    
    # If only one level, axes will not be an array, so we convert it to a single-element list
    if num_levels == 1:
        axes = [axes]

    # Color mapping
    color_map = {
        0: [0.6, 0.4, 0.2],  # Wall: Brown
        1: [0.9, 0.9, 0.9],  # Floor: Light Gray
        2: [1.0, 0.0, 0.0],  # Box: Red
        3: [0.0, 0.8, 0.0],  # Player: Green
        4: [0.0, 0.0, 1.0]   # Goal: Blue
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
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Visualization saved to {save_path}")
    else:
        plt.show()

    plt.close()

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

def post_process_level(level, num_boxes=3, min_size=8, max_size=15):
    # Convert the continuous values to discrete elements
    level = level.cpu().numpy()
    level = np.argmax(level, axis=0)
    
    # Randomly choose level size
    height = width = np.random.randint(min_size, max_size + 1)
    
    # Start with all walls
    level = np.zeros((height, width), dtype=int)
    
    # Create an inner area for the game
    inner_start = 1
    inner_end_h, inner_end_w = height - 1, width - 1
    
    # Fill the inner area with floor
    level[inner_start:inner_end_h, inner_start:inner_end_w] = 1
    
    def get_random_floor_pos():
        while True:
            h = np.random.randint(inner_start, inner_end_h)
            w = np.random.randint(inner_start, inner_end_w)
            if level[h, w] == 1:
                return h, w

    # Place one player
    player_pos = get_random_floor_pos()
    level[player_pos] = 3

    # Place boxes and goals
    for _ in range(num_boxes):
        # Place a box
        box_pos = get_random_floor_pos()
        level[box_pos] = 2

        # Place a goal
        goal_pos = get_random_floor_pos()
        level[goal_pos] = 4

    # Add some random walls inside the level
    num_inner_walls = (height * width) // 10
    for _ in range(num_inner_walls):
        wall_pos = get_random_floor_pos()
        level[wall_pos] = 0

    return level

# Update the generate_level function
def generate_level(model, num_levels=1, num_boxes=3, min_size=8, max_size=15):
    device = next(model.parameters()).device
    model.eval()
    generated_levels = []
    with torch.no_grad():
        for _ in range(num_levels):
            latent_vector = torch.randn(1, 256, 10, 10).to(device)
            generated_level = model.decoder(latent_vector)
            generated_level = generated_level.squeeze(0)
            post_processed_level = post_process_level(generated_level, num_boxes, min_size, max_size)
            generated_levels.append(post_processed_level)
    return generated_levels



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
    new_levels = generate_level(model, num_levels=5, num_boxes=3, min_size=8, max_size=15)
    
    print("New levels generated.")
    visualize_levels(new_levels, save_path="generated_levels.png")

    print("Visualization of generated levels saved as 'generated_levels.png'")
