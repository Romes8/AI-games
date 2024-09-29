# %% import
import flax
import gym
import gym_sokoban
import numpy as np
import matplotlib.pyplot as plt
import optax
import pickle
import jax
from jax import nn
import jax.numpy as jnp
import flax.linen as nn
import optax
from flax.training import train_state
import jax.numpy as jnp


# %% Get levels -----------------------------------
# Create Sokoban environment
env = gym.make('Sokoban-v0')

# Initialize levels
levels_per_file = 250

for i in range(4):
    # Initialize levels for this section
    levels = []
    
    # Collect 250 levels
    for l in range(levels_per_file):
        observation = env.reset()
        levels.append(observation)
        print("Shape" + observation.shape)
        if l % 10 == 0:
            print(f"Still going... ,", l)
    
    # Convert levels to a numpy array
    levels_array = np.array(levels)
    
    # Save levels to a pickle file
    with open(f'sokoban_levels_part2_{i+1}.pkl', 'wb') as f:
        pickle.dump(levels_array, f)
    
    print(f"Levels saved to sokoban_levels_part2_{i+1}.pkl.")


# %% Combine levels
all_levels = []

# Load each pickle file and append levels to all_levels
for i in range(4):
    with open(f'sokoban_levels_part2_{i+1}.pkl', 'rb') as f:
        levels = pickle.load(f)
        all_levels.extend(levels)

# Convert all levels to a numpy array
all_levels_array = np.array(all_levels)

# Save the combined levels to a new pickle file
with open('sokoban_levels_combined2.pkl', 'wb') as f:
    pickle.dump(all_levels_array, f)

print("All levels combined and saved to sokoban_levels_combined.pkl.")

# %% Load levels ---------------------------------------------------
with open('sokoban_levels_combined.pkl', 'rb') as f:
    levels_loaded = pickle.load(f)

# %% Show one level, get the shape as well --------------------
level_index = 600

print("Original Min pixel value:", jnp.min(levels_loaded[8]))
print("Original Max pixel value:", jnp.max(levels_loaded[8]))

levels_loaded = jnp.array(levels_loaded).astype(jnp.float32) / 242.0
print("Level shape: ",levels_loaded[level_index].shape)
print("Dataset shape: ", levels_loaded.shape)

print("New Min pixel value:", jnp.min(levels_loaded[8]))
print("New Max pixel value:", jnp.max(levels_loaded[8]))

# Split into training and test sets - first 900 levels used for traning and 100 for testing
levels_loaded = levels_loaded[:900]  # Train data
test_levels = levels_loaded[900:]    # Test data


plt.imshow(levels_loaded[level_index])
plt.title("Sokoban Level")
plt.axis('off')
plt.show()



# %% Define the Encoder and Decoder -------------------------------------
class Encoder(nn.Module):
    ##latent_dim: int
    @nn.compact
    def __call__(self, x):
        x = nn.Conv(features=32, kernel_size=(3, 3), strides=(2, 2))(x)
        x = nn.relu(x)
        x = nn.Conv(features=64, kernel_size=(3, 3), strides=(2, 2))(x)
        x = nn.relu(x)
        x = nn.Conv(features=128, kernel_size=(3, 3), strides=(2, 2))(x)
        x = nn.relu(x)
        x = x.reshape((x.shape[0], -1))  # Flatten
        return x

class Decoder(nn.Module):
    ##output_shape: tuple
    @nn.compact
    def __call__(self, x):
        x = x.reshape((-1, 20, 20, 128))
        x = nn.ConvTranspose(features=64, kernel_size=(3, 3), strides=(2, 2))(x)
        x = nn.relu(x)
        x = nn.ConvTranspose(features=32, kernel_size=(3, 3), strides=(2, 2))(x)
        x = nn.relu(x)
        x = nn.ConvTranspose(features=3, kernel_size=(3, 3), strides=(2, 2))(x)
        return x

# %% Autoencoder
class Autoencoder(nn.Module):
    #latent_dim: int
    #output_shape: tuple

    def setup(self):
        self.encoder = Encoder()
        self.decoder = Decoder()
    
    def __call__(self, x):
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed
    
    def decode(self, z):
        return self.decoder(z)

def create_train_state(rng, learning_rate):
    model = Autoencoder()
    variables = model.init(rng, jnp.ones((1, 160, 160, 3)))
    params = variables['params']
    tx = optax.adam(learning_rate)
    return train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)

@jax.jit
def train_step(state, batch):
    def loss_fn(params):
        reconstructed = state.apply_fn({'params': params}, batch)
        loss = jnp.mean((reconstructed - batch) ** 2)
        return loss
    
    grad_fn = jax.value_and_grad(loss_fn)
    loss, grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)
    return state, loss

def visualize_reconstruction(original, reconstructed, epoch):
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(original)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    axes[1].imshow(reconstructed)
    axes[1].set_title(f'Reconstructed Image (Epoch {epoch})')
    axes[1].axis('off')
    
    plt.show()

    # Save checkpoint function
def save_checkpoint(state, filename):
    with open(filename, 'wb') as f:
        f.write(flax.serialization.to_bytes(state))

def load_checkpoint(filename):
    with open(filename, 'rb') as f:
        state_dict = flax.serialization.from_bytes(state, f.read())
        return state.replace(params=state_dict['params'], opt_state=state_dict['opt_state'])

# %% Training loop
rng = jax.random.PRNGKey(0)
state = create_train_state(rng, learning_rate=0.001)

for epoch in range(100):
    state, loss = train_step(state, levels_loaded)
    print(f'Epoch {epoch}, Loss: {loss}')

    sample_image = levels_loaded[0]
    reconstructed_image = state.apply_fn({'params': state.params}, sample_image[None, ...])[0]
    
    if epoch % 5 == 0:
        save_checkpoint(state, f'checkpoint_epoch_{epoch}.pkl')

    visualize_reconstruction(sample_image, reconstructed_image, epoch)

# %% save the final state
save_checkpoint(state, f'final_model.pkl')

# %% Reconstruct images
reconstructed_images = state.apply_fn({'params': state.params}, levels_loaded)
# %% Display original and reconstructed images
show_level = 20
fig, axes = plt.subplots(1, 2, figsize=(10, 5))
axes[0].imshow(levels_loaded[show_level])
axes[0].set_title('Original Image')
axes[0].axis('off')

axes[1].imshow(reconstructed_images[show_level])
axes[1].set_title(f'Reconstructed Image (NUM: {show_level})')
axes[1].axis('off')

plt.show()

# %% Generate new levels
num_samples = 10  # Number of new levels to generate

latent_dim = 20 * 20 * 128  # This should match the flattened output size

rng = jax.random.PRNGKey(42)
latent_samples = jax.random.normal(rng, (num_samples, latent_dim))

generated_levels = state.apply_fn({'params': state.params}, latent_samples, method=Autoencoder.decode)

for i, level in enumerate(generated_levels):
    plt.imshow(jnp.clip(level, 0, 1))
    plt.title(f"Generated Level {i+1}")
    plt.axis('off')
    plt.show()