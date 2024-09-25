# %% import
import gym
import gym_sokoban
import numpy as np
import matplotlib.pyplot as plt
import optax
import pickle
import jax
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
level_index = 700

print("Original Min pixel value:", jnp.min(levels_loaded[8]))
print("Original Max pixel value:", jnp.max(levels_loaded[8]))

levels_loaded = jnp.array(levels_loaded).astype(jnp.float32) / 242.0
print("Level shape: ",levels_loaded[level_index].shape)
print("Dataset shape: ", levels_loaded.shape)

print("New Min pixel value:", jnp.min(levels_loaded[8]))
print("New Max pixel value:", jnp.max(levels_loaded[8]))


plt.imshow(levels_loaded[level_index])
plt.title("Sokoban Level")
plt.axis('off')
plt.show()



# %% Define the Encoder and Decoder -------------------------------------
class Encoder(nn.Module):
    latent_dim: int

    @nn.compact
    def __call__(self, x):
        x = nn.Conv(32, (3, 3), strides=(2, 2))(x)
        x = nn.relu(x)
        x = nn.Conv(64, (3, 3), strides=(2, 2))(x)
        x = nn.relu(x)
        x = nn.Conv(128, (3, 3), strides=(2, 2))(x)
        x = nn.relu(x)
        x = x.reshape((x.shape[0], -1))  # Flatten
        x = nn.Dense(self.latent_dim)(x)
        return x

class Decoder(nn.Module):
    output_shape: tuple

    @nn.compact
    def __call__(self, z):
        x = nn.Dense(128 * 20 * 20)(z)
        x = nn.relu(x)
        x = x.reshape((-1, 20, 20, 128))
        x = nn.ConvTranspose(64, (3, 3), strides=(2, 2))(x)
        x = nn.relu(x)
        x = nn.ConvTranspose(32, (3, 3), strides=(2, 2))(x)
        x = nn.relu(x)
        x = nn.ConvTranspose(self.output_shape[-1], (3, 3), strides=(2, 2), padding='SAME')(x)
        x = nn.sigmoid(x)  # Assuming normalized inputs
        return x

# %% Autoencoder
class Autoencoder(nn.Module):
    latent_dim: int
    output_shape: tuple

    def setup(self):
        self.encoder = Encoder(self.latent_dim)
        self.decoder = Decoder(self.output_shape)
    
    def __call__(self, x):
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed

def create_train_state(rng, learning_rate):
    model = Autoencoder(latent_dim=64, output_shape=(1, 160, 160, 3))
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

# %% Training loop
rng = jax.random.PRNGKey(0)
state = create_train_state(rng, learning_rate=0.001)

for epoch in range(10):  # Adjust the number of epochs as needed
    state, loss = train_step(state, levels_loaded)
    print(f'Epoch {epoch}, Loss: {loss}')

    sample_image = levels_loaded[0]
    reconstructed_image = state.apply_fn({'params': state.params}, sample_image[None, ...])[0]
    
    visualize_reconstruction(sample_image, reconstructed_image, epoch)

# %% Reconstruct images
reconstructed_images = state.apply_fn({'params': state.params}, levels_loaded)
# %% Display original and reconstructed images
fig, axes = plt.subplots(2, 10, figsize=(20, 4))
for i in range(10):
    axes[0, i].imshow(levels_loaded[i])
    axes[0, i].axis('off')
    axes[1, i].imshow(reconstructed_images[i])
    axes[1, i].axis('off')
plt.show()