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
plt.imshow(levels_loaded[level_index])
plt.title("Sokoban Level")
plt.axis('off')
plt.show()

print("Level shape: ",levels_loaded[level_index].shape)
# Ensure the levels are the correct shape
levels_loaded = [jnp.array(level) for level in levels_loaded]
levels_loaded = [jnp.resize(level, (160, 160, 3)) for level in levels_loaded]

# %% Autoencoder -----------------------------------------
class Encoder(nn.Module):
    latent_dim: int

    @nn.compact
    def __call__(self, x):
        x = nn.Conv(16, (3, 3), strides=(2, 2))(x)
        x = nn.relu(x)
        x = nn.Conv(32, (3, 3), strides=(2, 2))(x)
        x = nn.relu(x)
        x = nn.Conv(64, (3, 3), strides=(2, 2))(x)
        x = nn.relu(x)
        x = x.reshape((x.shape[0], -1))  # Flatten
        x = nn.Dense(self.latent_dim)(x)
        return x

class Decoder(nn.Module):
    output_shape: tuple

    @nn.compact
    def __call__(self, z):
        x = nn.Dense(64 * 20 * 20)(z)
        x = nn.relu(x)
        x = x.reshape((-1, 20, 20, 64))
        x = nn.ConvTranspose(32, (3, 3), strides=(2, 2))(x)
        x = nn.relu(x)
        x = nn.ConvTranspose(16, (3, 3), strides=(2, 2))(x)
        x = nn.relu(x)
        x = nn.ConvTranspose(self.output_shape[-1], (3, 3), strides=(2, 2), padding='SAME')(x)
        x = nn.sigmoid(x)  # Assuming normalized inputs
        return x
    
class Autoencoder(nn.Module):
    latent_dim: int
    output_shape: tuple

    def setup(self):
        self.encoder = Encoder(self.latent_dim)
        self.decoder = Decoder(self.output_shape)

    def __call__(self, x):
        z = self.encoder(x)
        x_reconstructed = self.decoder(z)
        return x_reconstructed
    
# %% Prepare training -----------------------------------------
latent_dim = 10
output_shape = (1, 160, 160, 3)

model = Autoencoder(latent_dim=latent_dim, output_shape=output_shape)

def create_train_state(rng, learning_rate):
    params = model.init(rng, jnp.ones(output_shape))['params']
    tx = optax.adam(learning_rate)
    return train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)

def compute_loss(params, batch):
    reconstructions = model.apply({'params': params}, batch)
    loss = jnp.mean((reconstructions - batch) ** 2)
    return loss

@jax.jit
def train_step(state, batch):
    grads = jax.grad(compute_loss)(state.params, batch)
    state = state.apply_gradients(grads=grads)
    return state

# %% Train -------------------------------------------------------
rng = jax.random.PRNGKey(0)
learning_rate = 0.001
state = create_train_state(rng, learning_rate)
batch_size = 32

def create_batches(data, batch_size):
    num_batches = len(data) // batch_size
    return np.array_split(data, num_batches)

batches = create_batches(levels_loaded, batch_size)

for epoch in range(10):
    for batch in batches:
        batch = jnp.array(batch)  # Ensure batch is JAX array
        state = train_step(state, batch)
    print(f"Epoch {epoch + 1} completed.")