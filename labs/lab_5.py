# %% Imports
import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.training import train_state
import optax
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt


# %%Get sample images from the environment
def get_sample_images():
    env = gym.make("CarRacing-v2", render_mode="rgb_array")
    states = []
    state, _ = env.reset()
    for i in range(1500):
        action = env.action_space.sample()
        state, *_ = env.step(action)
        if i >= 500:
            states.append(state)
    return jnp.array(states).astype(jnp.float32) / 255.0  # Normalize

data = get_sample_images()

plt.imshow(data[500])
plt.show()

# %% Encoder and decoder
class Encoder(nn.Module):
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
    @nn.compact
    def __call__(self, x):
        x = x.reshape((-1, 12, 12, 128))
        x = nn.ConvTranspose(features=64, kernel_size=(3, 3), strides=(2, 2))(x)
        x = nn.relu(x)
        x = nn.ConvTranspose(features=32, kernel_size=(3, 3), strides=(2, 2))(x)
        x = nn.relu(x)
        x = nn.ConvTranspose(features=3, kernel_size=(3, 3), strides=(2, 2))(x)
        return x

# %% Autoencoder
class Autoencoder(nn.Module):
    def setup(self):
        self.encoder = Encoder()
        self.decoder = Decoder()
    
    def __call__(self, x):
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed

def create_train_state(rng, learning_rate):
    model = Autoencoder()
    variables = model.init(rng, jnp.ones((1, 96, 96, 3)))
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

rng = jax.random.PRNGKey(0)
state = create_train_state(rng, learning_rate=1e-3)

def visualize_reconstruction(original, reconstructed, epoch):
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(original)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    axes[1].imshow(reconstructed)
    axes[1].set_title(f'Reconstructed Image (Epoch {epoch})')
    axes[1].axis('off')
    
    plt.show()

# %% Actual data train
for epoch in range(50):  # Adjust the number of epochs as needed
    state, loss = train_step(state, data)
    print(f'Epoch {epoch}, Loss: {loss}')

    sample_image = data[0]
    reconstructed_image = state.apply_fn({'params': state.params}, sample_image[None, ...])[0]
    
    visualize_reconstruction(sample_image, reconstructed_image, epoch)


# %%Reconstruct images
reconstructed_images = state.apply_fn({'params': state.params}, data)

# %%Display original and reconstructed images
fig, axes = plt.subplots(2, 2, figsize=(10, 10))
for i in range(10):
    axes[0, i].imshow(data[i])
    axes[0, i].axis('off')
    axes[1, i].imshow(reconstructed_images[i])
    axes[1, i].axis('on')

plt.show()
# %% Step 3: create a deconvolutional model that maps the latent space back to an image

# %% Step 4: train the model to minimize the reconstruction error

# %% Step 5: generate some images by sampling from the latent space

# %% Step 6: visualize the images

# %% Step 7: (optional) try to interpolate between two images by interpolating between their latent representations

# %% Step 8: (optional) try to generate images that are similar to a given image by optimizing the latent representation

# %% Step 9: instead of mapping the image to a latent space, map the image to a distribution over latent spaces (VAE)

# %% Step 10: sample from the distribution over latent spaces and generate images

# %% Step 11: (optional) try to interpolate between two images by interpolating between their distributions over latent spaces

# %% Step 12: (optional) try to generate images that are similar to a given image by optimizing the distribution over latent spaces

# %% Step 13: (optional) try to switch out the VAE for a GAN
