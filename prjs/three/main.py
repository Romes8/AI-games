# %% Import
import os
import pickle
import cv2
import jax
import numpy as np
import jax.numpy as jnp
from jax import jit, random, vmap
import matplotlib.pyplot as plt
import optax
from jax import lax

# %% Train
def load():
    dataset_path = 'C:\\Transfer\\Dokumenty\\DANSKO\\ITU\\Games\\3rdSemester\\AI\\AI-games\\prjs\\three\\fingers'
    
    images = []
    labels = []

    print("Processing images...")

    for filename in os.listdir(dataset_path):
        if filename.endswith(('R.png')):  # Add other extensions if needed
            img_path = os.path.join(dataset_path, filename)
            img = cv2.imread(img_path)
            img = cv2.resize(img, (128, 128))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            label = filename[-6:-4]
            images.append(img)
            labels.append(label)

    X = jnp.array(images)
    X = X.astype('float32') / 255.0

    unique_labels = sorted(set(labels))
    label_to_index = {label: index for index, label in enumerate(unique_labels)}
    y = jnp.array([label_to_index[label] for label in labels])

    # Split the data into training and test sets
    train_size = 3000
    X, X_test = X[:train_size], X[train_size:]
    y, y_test = y[:train_size], y[train_size:]

    index_to_label = {index: label for label, index in label_to_index.items()}

    key = random.PRNGKey(0)
    random_index = random.randint(key, (), 0, len(X))

    image = X[random_index]
    label = y[random_index]

    plt.figure(figsize=(8, 8))
    plt.imshow(image)
    plt.axis('off')
    plt.title(f"Label: {index_to_label[int(label)]}")
    plt.show()

    print(f"Dataset shape: {X.shape}")
    print(f"Number of classes: {len(unique_labels)}")

    return X, y,X_test, y_test, unique_labels

def init_params(key, input_shape, n_classes):
    k1, k2, k3, k4, k5 = random.split(key, 5)

    flattened_size = 768  # Update this value based on the output

    # Use He initialization
    def he_init(key, shape):
        return random.normal(key, shape) * jnp.sqrt(2 / shape[0])

    return {
        'conv1': he_init(k1, (32, 3, 3, 3)),
        'conv2': he_init(k2, (64, 32, 3, 3)),
        'conv3': he_init(k3, (128, 64, 3, 3)),
        'dense1': he_init(k4, (flattened_size, 512)),
        'dense2': he_init(k5, (512, n_classes)),
    }

@jit
def conv(x, w):
    return lax.conv_general_dilated(x, w, (1, 1), 'SAME', dimension_numbers=('NHWC', 'HWIO', 'NHWC'))

@jit
def dropout(x, rate, key):
    mask = random.bernoulli(key, 1 - rate, x.shape)
    return jnp.where(mask, x / (1 - rate), 0)

@jit
def model(params, x, key, train):
    dropout_key1, dropout_key2 = random.split(key)
    
    # Add a batch dimension if x is a single sample
    if x.ndim == 3:
        x = x[None, ...]
    
    x = jax.nn.relu(conv(x, params['conv1']))
    x = lax.reduce_window(x, -jnp.inf, lax.max, (1, 2, 2, 1), (1, 2, 2, 1), 'VALID')
    x = jax.nn.relu(conv(x, params['conv2']))
    x = lax.reduce_window(x, -jnp.inf, lax.max, (1, 2, 2, 1), (1, 2, 2, 1), 'VALID')
    x = jax.nn.relu(conv(x, params['conv3']))
    x = lax.reduce_window(x, -jnp.inf, lax.max, (1, 2, 2, 1), (1, 2, 2, 1), 'VALID')
    x = x.reshape((x.shape[0], -1))
    x = jax.nn.relu(jnp.dot(x, params['dense1']))
    
    def dropout_true():
        return dropout(x, 0.5, dropout_key1)
    
    def dropout_false():
        return x
    
    x = lax.cond(train, dropout_true, dropout_false)
    
    logits = jnp.dot(x, params['dense2'])
    
    def dropout_logits_true():
        return dropout(logits, 0.2, dropout_key2)
    
    def dropout_logits_false():
        return logits
    
    logits = lax.cond(train, dropout_logits_true, dropout_logits_false)
    
    # Remove the batch dimension if we added it
    if logits.shape[0] == 1:
        logits = logits[0]
    
    return logits

@jit
def loss(params, x, y, key):
    logits = model(params, x, key, train=True)
    return -jnp.mean(jax.nn.log_softmax(logits)[jnp.arange(y.shape[0]), y])

@jit
def update(params, x, y, key, opt_state):
    loss_value, grads = jax.value_and_grad(loss)(params, x, y, key)
    updates, opt_state = optimizer.update(grads, opt_state)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss_value


# Load the dataset
X, y, X_test, y_test, unique_labels = load()

# Initialize model and optimizer
learning_rate = 0.0001
num_epochs = 3


key = random.PRNGKey(0)
n_classes = len(unique_labels)
input_shape = (128, 128, 3)
params = init_params(key, input_shape, n_classes)
optimizer = optax.chain(
    optax.clip_by_global_norm(1.0),  # Gradient clipping
    optax.adam(learning_rate)
)
opt_state = optimizer.init(params)

# Prepare for GIF creation
conv_layer_frames = [[], [], []]

# Training loop
for epoch in range(num_epochs):
    start_time = time.time()
    
    key, subkey = random.split(key)
    
    params, opt_state, loss_value = update(params, X, y, subkey, opt_state)
    
    epoch_time = time.time() - start_time

    # Print the epoch information
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss_value:.4f}, Time: {epoch_time:.2f}s")    


# After training, save the model parameters
model_filename = 'trained_model_params.pkl'
X_params = 'X_params.pkl'
y_params = 'y_params.pkl'

with open(model_filename, 'wb') as f:
    pickle.dump(params, f)

with open(X_params, 'wb') as f:
    pickle.dump(X, f)

with open(y_params, 'wb') as f:
    pickle.dump(y, f)

print(f"Model parameters saved to {model_filename}")

# %% Load and Run the trained model

with open('trained_model_params.pkl', 'rb') as f:
    params = pickle.load(f)

with open('X_params.pkl', 'rb') as f:
    X = pickle.load(f)

with open('y_params.pkl', 'rb') as f:
    y = pickle.load(f)

print("X shape:", X.shape)
print("X_test shape:", X_test.shape)

# Evaluation
@jit
def evaluate(params, X, y):
    key = random.PRNGKey(1)
    logits = model(params, X, key, train=False)
    predictions = jnp.argmax(logits, axis=-1)
    return jnp.mean(predictions == y)

train_accuracy = evaluate(params, X, y)
test_accuracy = evaluate(params, X_test, y_test)
print(f"Training Accuracy: {train_accuracy:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")

def visualize_results(params, X, y, unique_labels):
    key = random.PRNGKey(1)
    logits = model(params, X, key, train=False)
    predictions = jnp.argmax(logits, axis=-1)
    
    n_samples = 20
    n_cols = 5
    n_rows = (n_samples + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 4 * n_rows))
    
    for i in range(n_samples):
        idx = random.randint(random.PRNGKey(i), (), 0, len(X))
        
        row = i // n_cols
        col = i % n_cols
        
        axes[row, col].imshow(X[idx])
        original_label = unique_labels[y[idx]]
        predicted_label = unique_labels[predictions[idx]]
        
        color = 'green' if original_label == predicted_label else 'red'
        
        axes[row, col].set_title(f"O: {original_label}\nP: {predicted_label}", color=color)
        axes[row, col].axis('off')
    
    # Remove any unused subplots
    for i in range(n_samples, n_rows * n_cols):
        row = i // n_cols
        col = i % n_cols
        fig.delaxes(axes[row, col])
    
    plt.tight_layout()
    plt.show()

visualize_results(params, X_test, y_test, unique_labels)
# %%
