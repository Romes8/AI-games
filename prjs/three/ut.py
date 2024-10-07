# %% Run
import os
import cv2
import numpy as np
import jax.numpy as jnp
from jax import random
import matplotlib.pyplot as plt

# %% Run
def load():
    dataset_path = 'C:\\Transfer\\Dokumenty\\DANSKO\\ITU\\Games\\3rdSemester\\AI\\AI-games\\prjs\\three\\fingers'
    
    images = []
    labels = []

    print("Processing images...")

    for filename in os.listdir(dataset_path):
        if filename.endswith(('.png')):  # Add other extensions if needed
            # Read the image
            img_path = os.path.join(dataset_path, filename)
            img = cv2.imread(img_path)
            
            # Resize the image (optional, but recommended for consistency)
            img = cv2.resize(img, (128, 128))  # You can adjust the size as needed
            
            # Convert BGR to RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Extract the label from the filename
            label = filename[-6:-4]
            
            # Add the image and label to our lists
            images.append(img)
            labels.append(label)

    X = jnp.array(images)
    X = X.astype('float32') / 255.0

    unique_labels = sorted(set(labels))
    label_to_index = {label: index for index, label in enumerate(unique_labels)}
    y = jnp.array([label_to_index[label] for label in labels])

    # Create a reverse mapping from index to label
    index_to_label = {index: label for label, index in label_to_index.items()}

    # Choose a random image to visualize
    key = random.PRNGKey(0)
    random_index = random.randint(key, (), 0, len(X))

    # Get the image and label
    image = X[random_index]
    label = y[random_index]

    # Visualize the image with its label
    plt.figure(figsize=(8, 8))
    plt.imshow(image)
    plt.axis('off')
    plt.title(f"Label: {index_to_label[int(label)]}")  # Convert label to int
    plt.show()

    # Print dataset information
    print(f"Dataset shape: {X.shape}")
    print(f"Number of classes: {len(unique_labels)}")

    return X, y, unique_labels
# %%
