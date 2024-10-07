# %% Import
import random
from ut import load


# %% Run
data = load()

print(f"Input shape: {data}")


X = data['X']
y = data['y']
input_shape = data['input_shape']
num_classes = data['num_classes']

print(f"Input shape: {input_shape}")
print(f"Number of samples: {X.shape[0]}")
print(f"Number of classes: {num_classes}")
# %%
