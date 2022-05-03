# Before running, install required packages:
# pip install numpy sklearn torchvision

import numpy as np
import sklearn
from sklearn.ensemble import RandomForestClassifier
from torchvision import datasets, transforms
import urllib
import zipfile

# COMMENT THIS OUT IF YOU USE YOUR OWN DATA.
# Download example data into ./data/image-data (4 image files, 2 for "dog", 2 for "cat").
url = "https://github.com/jrieke/traingenerator/raw/main/data/fake-image-data.zip"
zip_path, _ = urllib.request.urlretrieve(url)
with zipfile.ZipFile(zip_path, "r") as f:
    f.extractall("data")


# ----------------------------------- Setup -----------------------------------
# INSERT YOUR DATA HERE
# Expected format: One folder per class, e.g.
# train
# --- dogs
# |   +-- lassie.jpg
# |   +-- komissar-rex.png
# --- cats
# |   +-- garfield.png
# |   +-- smelly-cat.png
#
# Example: https://github.com/jrieke/traingenerator/tree/main/data/image-data
train_data = "data/image-data"  # required
val_data = "data/image-data"    # optional
test_data = None                # optional


# ------------------------------- Preprocessing -------------------------------
# Set up scaler.
scaler = sklearn.preprocessing.StandardScaler()

def preprocess(data, name):
    if data is None:  # val/test can be empty
        return None
    # Read image files to pytorch dataset (only temporary).
    transform = transforms.Compose([
        transforms.Resize(28), 
        transforms.CenterCrop(28), 
        transforms.ToTensor()
    ])
    data = datasets.ImageFolder(data, transform=transform)

    # Convert to numpy arrays.
    images_shape = (len(data), *data[0][0].shape)
    images = np.zeros(images_shape)
    labels = np.zeros(len(data))
    for i, (image, label) in enumerate(data):
        images[i] = image
        labels[i] = label

    # Flatten.
    images = images.reshape(len(images), -1)

    # Scale to mean 0 and std 1.
    if name == "train":
        scaler.fit(images)
    images = scaler.transform(images)

    # Shuffle train set.
    if name == "train":
        images, labels = sklearn.utils.shuffle(images, labels)

    return [images, labels]

processed_train_data = preprocess(train_data, "train")
processed_val_data = preprocess(val_data, "val")
processed_test_data = preprocess(test_data, "test")


# ----------------------------------- Model -----------------------------------
model = RandomForestClassifier()


# --------------------------------- Training ----------------------------------
def evaluate(data, name):
    if data is None:  # val/test can be empty
        return

    images, labels = data
    acc = model.score(images, labels)
    print(f"{name + ':':6} accuracy: {acc}")

# Train on train_data.
model.fit(*processed_train_data)

# Evaluate on all datasets.
evaluate(processed_train_data, "train")
evaluate(processed_val_data, "val")
evaluate(processed_test_data, "test")