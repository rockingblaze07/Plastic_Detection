import os
import numpy as np
from tqdm import tqdm
from tensorflow.keras.preprocessing import image
from sklearn.model_selection import train_test_split

# Paths
dataset_path = "dataset/raw"
classes = ["non_plastic", "plastic"]

images = []
labels = []

# Load images
for idx, cls in enumerate(classes):
    cls_path = os.path.join(dataset_path, cls)
    for img_file in tqdm(os.listdir(cls_path), desc=f"Processing {cls}"):
        img_path = os.path.join(cls_path, img_file)
        
         # Skip if it's not a file
        if not os.path.isfile(img_path):
            continue

        # Skip if extension is not image type
        if not img_file.lower().endswith((".jpg", ".jpeg", ".png")):
            continue

        img = image.load_img(img_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        images.append(img_array)
        labels.append(idx)

images = np.array(images)
labels = np.array(labels)

# Determine test size safely
num_classes = len(np.unique(labels))
num_samples = len(images)

# Ensure test set has at least one sample per class
test_size = max(int(0.2 * num_samples), num_classes)

# Use stratified split if possible
if num_samples >= 2 * num_classes:
    X_train, X_test, y_train, y_test = train_test_split(
        images, labels, test_size=test_size, stratify=labels, random_state=42
    )
else:
    # Manual split for very small datasets
    train_indices = []
    test_indices = []

    for cls_idx in np.unique(labels):
        cls_indices = np.where(labels == cls_idx)[0]
        np.random.shuffle(cls_indices)
        if len(cls_indices) >= 2:
            train_indices.append(cls_indices[0])
            test_indices.append(cls_indices[1])
        else:
            # If only 1 sample in class, assign to train
            train_indices.append(cls_indices[0])

    X_train = images[train_indices]
    y_train = labels[train_indices]
    X_test = images[test_indices] if test_indices else np.array([])
    y_test = labels[test_indices] if test_indices else np.array([])

print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train:", y_train)
print("y_test:", y_test)

# Save arrays
os.makedirs("dataset/processed/train", exist_ok=True)
os.makedirs("dataset/processed/test", exist_ok=True)

np.save("dataset/processed/train/images.npy", X_train)
np.save("dataset/processed/train/labels.npy", y_train)
np.save("dataset/processed/test/images.npy", X_test)
np.save("dataset/processed/test/labels.npy", y_test)
np.save("dataset/processed/classes.npy", np.array(classes))
