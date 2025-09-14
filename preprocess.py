import os
import cv2
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

DATASET_DIR = "dataset/raw"
OUTPUT_DIR = "dataset/processed"
IMG_SIZE = 224

os.makedirs(OUTPUT_DIR + "/train", exist_ok=True)
os.makedirs(OUTPUT_DIR + "/test", exist_ok=True)

images, labels = [], []

for label in os.listdir(DATASET_DIR):
    folder = os.path.join(DATASET_DIR, label)
    if os.path.isdir(folder):
        for file in tqdm(os.listdir(folder), desc=f"Processing {label}"):
            path = os.path.join(folder, file)
            try:
                img = cv2.imread(path)
                img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                img = img / 255.0   
                images.append(img)
                labels.append(label)
            except:
                pass


images = np.array(images, dtype="float32")

le = LabelEncoder()
labels = le.fit_transform(labels)

X_train, X_test, y_train, y_test = train_test_split(
    images, labels, test_size=0.2, stratify=labels
)

np.save(OUTPUT_DIR + "/train/images.npy", X_train)
np.save(OUTPUT_DIR + "/train/labels.npy", y_train)
np.save(OUTPUT_DIR + "/test/images.npy", X_test)
np.save(OUTPUT_DIR + "/test/labels.npy", y_test)
np.save(OUTPUT_DIR + "/classes.npy", le.classes_)   

print("Preprocessing complete. Data saved in", OUTPUT_DIR)
print("Classes:", le.classes_)
