# Plastic Detection with Deep Learning  
**Theme:** Environmental Monitoring & Pollution Control  

## Preprocessing
- Imported necessary libraries.  
- Loaded dataset from `dataset/raw/`.  
- Explored dataset using `.info()`, `.describe()`, `.isnull().sum()`.  
- Visualized sample images and checked class distribution.  
- Created preprocessing script (`preprocess.py`) to:  
  - Load and resize images (224×224).  
  - Convert labels to numpy arrays.  
  - Split into train/test sets (80/20).  
  - Save processed arrays as `.npy` files.  

## 📦 Dataset  
We are using the **Waste Classification Dataset** from Kaggle:  

🔗 [Waste Classification Dataset (Kaggle)](https://www.kaggle.com/datasets/techsash/waste-classification-data)  

The dataset consists of two classes:  
- **Plastic** → plastic bags, bottles, wrappers, etc.  
- **Non-Plastic** → metal, glass, paper, organic waste, etc.   

## Training & Testing
## 📌 Trained Model

The trained model (`plastic_detector.h5`) is too large to be stored on GitHub (limit: 100 MB).  

👉 You can download it from Google Drive:  
[Download Model from Google Drive](https://drive.google.com/drive/u/0/folders/1G1QaoTS1yJc0HZIbZhGohy3nAXeyww-6?lfhs=2)

### How to Use
1. Download the model file from the above link.  
2. Place it inside the project directory at:

