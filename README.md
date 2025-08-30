# Plastic Detection with Deep Learning  
**Theme:** Environmental Monitoring & Pollution Control  

## Week 1 Milestone (Data Preprocessing - 30%)  
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
