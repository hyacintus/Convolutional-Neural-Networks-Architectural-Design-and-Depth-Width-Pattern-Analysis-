# README - CNN Medical Image Training

Overview
This Python script trains Convolutional Neural Networks (CNNs)
on multiple medical image datasets. It automatically loads images,
preprocesses them, splits into training, validation, and test sets,
and evaluates the models on multiple metrics (accuracy, F1 score,
AUC, confusion matrix, ROC curve, etc.). Results are saved in CSV files.

# Dataset Download
Please download the following datasets from Kaggle:

CBIS-DDSM-H: https://kaggle.com/datasets/9cef8368a7037c7d63324c73fa6384c0e6f9d02e10c2b07a0c37b640dbf17c45
Chest-Xray-2017_V2: https://kaggle.com/datasets/ce5259e7e743223ff2888759297abd27d388cd59af68e7a55ffb25ce8352ab19
Chest-XRay-2018_V3: https://kaggle.com/datasets/c0273ec00430e8d06a1c6212c991aa298138954cd6f01a91c746a609bfcf8b4c
Brain Tumor Dataset H: https://kaggle.com/datasets/4e5596da6b1ebfa9044ac3ddf2a424d12968f6ee193ec7d3b02aa31b7247d5df
ISIC-2017-H: https://kaggle.com/datasets/4e83433e33e4b0eb6fa18f9ad6956a3340d317ac4156d9538ed95704a0db2b62

# Folder Structure
1. Create a folder called 'dataset' in the same location as the Python script.
2. Place all downloaded datasets inside the 'dataset' folder.
   Make sure the paths in the script match the dataset locations.
3. Place the CSV file 'conv_combinations.csv' in the same folder as the Python script.

# Running the Script
1. Open a terminal or your preferred Python environment.
2. Run the Python script.

The script will:
- Load and preprocess all datasets
- Train CNN models for all filter combinations specified in 'conv_combinations.csv'
- Compute metrics: accuracy, F1 score, recall, specificity, AUC, confusion matrix,
  ROC curves, and per-epoch training/validation accuracy and loss
- Save results in CSV files named '<dataset_name>_all_experiments_results.csv'

# Notes
- If you encounter errors, check that the column headers in each dataset's classification CSV
  match those expected by the script ('image_id' and 'class_label').
- No GPU is required, but training will be faster if a GPU is available.
