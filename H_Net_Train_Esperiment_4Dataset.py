import os
import numpy as np
import tensorflow as tf
import pandas as pd
from tensorflow.keras import layers, Sequential
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import AUC
from tensorflow.keras.utils import load_img, img_to_array
from tensorflow.keras.callbacks import EarlyStopping
import sklearn.metrics as skm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
from sklearn.utils import class_weight
from sklearn.metrics import roc_curve
import pydicom
from PIL import Image
import gc
import json

import matplotlib.pyplot as plt
import time

# Check available GPUs
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # Enable dynamic GPU memory growth to avoid crashes on low VRAM cards
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"✅ GPU detected and usable: {[gpu.name for gpu in gpus]}")
    except RuntimeError as e:
        print(f"⚠️ Error setting up GPU: {e}")
else:
    print("ℹ️ No GPU available. Training will use CPU.")


# ------------------------------------------ FILE PATHS ----------------------------------------------------------
datasets = [
    {"name": "Brain-Tumor-H", "path": r"C:\Path\Dataset\Brain Tumor Dataset H"},
    {"name": "ISIC-2017-H", "path": r"C:\Path\Dataset\Dataset-ISIC-2017-H"},
    {"name": "Chest-XRay-2018_V3", "path": r"C:\Path\Dataset\Dataset_ChestXRay2017_V2"},
    {"name": "CBIS-DDSM-H", "path": r"C:\Path\Dataset\CBIS-DDSM-H"},
]

for ds in datasets:
    print(f"\n\n🟢 Processing dataset: {ds['name']}")

    base_path = ds["path"]
    images_dir = os.path.join(base_path, "Images")
    csv_path = os.path.join(base_path, "Image_info_classification.csv")

    # Read CSV
    df = pd.read_csv(csv_path)
    combi_df = pd.read_csv("conv_combinations.csv")

    # Get unique classes automatically
    class_names = sorted(df["class_label"].unique())
    num_classes = len(class_names)

    # Create class → index mapping
    class_map = {class_name: idx for idx, class_name in enumerate(class_names)}

    print("Classes found:", class_names)
    print("Number of classes:", num_classes)

    # ------------------------------------ PREPROCESSING DATASET --------------------------------------------------
    # Load and resize dataset
    img_width, img_height = 256, 256


    def load_images_from_csv(images_dir, df, class_map, img_width, img_height, dataset_name):
        images = []
        labels = []

        max_pixel_value = 0
        min_width = float('inf')
        min_height = float('inf')

        for _, row in df.iterrows():
            img_path = os.path.join(images_dir, row["image_id"])

            if not os.path.exists(img_path):
                continue

            file_ext = os.path.splitext(img_path)[1].lower()

            # ==========================================================
            # 🔴 1️⃣ SPECIAL CASE: Chest-XRay-2018_V3 + JPEG
            # ==========================================================
            if dataset_name == "Chest-XRay-2018_V3" and file_ext in [".jpeg", ".jpg"]:

                with Image.open(img_path) as im:
                    im = im.convert("L")
                    im_resized = im.resize((img_width, img_height), Image.BILINEAR)
                    img_array = np.array(im_resized, dtype=np.float32)
                    img_array = np.expand_dims(img_array, axis=-1)

                    h, w = img_array.shape[:2]
                    min_width = min(min_width, w)
                    min_height = min(min_height, h)
                    max_pixel_value = max(max_pixel_value, np.max(img_array))

            # ==========================================================
            # 🔵 2️⃣ SPECIAL CASE: CBIS-DDSM-H + DICOM
            # ==========================================================
            elif dataset_name == "CBIS-DDSM-H" and file_ext == ".dcm":

                dicom_data = pydicom.dcmread(img_path)
                image = dicom_data.pixel_array

                h, w = image.shape
                min_width = min(min_width, w)
                min_height = min(min_height, h)
                max_pixel_value = max(max_pixel_value, np.max(image))

                image_pil = Image.fromarray(image)
                image_resized = image_pil.resize((img_width, img_height), Image.BILINEAR)
                img_array = np.array(image_resized, dtype=image.dtype)

                if img_array.ndim == 2:
                    img_array = np.expand_dims(img_array, axis=-1)

            # ==========================================================
            # 🟣 3️⃣ SPECIAL CASE: Brain-Tumor-H + TIF
            # ==========================================================
            elif dataset_name == "Brain-Tumor-H" and file_ext in [".tif", ".tiff"]:

                with Image.open(img_path) as im:
                    w, h = im.size
                    min_width = min(min_width, w)
                    min_height = min(min_height, h)
                    img_array_original = np.array(im)
                    max_pixel_value = max(max_pixel_value, np.max(img_array_original))

                img = load_img(img_path, color_mode="rgb", target_size=(img_height, img_width))
                img_array = img_to_array(img).astype(np.float32)

            # ==========================================================
            # 🟠 4️⃣ SPECIAL CASE: ISIC-2017-H + JPG
            # ==========================================================
            elif dataset_name == "ISIC-2017-H" and file_ext in [".jpg", ".jpeg"]:

                img = load_img(img_path, color_mode="rgb", target_size=(img_height, img_width))
                img_array = img_to_array(img).astype(np.float32)

                h, w = img_array.shape[:2]
                min_width = min(min_width, w)
                min_height = min(min_height, h)
                max_pixel_value = max(max_pixel_value, np.max(img_array))

            images.append(img_array)
            labels.append(class_map[row["class_label"]])

        images = np.array(images, dtype=np.float32)
        labels = np.array(labels, dtype=np.int32)
        num_channels = images.shape[-1]

        return images, labels, max_pixel_value, min_width, min_height, num_channels


    images, labels, max_pixel_value, min_width, min_height, num_channels = load_images_from_csv(
        images_dir, df, class_map, img_width, img_height, ds["name"]
    )

    print("Image data type:", images.dtype)
    print("Actual max pixel value:", np.max(images))

    # -------------------------------- NORMALIZZAZIONE AUTOMATICA --------------------------------

    if max_pixel_value <= 255:
        normalization_factor = 255.0
        print("📏 Dataset rilevato come 8-bit → Normalizzazione con 255")
    else:
        normalization_factor = 65535.0
        print("📏 Dataset rilevato come 16-bit → Normalizzazione con 65535")

    print(f"\n🌟 Valore massimo di pixel: {max_pixel_value} 📏")
    print(f"📐 Dimensioni minime - Larghezza: {min_width}, Altezza: {min_height}, Canali: {num_channels} 🖼️")
    print(
        f"🔄 Ridimensionamento delle immagini a {img_width} x {img_height} pixel (Larghezza x Altezza) come input per la CNN 📊")


    # ------------------------------------ TRAINING E TEST SET -------------------------------------------------------------
    # First split: separate 80% train+val and 20% test
    train_val_images, test_images, train_val_labels, test_labels = train_test_split(
        images, labels, test_size=0.2, shuffle=True, stratify=labels, random_state=42
    )

    # Second split: separate 70% train and 10% val from the remaining 80%
    train_images, val_images, train_labels, val_labels = train_test_split(
        train_val_images, train_val_labels, test_size=0.125,  
        shuffle=True, stratify=train_val_labels, random_state=42

    # Reshape labels
    train_labels = train_labels.reshape(-1)
    val_labels = val_labels.reshape(-1)
    test_labels = test_labels.reshape(-1)

    # ---------------------------------------------- MEMORY CLEANUP --------------------------------------------------------
    del images, labels

    # -------------------------------- DATASET CONFIGURATION FOR PERFORMANCE -----------------------------------------------
    # Define batch size
    batch_size = 16  
    epochs = 100

    # Training dataset with shuffle, batching, and caching
    train_ds = tf.data.Dataset.from_tensor_slices((train_images, train_labels))
    train_ds = train_ds.shuffle(1000).batch(batch_size).cache().prefetch(tf.data.AUTOTUNE)

    # Validation dataset: batching and caching only
    val_ds = tf.data.Dataset.from_tensor_slices((val_images, val_labels))
    val_ds = val_ds.batch(batch_size).cache().prefetch(tf.data.AUTOTUNE)

    # Test dataset: batching and caching only
    test_ds = tf.data.Dataset.from_tensor_slices((test_images, test_labels))
    test_ds = test_ds.batch(batch_size).cache().prefetch(tf.data.AUTOTUNE)

    # ------------------------------------------ MEMORY CLEANUP  -----------------------------------------------------------
    del train_images, train_labels, test_images, test_labels

    # -------------------------------- CREATE CONVOLUTIONAL NEURAL NETWORK MODEL -------------------------------------------
    def create_cnn_model(conv_filters, input_shape=(img_height, img_width, num_channels), num_classes=num_classes,
                         normalization_factor=255.0):
        """
        conv_filters: list of 3 values [Conv1, Conv2, Conv3]
        if a value is 0, the corresponding block is not added
        """
        model_layers = [layers.Input(shape=input_shape),
                        layers.Rescaling(1.0 / normalization_factor)]

        # First block
        if conv_filters[0] > 0:
            model_layers += [
                layers.Conv2D(conv_filters[0], (3, 3), padding='same', activation=None),
                layers.BatchNormalization(),
                layers.Activation('relu'),
                layers.MaxPooling2D((2, 2))
            ]

        # Second block
        if conv_filters[1] > 0:
            model_layers += [
                layers.Conv2D(conv_filters[1], (3, 3), padding='same', activation=None),
                layers.BatchNormalization(),
                layers.Activation('relu'),
                layers.MaxPooling2D((2, 2))
            ]

        # Third block
        if conv_filters[2] > 0:
            model_layers += [
                layers.Conv2D(conv_filters[2], (3, 3), padding='same', activation=None),
                layers.BatchNormalization(),
                layers.Activation('relu'),
                layers.MaxPooling2D((2, 2))
            ]

        # Flatten + Fully connected layers
        model_layers += [
            layers.GlobalAveragePooling2D(),    # layers.Flatten(),
            layers.Dense(256, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(num_classes, activation='softmax')
        ]

        model = Sequential(model_layers)

        optimizer = Adam(learning_rate=1e-3)
        model.compile(optimizer=optimizer,
                      loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                      metrics=['accuracy'])

        return model

    # -------------------------------------  MODEL TRAINING --------------------------------------------------------
    # Output CSV file name
    output_file = f"{ds['name']}_all_experiments_results.csv"

    # ==================== LOOP FOR EACH FILTER COMBINATION ====================
    for idx, row in combi_df.iterrows():
        conv_filters = [int(row['Conv1']), int(row['Conv2']), int(row['Conv3'])]
        print(f"\n🟢 Simulazione {idx+1}: filtri = {conv_filters}")

        # Create the model
        model = create_cnn_model(conv_filters, input_shape=(img_height, img_width, num_channels),
                                 num_classes=num_classes, normalization_factor=normalization_factor)

        # Early stopping to prevent overfitting
        early_stop = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)

        start_time = time.time()

        # Training
        history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=epochs,
            verbose=1,
            callbacks=[early_stop]
            # class_weight=class_weights  # Uncomment if using class weights for balance
        )

        training_time = round(time.time() - start_time, 2)

        # Predictions on test set
        y_pred_probs = model.predict(test_ds)
        y_pred = np.argmax(y_pred_probs, axis=1)
        y_true = np.concatenate([y for x, y in test_ds], axis=0)

        # Compute metrics
        cm = skm.confusion_matrix(y_true, y_pred)
        TP = np.diag(cm).astype(float)
        FP = (cm.sum(axis=0) - TP).astype(float)
        FN = (cm.sum(axis=1) - TP).astype(float)
        TN = (cm.sum() - (TP + FP + FN)).astype(float)
        Sensitivity = np.divide(TP, TP + FN, out=np.zeros_like(TP), where=(TP + FN) != 0)
        Specificity = np.divide(TN, TN + FP, out=np.zeros_like(TN), where=(TN + FP) != 0)
        Accuracy_class = np.divide(TP + TN, TP + TN + FP + FN, out=np.zeros_like(TP), where=(TP + TN + FP + FN) != 0)
        Precision = np.divide(TP, TP + FP, out=np.zeros_like(TP), where=(TP + FP) != 0)
        FRP = np.divide(FP, FP + TN, out=np.zeros_like(FP), where=(FP + TN) != 0)
        F1_score = np.divide(2 * (Precision * Sensitivity), (Precision + Sensitivity), out=np.zeros_like(Precision),
            where=(Precision + Sensitivity) != 0)
        kappa = skm.cohen_kappa_score(y_true, y_pred)
        final_train_loss = history.history['loss'][-1]
        final_val_loss = history.history['val_loss'][-1]
        final_train_acc = history.history['accuracy'][-1]
        final_val_acc = history.history['val_accuracy'][-1]

        # Compute AUC
        unique_classes = np.unique(y_true)
        if len(unique_classes) == 2:
            auc_score = skm.roc_auc_score(y_true, y_pred_probs[:, 1])
        else:
            y_true_bin = label_binarize(y_true, classes=range(num_classes))
            auc_score = skm.roc_auc_score(y_true_bin, y_pred_probs, average='macro', multi_class='ovr')

        # Save results in dict
        final_dict = {
            'Conv1': conv_filters[0],
            'Conv2': conv_filters[1],
            'Conv3': conv_filters[2],
            'TP_avg': np.mean(TP),
            'TN_avg': np.mean(TN),
            'FP_avg': np.mean(FP),
            'FN_avg': np.mean(FN),
            'Recall_avg': np.mean(Sensitivity),
            'Specificity_avg': np.mean(Specificity),
            'Accuracy_avg': np.mean(Accuracy_class),
            'Precision_avg': np.mean(Precision),
            'F1_score_avg': np.mean(F1_score),
            'Kappa': kappa,
            'Final_Train_Loss': final_train_loss,
            'Final_Val_Loss': final_val_loss,
            'Final_Train_Accuracy': final_train_acc,
            'Final_Val_Accuracy': final_val_acc,
            'AUC': auc_score,
            'Training_Time_sec': training_time
        }

        # 2️⃣ RAW METRICS (per class) compact
        metrics_per_class = {
            'TP_per_class': [TP.tolist()],
            'TN_per_class': [TN.tolist()],
            'FP_per_class': [FP.tolist()],
            'FN_per_class': [FN.tolist()],
            'Recall_per_class': [Sensitivity.tolist()],
            'Specificity_per_class': [Specificity.tolist()],
            'Accuracy_per_class': [Accuracy_class.tolist()],
            'Precision_per_class': [Precision.tolist()],
            'F1_score_per_class': [F1_score.tolist()],
            'FRP_per_class': [FRP.tolist()]
        }

        # Add to the final dictionary
        final_dict.update(metrics_per_class)

        # 3️⃣ ROC CURVE (only for binary or multi-class)
        roc_points = []
        if len(unique_classes) == 2:
            fpr, tpr, thresholds = roc_curve(y_true, y_pred_probs[:, 1])
            roc_points.append(list(zip(fpr, tpr)))  
        else:
            y_true_bin = label_binarize(y_true, classes=range(num_classes))
            for c in range(num_classes):
                fpr, tpr, thresholds = roc_curve(y_true_bin[:, c], y_pred_probs[:, c])
                roc_points.append(list(zip(fpr, tpr)))

        final_dict['ROC_points'] = json.dumps(roc_points)

        # 4️⃣ TRAINING HISTORY (Accuracy + Loss per epoch)
        # Ensure a variable exists that keeps track of maximum epochs
        if 'max_epochs' not in locals():
            max_epochs = epochs  

        actual_epochs = len(history.history['accuracy'])
        if actual_epochs > max_epochs:
            for i in range(max_epochs + 1, actual_epochs + 1):
                new_cols = [
                    f'Epoch_{i}_Train_Acc',
                    f'Epoch_{i}_Val_Acc',
                    f'Epoch_{i}_Train_Loss',
                    f'Epoch_{i}_Val_Loss'
                ]
                for col in new_cols:
                    final_dict[col] = np.nan  # initialize to NaN
            max_epochs = actual_epochs  

        for i in range(len(history.history['accuracy'])):
            final_dict[f'Epoch_{i + 1}_Train_Acc'] = history.history['accuracy'][i]
            final_dict[f'Epoch_{i + 1}_Val_Acc'] = history.history['val_accuracy'][i]
            final_dict[f'Epoch_{i + 1}_Train_Loss'] = history.history['loss'][i]
            final_dict[f'Epoch_{i + 1}_Val_Loss'] = history.history['val_loss'][i]


        # Save to CSV (append if file already exists)
        df_final = pd.DataFrame([final_dict])
        if os.path.exists(output_file):
            df_final.to_csv(output_file, mode='a', header=False, index=False)
        else:
            df_final.to_csv(output_file, index=False)

        print(f"✅ Simulazione {idx+1} salvata in CSV")

        # Memory cleanup
        del model
        tf.keras.backend.clear_session()

    # ---------- Complete memory cleanup for current dataset ----------
    del train_ds, val_ds, test_ds  
    tf.keras.backend.clear_session()  

    gc.collect()  # force garbage collection


