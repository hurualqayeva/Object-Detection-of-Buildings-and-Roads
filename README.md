# 🛰️ Object Detection of Roads using U-Net

A deep learning project focused on **road segmentation from satellite imagery** using the **U-Net architecture**. This work was completed as part of the **CSCI 6364 - Machine Learning** course (Fall 2024) by **Hürü Algayeva**.

---

## 🚀 Overview

This project automates the detection of roads from high-resolution satellite images using **semantic segmentation**.
It uses the **Massachusetts Roads Dataset** from the University of Toronto and applies a **U-Net** model trained on paired satellite and map TIFF images.

---

## 🧱 Architecture

The U-Net model was designed from scratch using **TensorFlow** and **Keras**.

**Main pipeline:**

1. **DataLoader** – scrapes and downloads TIFF images from dataset URLs, handling satellite and map images separately.
2. **TiffGenerator** – efficiently loads TIFF data in batches with optional data augmentation (flip, rotate).
3. **U-Net model** – encoder-decoder architecture with skip connections and dropout regularization.
4. **Training** – monitored with `ModelCheckpoint`, `EarlyStopping`, and `ReduceLROnPlateau` callbacks.
5. **Evaluation** – computes IoU, precision, recall, and F1 score on test data.

---

## 📦 Requirements

Make sure you have Python 3.10+ and the following libraries installed:

```bash
pip install tensorflow rasterio beautifulsoup4 tqdm scikit-learn matplotlib requests
```

---

## 🗂️ Dataset

The model uses the **Massachusetts Roads Dataset**:

* Train:

  * Satellite: [train/sat](https://www.cs.toronto.edu/~vmnih/data/mass_roads/train/sat/index.html)
  * Map: [train/map](https://www.cs.toronto.edu/~vmnih/data/mass_roads/train/map/index.html)

* Validation:

  * Satellite: [valid/sat](https://www.cs.toronto.edu/~vmnih/data/mass_roads/valid/sat/index.html)
  * Map: [valid/map](https://www.cs.toronto.edu/~vmnih/data/mass_roads/valid/map/index.html)

* Test:

  * Satellite: [test/sat](https://www.cs.toronto.edu/~vmnih/data/mass_roads/test/sat/index.html)
  * Map: [test/map](https://www.cs.toronto.edu/~vmnih/data/mass_roads/test/map/index.html)

All images are automatically downloaded and organized into `data/train/`, `data/valid/`, and `data/test/`.

---

## ⚙️ Training

To train the model:

```python
train_gen = TiffGenerator(train_sat_paths, train_map_paths, batch_size=8, augment=True)
valid_gen = TiffGenerator(valid_sat_paths, valid_map_paths, batch_size=8)

model = create_unet()
model.compile(
    optimizer='adam',
    loss=combined_loss,
    metrics=['accuracy', tf.keras.metrics.IoU(num_classes=2, target_class_ids=[1])]
)

history = model.fit(train_gen, validation_data=valid_gen, epochs=20, callbacks=callbacks)
model.save('final_model.keras')
```

---

## 📈 Results

| Metric        | Score  |
| ------------- | ------ |
| **IoU**       | 0.7470 |
| **Precision** | 0.6609 |
| **Recall**    | 0.7100 |
| **F1 Score**  | 0.6846 |

Visualization example:

| Satellite Image               | Ground Truth          | Predicted Mask            |
| ----------------------------- | --------------------- | ------------------------- |
| ![satellite](example_sat.png) | ![gt](example_gt.png) | ![pred](example_pred.png) |

---

## 🧠 Model Details

* **Input size:** 384×384×3
* **Loss function:** Binary Crossentropy + Dice Loss
* **Optimizer:** Adam
* **Augmentations:** Random horizontal/vertical flips
* **Framework:** TensorFlow / Keras

---

## 🧩 File Structure

```
.
├── data/
│   ├── train/
│   ├── valid/
│   └── test/
├── DataLoader.py
├── TiffGenerator.py
├── unet_model.py
├── train.py
├── evaluate.py
├── requirements.txt
└── README.md
```

---

## ✍️ Author

**Hürü Algayeva**
📧 [huru.algayeva@gwu.edu](mailto:huru.algayeva@gwu.edu)
🔗 [LinkedIn](https://www.linkedin.com/in/hurualqayeva/)

---

Would you like me to make it Markdown-formatted and downloadable as `README.md`?
