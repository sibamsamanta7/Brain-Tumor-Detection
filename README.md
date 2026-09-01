# Brain-Tumor-Detection
# Brain Tumor Detection

A deep learning web application that classifies brain MRI scans into four categories — **glioma**, **meningioma**, **pituitary tumor**, and **no tumor** — using a fine-tuned Xception CNN, served through a Flask web app.

## Overview

This project trains an image classification model on brain MRI scans and exposes it through a simple web interface where a user can upload an MRI image and receive a predicted diagnosis with a confidence score.

- **Model:** Transfer learning on Xception (ImageNet weights), fine-tuned on brain MRI data
- **Input size:** 299 × 299 × 3 RGB images
- **Classes:** `glioma`, `meningioma`, `pituitary`, `notumor`
- **Backend:** Flask
- **Frontend:** HTML templates (see `Templates/`)

## Repository Structure

```
Brain-Tumor-Detection/
├── MRI Images/                       # Sample / dataset MRI images
├── Templates/                        # HTML templates for the Flask app
├── Brain_Tumor_Detection.ipynb       # Model development notebook
├── brain_tumor_mri_accuracy_99.ipynb # Training notebook (~99% accuracy)
├── main.py                           # Flask application entry point
├── LICENSE                           # Apache-2.0
└── README.md
```

## Getting Started

### Prerequisites

- Python 3.10+
- pip

### Installation

```bash
git clone https://github.com/sibamsamanta7/Brain-Tumor-Detection.git
cd Brain-Tumor-Detection
python -m venv venv
source venv/bin/activate      # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

> If `requirements.txt` isn't present yet, install the core dependencies directly:
> ```bash
> pip install tensorflow flask numpy pillow pandas scikit-learn matplotlib seaborn
> ```

### Running the App

```bash
python main.py
```

Then open `http://127.0.0.1:5000` in your browser, upload an MRI image, and view the predicted class.

## Training the Model

The model is trained in `brain_tumor_mri_accuracy_99.ipynb` using the [Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset), which contains labeled `Training/` and `Testing/` folders for the four classes.

Key steps:
1. Load image paths and labels into a DataFrame
2. Preprocess and augment images with `ImageDataGenerator`
3. Fine-tune an Xception base model with a custom classification head
4. Evaluate with accuracy, precision, recall, and a confusion matrix
5. Export the trained model (`model.save('brain_tumor_model.h5')`)

**Note:** The Xception backbone expects **299×299** input images — make sure any inference code resizes images to match, or you'll hit a shape-mismatch error at prediction time.

## Results

The training notebook reports ~99% validation accuracy on the held-out test set. See `brain_tumor_mri_accuracy_99.ipynb` for the full classification report and confusion matrix.

## Disclaimer

This project is for educational and research purposes only. It is **not** a certified medical diagnostic tool and should not be used for real clinical decision-making.

## License

This project is licensed under the [Apache License 2.0](LICENSE).
