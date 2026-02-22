# ⚕️ Multimodal Skin Lesion Clinical Decision Support System (CDSS)

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15+-orange.svg)](https://tensorflow.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Live_App-red.svg)](https://skin-lesion-detection.streamlit.app/) **Live Web Application:** [Click here to view the deployed CDSS dashboard](https://legiondetection.streamlit.app/)

## 🚀 Project Overview
This project is a clinical-grade Multimodal Artificial Intelligence tool designed to assist dermatologists in diagnosing skin lesions. Instead of relying solely on image pixels like traditional CNNs, this Y-Shaped Neural Network mimics real-world clinical diagnosis by fusing **Dermatoscopic Imagery** with **Patient Metadata** (Age, Biological Sex, and Anatomical Site).

<img width="1310" height="876" alt="Screenshot 2026-02-22 141259" src="https://github.com/user-attachments/assets/21b34ea6-60b3-43ee-9687-ff8917769ba1" />


## 🧠 Core Features & Engineering
* **Y-Shaped Multimodal Architecture:** Fuses 1,280 visual features (extracted via EfficientNetB0) with 32 statistical risk features (processed via customized Dense layers).
* **Explainable AI (XAI):** Utilizes Activation Saliency Mapping to generate real-time "X-Ray" heatmaps, allowing clinicians to visually verify the focal points of the AI's decision.
* **Clinical Safety Net:** Hardcoded 65% Confidence Threshold. If the model exhibits high uncertainty (e.g., struggling between a Benign Mole and Melanoma), the UI automatically flags the diagnosis for immediate human review to prevent False Negatives.

## 📊 The Dataset & Preprocessing
**Source:** [ISIC 2019 Challenge Dataset](https://challenge.isic-archive.com/data/) (International Skin Imaging Collaboration)

Dealing with real-world medical data required rigorous preprocessing to prevent AI hallucinations and phantom accuracy:
1. **Handling Missing Data:** Missing patient ages were imputed using the dataset's median (scaled between 0-1 to match pixel weights). Missing categorical text was assigned an `'unknown'` class to teach the AI how to handle incomplete clinical forms.
2. **Preventing Data Leakage:** A strict **70/15/15 (Train/Val/Test) split** was enforced *before* any data augmentation. 
3. **Hybrid Balancing:** To handle severe class imbalances (thousands of NV vs. few Melanomas), a custom 8,000-image sampling strategy (oversampling minority, undersampling majority) was applied **exclusively** to the training set.

## 🏗️ Model Architecture

* **Left Brain (Vision):** `EfficientNetB0` (ImageNet weights). The bottom 100 layers are frozen for edge detection; the top layers are unfrozen for fine-tuning on skin lesions. Includes built-in spatial augmentation (Flip, Rotation, Zoom, Contrast).
* **Right Brain (Metadata):** Custom Dense network (64 -> 32 neurons) processing One-Hot Encoded demographics.
* **The Merge:** A Keras `Concatenate` layer bridges the modalities, feeding into a final 256-neuron Dense classification head with an aggressive 40% Dropout rate to prevent overfitting.

## 📈 Performance & Results
The model was evaluated strictly on the 15% unseen Test Set, achieving a **True Generalization Accuracy of 87%**.
* **Basal Cell Carcinoma (BCC):** 92% Recall
* **Melanocytic Nevus (NV):** 92% Recall
* **Melanoma (MEL):** 74% Recall. *(Note: Because distinguishing early-stage Melanoma from atypical nevi is notoriously difficult, the CDSS UI compensates for this 0.74 recall via the 65% Uncertainty Warning, ensuring doctors double-check borderline cases).*
<img width="645" height="343" alt="image" src="https://github.com/user-attachments/assets/9c3ff227-4ad1-45c8-b81b-bdcf8779c01e" />

## 💻 How to Run Locally

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/Adityabhatt1002/Skin-Lesion-Detection.git](https://github.com/Adityabhatt1002/Skin-Lesion-Detection.git)
   cd Skin-Lesion-Detection
