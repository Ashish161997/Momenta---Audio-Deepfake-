# 🎧 **Momenta - Audio Deepfake Detection ** 🎶

Welcome to the documentation of my approach and results for the **Audio Deepfake Detection** take-home assessment. The task involves detecting synthetic or deepfake audio using state-of-the-art models and techniques. Below is a breakdown of the steps taken, model selection, and the final results.

---

## 🔍 **Part 1: Research & Selection** 📚

### 🏠 **GitHub Repository Review**

I thoroughly reviewed the [**Audio Deepfake Detection GitHub Repository**](https://github.com/media-sec-lab/Audio-Deepfake-Detection), which serves as a curated collection of papers and resources dedicated to synthetic/deepfake audio detection. The repository categorizes key resources into the following:

- **📄 Research Papers**: Includes papers from top publishers (e.g., IEEE) on synthetic speech detection, including ASVspoof challenges and generalized deepfake audio detection.
- **📊 Datasets**: Datasets like AVSspoof and SpoofCeleb for testing models.
- **🛠 Detection Methods**:
  - **Traditional Approaches**: LFCC, Spectrogram Analysis
  - **Deep Learning**: CNN, LSTM, Transformer
  - **End-to-End Models**: ResNet-based classifiers

---

### 📝 **Conclusion**

This repository offers valuable insights and is an essential resource for literature review and surveying the **state-of-the-art (SOTA)** methods in audio deepfake detection.

---

## 🧠 **Part 2: Model Selection** 🏆

After careful evaluation, I selected three promising models for detecting **AI-generated human speech**, especially for **real-time or near-real-time detection** in real conversations. I used **Equal-Error-Rate (EER)** and **Tandem Detection Cost Function (t-DCF)** for model selection.

### 📊 **Evaluation Metrics**

1. **EER (Equal-Error-Rate)**:
   - Measures the threshold where False Acceptance (FA) and False Rejection (FR) rates are equal. It helps evaluate the model's ability to distinguish between real and AI-generated speech.
   
2. **t-DCF (Tandem Detection Cost Function)**:
   - Combines spoof detection and speaker verification errors, making it crucial for real-world applications.

---

### **Models Selected:**

#### 1. **Model 1: Dual-Branch Network**
   - **🔧 Key Technical Innovation**: Dual-branch architecture with LFCC and CQT.
   - **📈 Performance Metrics**:
     - EER: LA: 0.80
     - t-DCF: LA: 0.021
   - **✨ Strengths**: Lightweight, robust to background noise, effective at detecting voice cloning.
   - **⚠ Limitations**: Not tested on replay scenarios, requires large-scale training data.

#### 2. **Model 2: ResMax**
   - **🔧 Key Technical Innovation**: Residual network with skip connections and max feature map (MFM) for highlighting important features.
   - **📈 Performance Metrics**:
     - EER: LA: 2.19, PA: 0.37
     - t-DCF: LA: 0.060, PA: 0.009
   - **✨ Strengths**: Effective at spotting neural vocoder artifacts, low latency.
   - **⚠ Limitations**: Pruning required for edge deployment, struggles with high-quality voice conversion.

#### 3. **Model 3: Voice Spoofing Countermeasure**
   - **🔧 Key Technical Innovation**: Large Margin Cosine Loss Function and FreqAugment for better generalization.
   - **📈 Performance Metrics**:
     - EER: LA: 1.81
     - t-DCF: LA: 0.052
   - **✨ Strengths**: Robust, lightweight, high accuracy.
   - **⚠ Limitations**: Not tested on replay scenarios.

---

## 📝 **Part 3: Documentation & Analysis** 🔍

### 🚧 **Challenges Encountered**

While implementing the model, I did not face major challenges, as I leveraged the provided GitHub repository. However, adapting the MATLAB code for LFCC into PyTorch posed some difficulties. I also had to develop my own **data generator** for extracting LFCC, CQT, labels, and fake labels using `torchaudio`.

Being new to this domain, I initially struggled with terminology such as **EER**, **t-DCF**, and **MEL Spectrograms**.

---

### 💡 **How I Addressed These Challenges**

To resolve these issues, I turned to **StackOverflow**, **DeepSeek**, **ChatGPT**, and other online communities for support. While I could have implemented EER directly from the repository, I chose **F1 score** as the evaluation metric instead.

---

### ⚖️ **Assumptions**

I balanced the dataset by ensuring an equal number of bonafide and spoof examples. Additionally, I considered **system_id** (e.g., A_01, A_02) in the spoof data and split the dataset into training and validation sets.

---

### 🏆 **Why I Selected This Model for Implementation**

The **Dual-Branch Network** model was chosen because of its superior performance on **Multi-task Learning-based Forgery Detection** in LA (as indicated by the GitHub repository). This model can detect synthetic speech and identify common features across various types of synthetic speech, making it versatile. I used the **ASVspoof 2019 dataset**, which was also tested in the original paper.

---

### 🔄 **Model Workflow**

1. **Speech Preprocessing**: 
   - Raw audio is processed to extract features using **LFCC (Branch 1)** and **CQT (Branch 2)**.
   
2. **Feature Extraction Module (FEM)**:
   - **Modified ResNet18** with **CBAM (Convolutional Block Attention Module)** dynamically highlights important channels and regions.
   
3. **Forgery Classification Module (FCM)**:
   - A binary classification model distinguishes real vs. synthetic speech.
   
4. **Forgery Type Classification Module (FTCM)**:
   - Multi-class classification identifies spoofing methods (e.g., TTS, VC).
   - **Adversarial training** using **GRL** forces the model to learn generic spoofing features.

---

### ✅ **Strengths and Weaknesses**

- **Strengths**: Works well for directly injected synthetic speech.
- **Weaknesses**: Not tested on recorded speech, which could affect performance on real-world data.

---

### 🧠 **Suggestions**

I recommend using **MFCC (Mel Frequency Cepstral Coefficients)** instead of LFCC combined with CQT to enhance detection. **MFCC** captures general spectral features and low/mid-frequency characteristics, while **CQT** detects harmonic anomalies. Additionally, diversifying the dataset with **more languages** would improve robustness.

---

## 📊 **Performance Results on Chosen Dataset**

The model achieved the following results on the validation data:

- **F1 Score**: 0.99
- **Accuracy**: 0.99

---

## 🎯 **Conclusion**

This model demonstrates **high performance** in detecting synthetic speech, with a focus on robustness and generalization. **Future work** could explore expanding the dataset, incorporating more diverse languages, and testing the model on different spoofing scenarios, such as **replay attacks**.

---
