# Momenta - Audio Deepfake Detection Take-Home Assessment

This repository documents my approach and results for the Audio Deepfake Detection take-home assessment. The task involves detecting synthetic or deepfake audio using state-of-the-art models and techniques. Below is a breakdown of the steps taken, model selection, and the final results.

## Part 1: Research & Selection

### GitHub Repository Review

I reviewed the [Audio Deepfake Detection GitHub Repository](https://github.com/media-sec-lab/Audio-Deepfake-Detection), which is a curated collection of papers and resources focused on detecting synthetic/deepfake audio. The repository categorizes key resources into the following:

- **Research Papers**: Includes papers from top publishers (e.g., IEEE) covering synthetic speech detection, including ASVspoof challenges and generalized deepfake audio detection.
- **Datasets**: The repository contains datasets like AVSspoof and SpoofCeleb.
- **Detection Methods**:
  - Traditional Approaches: LFCC, Spectrogram Analysis
  - Deep Learning: CNN, LSTM, Transformer
  - End-to-End Models: ResNet-based classifiers

### Conclusion

This repository provides valuable insights for anyone working in the field of audio deepfake detection. It is an essential resource for literature review and surveying SOTA methods.

## Part 2: Model Selection

I identified three models that show the most promise for our specific use case, which involves detecting AI-generated human speech, especially for real-time or near-real-time detection in real conversations. I have used Equal-Error-Rate (EER) and Tandem Detection Cost Function (t-DCF) for model selection.

### EER (Equal-Error-Rate):
EER measures the threshold at which the False Acceptance (FA) and False Rejection (FR) rates are equal, helping evaluate the model's ability to distinguish real and AI-generated speech.

### t-DCF (Tandem Detection Cost Function):
This metric combines spoof detection and speaker verification errors, making it crucial for real-world deployments where both identity and speech generation matter.

### Models Selected:

1. **Model 1: Dual-Branch Network**
   - **Key Technical Innovation**: Dual-branch architecture with one branch focusing on LFCC and the other on CQT.
   - **Performance Metrics**: 
     - EER: LA: 0.80
     - t-DCF: LA: 0.021
   - **Strengths**: Lightweight, robust to background noise, effective at detecting voice cloning.
   - **Limitations**: Not tested on replay scenarios, requires large-scale training data, increased parameters need quantization for edge deployment.

2. **Model 2: ResMax**
   - **Key Technical Innovation**: Residual network with skip connections and max feature map (MFM) to highlight important audio features.
   - **Performance Metrics**:
     - EER: LA: 2.19, PA: 0.37
     - t-DCF: LA: 0.060, PA: 0.009
   - **Strengths**: Effective at spotting neural vocoder artifacts, low latency with quantization.
   - **Limitations**: Pruning required for edge deployment, struggles with high-quality voice conversion.

3. **Model 3: Voice Spoofing Countermeasure for Logical Access Attacks Detection**
   - **Key Technical Innovation**: Uses a Large Margin Cosine Loss Function and FreqAugment to improve generalization.
   - **Performance Metrics**:
     - EER: LA: 1.81
     - t-DCF: LA: 0.052
   - **Strengths**: Robust end-to-end deep learning framework, lightweight, high accuracy.
   - **Limitations**: Not tested on replay scenarios.

## Part 3: Documentation & Analysis

### Challenges Encountered

While implementing the model, I did not face major challenges since I utilized the provided GitHub repository. However, the original paper used MATLAB for LFCC, and I had to implement this in PyTorch. Additionally, I developed my own data generator for extracting LFCC, CQT, labels, and fake labels using torchaudio.

Being new to this domain, I also faced challenges in understanding terminology like EER, t-DCF, and MEL Spectrograms.

### How I Addressed These Challenges

To resolve these issues, I utilized resources like StackOverflow, DeepSeek, ChatGPT, and other online communities. While I could have implemented EER directly from the repository, I chose to use F1 score as the evaluation metric instead.

### Assumptions

I have balanced the dataset by ensuring an equal number of bonafide and spoof examples. Additionally, I considered system_id (e.g., A_01, A_02, etc.) in the spoof data and then split the dataset into training and validation sets.

### Why I Selected This Model for Implementation

The dual-branch network model was selected because it outperforms other models on Multi-task Learning-based Forgery Detection in LA (as per the GitHub repository). This model is capable of detecting not just synthetic speech, but also identifying common features across various types of synthetic speech using multitask learning. I used the ASVspoof 2019 dataset, which was also tested in the original paper.

### Model Workflow

1. **Speech Preprocessing**:
   - Raw audio is passed as input.
   - Features are extracted using LFCC (Branch 1) and CQT (Branch 2).
   
2. **Feature Extraction Module (FEM)**:
   - Modified ResNet18 with CBAM (Convolutional Block Attention Module) dynamically highlights important channels and spatial regions.
   
3. **Forgery Classification Module (FCM)**:
   - A binary classification task distinguishes between real and synthetic speech.
   
4. **Forgery Type Classification Module (FTCM)**:
   - Multi-class classification identifies spoofing methods (e.g., TTS, VC).
   - Adversarial training using GRL forces the feature extraction module to learn generic spoofing features.

### Strengths and Weaknesses

**Strengths**: Works well for directly injected synthetic speech.  
**Weaknesses**: Not evaluated on recorded speech, which may affect its performance on real-world data.

### Suggestions

I recommend using MFCC (Mel Frequency Cepstral Coefficients) instead of LFCC in combination with CQT to improve detection. MFCC captures general spectral features and low/mid-frequency characteristics, while CQT detects harmonic anomalies. Additionally, diversifying the dataset with more languages would improve the model's robustness.

## Performance Results on Chosen Dataset

I used a balanced dataset for training and validation, which led to the following results on the validation data:

- **F1 Score of Validation Data**: 0.99
- **Accuracy of Validation Data**: 0.99

## Conclusion

This model demonstrates high performance in detecting synthetic speech with an emphasis on robustness and generalization. Future work could explore expanding the dataset, including more diverse language samples, and testing the model on different types of spoofing scenarios like replay attacks.

