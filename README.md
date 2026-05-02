# MAI 604 - Speech Emotion Recognition 🎙️🧠

## Project Overview
This project presents a comprehensive Deep Learning pipeline for **Speech Emotion Recognition (SER)** using the **RAVDESS** (Ryerson Audio-Visual Database of Emotional Speech and Song) dataset. Human emotion recognition from speech is a critical component in developing intuitive human-computer interaction (HCI) systems, with applications ranging from virtual assistants and customer service analysis to mental health monitoring. 

The primary objective of this project is to build, evaluate, and compare multiple neural network architectures to determine the most effective approach for classifying eight distinct human emotions from raw audio signals. The pipeline addresses the inherent challenges of acoustic modeling—such as the temporal dynamics of speech, varying acoustic energy, and dataset size constraints—by implementing robust feature extraction (MFCCs, Mel Spectrograms, Chroma) and data augmentation techniques (time-shifting, noise injection).

Specifically, this study provides an in-depth comparative analysis between different deep learning paradigms:
- **1D-CNNs (Convolutional Neural Networks):** Evaluated at different depths (a shallow baseline vs. a deeper model with batch normalization) to assess spatial feature extraction from acoustic representations.
- **BiLSTMs (Bidirectional Long Short-Term Memory):** Implemented to capture the sequential, long-range dependencies and temporal context of spoken statements.

By rigorously benchmarking these models and performing out-of-distribution cross-dataset testing on the **CREMA-D** dataset, this project highlights the trade-offs between architectural complexity, training stability, and classification accuracy in modern audio-based deep learning.

## Academic Context
**Course:** MAI 604 Deep Learning  
**Professor:** Dr. Firuz  
**Team Members & Contributions:** 
- **Farnaz Azarparand:** Data Preparation & LSTM Model Architecture
- **Rajeev Pandey:** Feature Extraction & Baseline CNN-A Architecture
- **Hussein Aboueldahab:** Data Augmentation & Advanced CNN-B Architecture

## Dataset
- **Name:** [RAVDESS Emotional Speech Audio](https://www.kaggle.com/datasets/uwrfkaggler/ravdess-emotional-speech-audio)
- **Description:** The dataset contains 2,880 audio files from 24 professional actors (12 female, 12 male), vocalizing lexically-matched statements. It includes 8 emotion classes: neutral, calm, happy, sad, angry, fearful, disgust, and surprised. 

## Detailed Methodology & Process Flow
The project is structured into an 11-step pipeline, fully documented and executable within the main Jupyter Notebook (`MAI604_Group_Project_SER_Pipeline.ipynb`):

### 1. Setup & Imports
Configuration of the environment, establishing Google Drive mounting for dataset access, and importing necessary deep learning and audio processing libraries (TensorFlow, Librosa, etc.).

### 2. Data Loading
Traversing the RAVDESS directory to parse filenames. The metadata is embedded in the filenames themselves (emotion, intensity, actor ID, gender), which is extracted to build a structured Pandas DataFrame.

### 3. Exploratory Data Analysis (EDA)
In-depth diagnostic visualizations to build intuition about the audio signals:
- **Class & Gender Distribution:** Ensuring the dataset is balanced across the 8 emotions and genders.
- **Waveplots:** Visualizing raw amplitude over time to identify energy patterns (e.g., flat low-energy for neutral, sustained high-amplitude for angry).
- **Mel Spectrograms:** Decomposing signals into frequency bands on the Mel scale to see frequency content over time.
- **MFCCs & Chroma:** Visualizing Mel-Frequency Cepstral Coefficients (the shape of the vocal tract) and Chroma features (12 pitch classes correlating with harmonic content).

### 4. Feature Extraction
Extracting vital acoustic features from the raw audio. The primary feature used is **MFCCs** (40 coefficients). We compute the Mel spectrogram, apply a log transform, and take the Discrete Cosine Transform. The coefficients are averaged across all time frames to produce a 40-dimensional summary vector per recording.

### 5. Data Augmentation
To simulate real-world recording conditions and prevent model overfitting, augmentation is applied *before* the train/test split:
- **Noise Injection:** Adding Gaussian white noise.
- **Time Shifting:** Rolling the signal in time to make the model onset-invariant.

### 6. Data Preparation
Splitting the augmented dataset into training and testing sets. Features are scaled using standard scaling, reshaped into 3D arrays to feed into the neural networks, and emotion labels are one-hot encoded.

### 7. Model Architectures
Three distinct architectures are implemented and compared:
- **CNN Model A (Shallow Baseline):** A straightforward 3-block 1D-CNN with MaxPooling and GlobalAveragePooling. Acts as an interpretable baseline (Expected accuracy: 72–80%).
- **CNN Model B (Deep with Batch Normalization):** A deeper 4-block architecture utilizing Batch Normalization and Dropout to stabilize training on a small dataset. It avoids MaxPooling on the last block to preserve spatial features (Expected accuracy: 82–90%).
- **LSTM Model (Bidirectional Stacked LSTM):** Utilizes 2 BiLSTM layers to process sequences forward and backward, capturing temporal dynamics and future context (e.g., falling intonations at the end of a sentence) (Expected accuracy: 75–85%).

### 8. Training
Training the models utilizing Keras callbacks such as `EarlyStopping` (to halt training when validation loss stops improving) and `ReduceLROnPlateau` (to dynamically adjust the learning rate).

### 9. Evaluation & Comparison
Assessing model performance on the test set using accuracy, F1-scores, and comprehensive classification reports. Confusion matrices are plotted to analyze misclassifications between acoustically similar emotions (e.g., angry vs. happy).

### 10. Cross-Dataset Testing
Evaluating the trained models on a random sample from the **CREMA-D** dataset to test out-of-distribution generalizability.

### 11. Live Demo (Bonus)
A record-and-predict interactive widget designed for presentations, allowing users to record live audio and feed it through the RAVDESS-style preprocessing pipeline to get real-time emotion predictions from the CNN-B and LSTM models.

## How to Run
1. Clone this repository to your local machine or mount it in your Google Drive.
2. Open `MAI604_Group_Project_SER_Pipeline.ipynb` using Google Colab or Jupyter Notebook.
3. The notebook is configured to automatically download the RAVDESS dataset via `kagglehub` if it is not already cached.
4. Execute the cells sequentially from top to bottom.

## Technologies Used
- **Python 3.10**
- **TensorFlow & Keras:** Deep learning model building and training.
- **Librosa:** Audio signal processing and feature extraction.
- **Scikit-learn:** Data preprocessing and evaluation.
- **Pandas, NumPy, Matplotlib, Seaborn:** Data manipulation and visual diagnostics.

---
*This repository was developed as a collaborative group project for the MAI 604 Deep Learning course.*
