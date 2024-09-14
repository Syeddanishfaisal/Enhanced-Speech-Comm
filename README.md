# Deep Learning for Enhanced Speech Communication

## Project Overview

This project integrates real-time voice command recognition using deep learning techniques for emotion classification. It explores the performance of two different models for classifying emotions from the TESS dataset:

1. **Random Forest Classifier**: A traditional machine learning model.
2. **Convolutional Neural Network (CNN)**: A deep learning model using 1D convolutions.

The goal is to compare these models and determine which has the best performance in classifying emotions.

## Project Structure

1. **Data Loading and Preparation**:
   - Load and preprocess audio data from the TESS dataset.
   - Extract features such as MFCCs for training and evaluation.

2. **Exploratory Data Analysis (EDA)**:
   - Visualize the distribution of emotions and audio durations.
   - Generate waveforms and spectrograms for audio samples.

3. **Model Training and Evaluation**:
   - Train and evaluate a Random Forest Classifier.
   - Train and evaluate a CNN with 1D convolution.
   - Compare the performance of both models.

4. **Real-Time Voice Command Recognition**:
   - Capture live audio using `pyaudio`.
   - Recognize emotions from live audio data.

## Dependencies

The project requires the following libraries:

- `numpy`
- `pandas`
- `seaborn`
- `matplotlib`
- `librosa`
- `plotly`
- `scikit-learn`
- `keras`
- `pyaudio`

You can install these dependencies using `pip`:

```bash
pip install numpy pandas seaborn matplotlib librosa plotly scikit-learn keras pyaudio
