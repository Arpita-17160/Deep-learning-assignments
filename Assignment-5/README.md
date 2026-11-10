# Assignment 5: RNN, LSTM and GRU for Sequence Classification

## 📌 Project Overview

This project implements and compares three Recurrent Neural Network architectures — **Simple RNN, LSTM (Long Short-Term Memory), and GRU (Gated Recurrent Unit)** — for sequence classification using the IMDB Movie Reviews dataset.

The objective is to understand how different recurrent neural network architectures process sequential data and compare their performance in sentiment analysis.

The models classify movie reviews into two categories:

- **Positive (1):** Positive movie reviews.
- **Negative (0):** Negative movie reviews.

All three models are implemented using TensorFlow/Keras and evaluated using various classification metrics.

## 🎯 Objectives

The main objectives of this assignment are:

- Understand Recurrent Neural Networks (RNN).
- Understand LSTM and GRU architectures.
- Load and preprocess the IMDB Movie Reviews dataset.
- Implement Simple RNN, LSTM, and GRU models.
- Train all three models using the same dataset.
- Perform binary sentiment classification.
- Evaluate models using Accuracy, Precision, Recall, and F1-Score.
- Generate confusion matrices.
- Compare model performance using visualizations.
- Identify the best-performing model.

## 📂 Dataset

### IMDB Movie Reviews Dataset

The IMDB dataset is a popular dataset used for binary sentiment classification.

It contains 50,000 movie reviews divided into:

| Parameter | Description |
|---|---|
| Total Reviews | 50,000 |
| Training Reviews | 25,000 |
| Testing Reviews | 25,000 |
| Number of Classes | 2 |
| Classification Type | Binary |
| Input Type | Text Sequences |

The dataset is loaded directly using TensorFlow/Keras:

```python
from tensorflow.keras.datasets import imdb

(X_train, y_train), (X_test, y_test) = imdb.load_data(
    num_words=10000
)
```

### Classification Labels

| Label | Sentiment |
|---|---|
| 0 | Negative |
| 1 | Positive |

## 🧠 Models Implemented

### 1. Simple RNN (Recurrent Neural Network)

A Simple RNN is a neural network designed to process sequential data by maintaining information from previous time steps.

Features:

- Processes input sequences one step at a time.
- Maintains a hidden state.
- Uses recurrent connections to capture sequential information.
- Can suffer from the vanishing gradient problem when learning long-term dependencies.

The model uses:

```python
SimpleRNN(64)
```

### 2. LSTM (Long Short-Term Memory)

LSTM is an advanced type of RNN designed to learn long-term dependencies in sequential data.

It uses three main gates:

- **Forget Gate:** Decides which information should be removed.
- **Input Gate:** Determines which new information should be stored.
- **Output Gate:** Controls the information passed to the next time step.

Features:

- Handles long-term dependencies.
- Helps overcome the vanishing gradient problem.
- Maintains information through a cell state.

The model uses:

```python
LSTM(64)
```

### 3. GRU (Gated Recurrent Unit)

GRU is a simplified version of LSTM that uses two main gates:

- **Update Gate:** Determines how much previous information should be retained.
- **Reset Gate:** Determines how much previous information should be forgotten.

Features:

- Simpler architecture than LSTM.
- Uses fewer gates.
- Can train faster than LSTM.
- Performs effectively on sequential data.

The model uses:

```python
GRU(64)
```

## 🔄 Sequence Classification Workflow

The general workflow followed in this project is:

```text
       IMDB Movie Reviews
               |
               v
        Text Sequences
               |
               v
       Vocabulary Encoding
               |
               v
       Sequence Padding
               |
               v
          Embedding
               |
               v
      RNN / LSTM / GRU
               |
               v
            Dropout
               |
               v
       Dense Output Layer
               |
               v
       Sigmoid Activation
               |
               v
    Positive / Negative Sentiment
```

## ⚙️ Data Preprocessing

### 1. Loading the Dataset

The IMDB dataset is loaded with a vocabulary size of 10,000 words.

```python
vocab_size = 10000

(X_train, y_train), (X_test, y_test) = imdb.load_data(
    num_words=vocab_size
)
```

### 2. Sequence Padding

Movie reviews have different lengths. Padding ensures that all sequences have the same length.

The maximum sequence length is set to 200.

```python
from tensorflow.keras.preprocessing.sequence import pad_sequences

max_length = 200

X_train = pad_sequences(
    X_train,
    maxlen=max_length
)

X_test = pad_sequences(
    X_test,
    maxlen=max_length
)
```

After padding, all input sequences have a fixed length of 200.

## 🏗️ Model Architecture

All three models use a similar architecture, with only the recurrent layer changing.

```text
          Input Sequence
                |
                v
           Embedding
         (10000, 64)
                |
                v
        RNN / LSTM / GRU
             (64)
                |
                v
            Dropout
             (0.5)
                |
                v
          Dense Layer
           (1 Neuron)
                |
                v
       Sigmoid Activation
                |
                v
      Sentiment Prediction
```

### Layer Description

**1. Embedding Layer**

Converts integer-encoded words into dense vector representations.

```python
Embedding(vocab_size, 64)
```

**2. Recurrent Layer**

Processes the sequential information using Simple RNN, LSTM, or GRU.

```python
SimpleRNN(64)
LSTM(64)
GRU(64)
```

**3. Dropout Layer**

Uses a dropout rate of 0.5 to reduce overfitting.

```python
Dropout(0.5)
```

**4. Dense Output Layer**

Contains one neuron with sigmoid activation for binary classification.

```python
Dense(1, activation='sigmoid')
```

## ⚙️ Training Configuration

The following configuration is used for all three models:

| Parameter | Value |
|---|---|
| Dataset | IMDB Movie Reviews |
| Vocabulary Size | 10,000 |
| Maximum Sequence Length | 200 |
| Embedding Dimension | 64 |
| Hidden Units | 64 |
| Optimizer | Adam |
| Loss Function | Binary
