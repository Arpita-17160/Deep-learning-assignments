# Assignment 1: Wine Dataset Loading, Preprocessing and Classification

## Aim

To load the Wine dataset, preprocess the data, split it into training and testing sets, and build a neural network model for classification.

---

## Dataset

The **Wine dataset** is a classification dataset containing chemical properties of wines belonging to **3 different classes**.

The dataset contains:

* Number of samples: **178**
* Number of features: **13**
* Number of classes: **3**
* Target: Wine class

The dataset is loaded using **Scikit-learn's `load_wine()`** function.

---

## Technologies and Libraries Used

* Python
* NumPy
* Pandas
* Matplotlib
* Seaborn
* Scikit-learn
* TensorFlow
* Keras

### Libraries

```python
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
```

---

## Data Loading

The Wine dataset is loaded using Scikit-learn and separated into input features (`X`) and target labels (`y`).

```python
wine = load_wine()

X = wine.data
y = wine.target
```

The dataset contains **178 records and 13 input features**.

---

## Data Exploration

The dataset is converted into a Pandas DataFrame to examine the first few records, dataset shape, information, and missing values.

```python
df = pd.DataFrame(X, columns=wine.feature_names)
df["Target"] = y

print(df.head())
print(df.shape)
print(df.info())
print(df.isnull().sum())
```

## The dataset has **178 rows and 14 columns**, including 13 features and 1 target column. No missing values were found in the dataset.

## Data Preprocessing

The input features are standardized using `StandardScaler`.

```python
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

Standardization is performed so that the input features are brought to a common scale before training the neural network.

---

## Train-Test Split

The preprocessed dataset is divided into training and testing sets using an **80:20 ratio**.

```python
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y,
    test_size=0.2,
    random_state=42
)
```

The resulting datasets are:

* Training data: **142 samples**
* Testing data: **36 samples**

---

## Neural Network Model

A feed-forward neural network is created using the Keras `Sequential` model.

The architecture consists of:

* Input layer with **13 input features**
* Hidden layer with **32 neurons** and ReLU activation
* Hidden layer with **16 neurons** and ReLU activation
* Output layer with **3 neurons** and Softmax activation

```python
model = Sequential([
    Dense(32, activation='relu', input_shape=(13,)),
    Dense(16, activation='relu'),
    Dense(3, activation='softmax')
])
```

The model is designed to classify the wine samples into one of the three target classes.

---

## Model Compilation

The model is compiled using the **Adam optimizer**, **Sparse Categorical Crossentropy** loss function, and **accuracy** as the evaluation metric.

```python
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
```

---

## Model Training

The neural network is trained for **20 epochs** with a batch size of **16**. A validation split of **20%** is used during training.

```python
history = model.fit(
    X_train,
    y_train,
    epochs=20,
    batch_size=16,
    validation_split=0.2,
    verbose=1
)
```

The training process achieved high training and validation accuracy throughout the epochs.

---

## Model Evaluation

The trained model is evaluated using the test dataset.

```python
loss, accuracy = model.evaluate(X_test, y_test)

print("Test Loss:", loss)
print("Test Accuracy:", accuracy)
```

The obtained results were:

* **Test Loss:** 0.0211
* **Test Accuracy:** 1.00 (100%)

---

## Result

The Wine dataset was successfully loaded and preprocessed. A neural network classification model was built and trained using TensorFlow/Keras. The model achieved **100% test accuracy** on the test dataset.
