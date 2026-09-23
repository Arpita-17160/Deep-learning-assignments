# Assignment 2: Multilayer Perceptron (MLP) for Iris Classification

## Aim

To design and implement a Multilayer Perceptron (MLP) for classification of the Iris dataset and evaluate its performance using accuracy and classification metrics.

---

## Dataset

The **Iris dataset** is a classification dataset containing measurements of iris flowers belonging to **3 different species**.

Each sample contains:

- Sepal Length
- Sepal Width
- Petal Length
- Petal Width

The dataset contains:

- Number of samples: **150**
- Number of features: **4**
- Number of classes: **3**
- Classes: **Setosa, Versicolor, Virginica**

The dataset is loaded using **Scikit-learn**.

---

## Technologies and Libraries Used

- Python
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- TensorFlow
- Keras

## Data Loading

The Iris dataset is loaded using Scikit-learn.

The input features and target classes are separated for further processing.

---

## Data Preprocessing

The dataset is divided into training and testing sets using an **80:20 ratio**.

The input features are standardized using **StandardScaler** so that all features are brought to a common scale.

---

## MLP Model

A **Multilayer Perceptron (MLP)** is implemented using **TensorFlow/Keras**.

The model consists of:

- Input layer with **4 features**
- First hidden layer with **16 neurons**
- Second hidden layer with **8 neurons**
- Output layer with **3 neurons**
- ReLU activation function in the hidden layers
- Softmax activation function in the output layer

---

## L2 Regularization

**L2 regularization** is applied to the hidden layers of the neural network to help reduce overfitting.

The regularization value used is **0.001**.

---

## Model Compilation

The model is compiled using:

- **Optimizer:** Adam
- **Loss Function:** Sparse Categorical Crossentropy
- **Evaluation Metric:** Accuracy

---

## Model Training

The MLP model is trained for **20 epochs** with a **batch size of 8**.

A **20% validation split** is used during training.

---

## Model Evaluation

The trained model is evaluated using the test dataset.

The performance of the model is measured using:

- Test Loss
- Test Accuracy
- Precision
- Recall
- F1-Score

---

## Classification Report

A classification report is generated to evaluate the performance of the model for each Iris class.

The report includes:

- Precision
- Recall
- F1-Score
- Support

---

## Result

The Multilayer Perceptron model was successfully implemented for Iris dataset classification.

The model achieved:

- **Test Loss:** 0.5777
- **Test Accuracy:** 73.33%

---

## Conclusion

The Iris dataset was successfully loaded, preprocessed, and divided into training and testing sets. A Multilayer Perceptron with two hidden layers and L2 regularization was implemented using TensorFlow/Keras. The model was trained and evaluated using accuracy and classification metrics.
