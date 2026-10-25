# AI & Deep Learning Lab (AIML)

[![Python](https://img.shields.io/badge/Python-3.9%20%7C%203.10%20%7C%203.11-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![Keras](https://img.shields.io/badge/Keras-D00000?style=for-the-badge&logo=keras&logoColor=white)](https://keras.io/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?style=for-the-badge&logo=jupyter&logoColor=white)](https://jupyter.org/)

This repository contains the complete laboratory curriculum and implementation code for the **Artificial Intelligence & Deep Learning Laboratory**. The experiments span fundamental machine learning algorithms built from scratch, multilayer perceptrons (MLP), regularization techniques, hyperparameter tuning, Convolutional Neural Networks (CNN), Transfer Learning, Recurrent Neural Networks (LSTM) for time-series and NLP, imbalanced dataset handling (SMOTE), Convolutional Autoencoders, and Generative Adversarial Networks (GAN).

---

## 👤 Student Information

- **Student Name:** Abhinava Akash Kotha
- **PRN:** `23070521075`
- **Course:** Artificial Intelligence & Machine Learning / Deep Learning Lab

---

## 📑 Table of Practicals

| Practical | File Name | Topic | Key Techniques / Architecture | Dataset |
|:---:|:---|:---|:---|:---|
| **01** | [prac1.py](file:///c:/Users/akask/OneDrive/Desktop/college/ai%20ml%20lab/prac1.py) | Simple Linear Regression from Scratch | Analytical Least Squares, NumPy vectorization, MAE, MSE, RMSE | Synthetic 2D data |
| **02** | [prac2.py](file:///c:/Users/akask/OneDrive/Desktop/college/ai%20ml%20lab/prac2.py) | Deep Neural Network for Regression | Scikit-learn Pipeline (`ColumnTransformer`, `StandardScaler`, `OneHotEncoder`), Multi-layer Keras Sequential MLP, `EarlyStopping` | Housing Dataset (`kagglehub`) |
| **03** | [prac3.py](file:///c:/Users/akask/OneDrive/Desktop/college/ai%20ml%20lab/prac3.py) | Artificial Neural Network for Binary Classification | Feedforward ANN with Sigmoid output, Binary Cross-Entropy, Confusion Matrix & Classification Report | Customer Churn Dataset |
| **04** | [DL_week4.ipynb](file:///c:/Users/akask/OneDrive/Desktop/college/ai%20ml%20lab/DL_week4.ipynb) | Regularization & Overfitting Mitigation | L1 (Lasso), L2 (Ridge) Regularization, Dropout Layer, Train vs Test $R^2$ Comparison | Housing Price Dataset |
| **05** | [prac5.ipynb](file:///c:/Users/akask/OneDrive/Desktop/college/ai%20ml%20lab/prac5.ipynb) | Hyperparameter Tuning & Architecture Optimization | Comparative analysis of layer depths (2–4), activations (`relu`, `tanh`), optimizers (`adam`, `rmsprop`), learning rates, batch sizes | Housing Dataset |
| **06** | [prac6.ipynb](file:///c:/Users/akask/OneDrive/Desktop/college/ai%20ml%20lab/prac6.ipynb) | Convolutional Neural Network (CNN) for Image Classification | `Conv2D`, `MaxPooling2D`, `Flatten`, `Dense`, Softmax, Pixel Normalization, Prediction Visualization | MNIST Handwritten Digits |
| **07** | [prac7.ipynb](file:///c:/Users/akask/OneDrive/Desktop/college/ai%20ml%20lab/prac7.ipynb) | Transfer Learning & Fine-Tuning | Pre-trained `MobileNetV2` backbone, Feature Extraction, `GlobalAveragePooling2D`, Layer Unfreezing for Fine-Tuning | Stanford Dogs (120 Breeds) |
| **08** | [prac8.ipynb](file:///c:/Users/akask/OneDrive/Desktop/college/ai%20ml%20lab/prac8.ipynb) | Time Series Forecasting with Recurrent Neural Networks | LSTM (Long Short-Term Memory), Sliding Window Sequential Framing, `MinMaxScaler`, Trend Plotting | NASDAQ Stock Data (`yfinance`) |
| **09** | [p9 (1).ipynb](file:///c:/Users/akask/OneDrive/Desktop/college/ai%20ml%20lab/p9%20(1).ipynb) | Named Entity Recognition (NER) / Slot Filling | Keras `Embedding` Layer, LSTM Sequence Tagging, Sequence Padding, BIO Entity Schema | Travel Flight Booking Corpus |
| **10** | [p10 (1).ipynb](file:///c:/Users/akask/OneDrive/Desktop/college/ai%20ml%20lab/p10%20(1).ipynb) | Imbalanced Classification with SMOTE & DNN | Synthetic Minority Over-sampling (`SMOTE`), Leakage-Free Stratified Train/Test Split, Deep MLP with Dropout | Credit Card Fraud Detection (284k rows) |
| **11** | [p11 (1).ipynb](file:///c:/Users/akask/OneDrive/Desktop/college/ai%20ml%20lab/p11%20(1).ipynb) | Convolutional Autoencoder for Image Reconstruction | Unsupervised Representation Learning, Strided `Conv2D` Encoder, Strided `Conv2DTranspose` Decoder | MNIST Handwritten Digits |
| **12** | [DL_PRACTICAL_12.ipynb](file:///c:/Users/akask/OneDrive/Desktop/college/ai%20ml%20lab/DL_PRACTICAL_12.ipynb) | Generative Adversarial Network (GAN) for Image Generation | Adversarial Training, Generator $G(z)$, Discriminator $D(x)$, Custom `tf.GradientTape` training step, Synthetic Sampling | MNIST Handwritten Digits |

---

## 🔬 Practical Details & Implementation Walkthrough

### [Practical 1: Simple Linear Regression from Scratch](file:///c:/Users/akask/OneDrive/Desktop/college/ai%20ml%20lab/prac1.py)
- **Objective:** Implement simple 1D linear regression without machine learning libraries to understand gradient/analytical foundations.
- **Key Concepts:**
  - Mean calculation: $\bar{x}$, $\bar{y}$
  - Slope coefficient: $b_1 = \frac{\sum (x - \bar{x})(y - \bar{y})}{\sum (x - \bar{x})^2}$
  - Y-intercept: $b_0 = \bar{y} - b_1 \bar{x}$
  - Metrics: Mean Absolute Error (MAE), Mean Squared Error (MSE), Root Mean Squared Error (RMSE).
- **Output:** Regression line plotted alongside actual data points using Matplotlib.

### [Practical 2: Multilayer Perceptron (MLP) for Regression](file:///c:/Users/akask/OneDrive/Desktop/college/ai%20ml%20lab/prac2.py)
- **Objective:** Build an end-to-end regression pipeline for house price prediction using deep feedforward networks.
- **Preprocessing:**
  - Automated download using `kagglehub`.
  - Scikit-learn `ColumnTransformer` applying `SimpleImputer(strategy='median')` + `StandardScaler()` for continuous variables, and `SimpleImputer(strategy='most_frequent')` + `OneHotEncoder(handle_unknown='ignore')` for discrete variables.
- **Architecture:** Sequential network with Dense(128) $\rightarrow$ Dense(64) $\rightarrow$ Dense(32) $\rightarrow$ Dense(1) with ReLU activations.
- **Optimization:** Adam optimizer, MSE loss, `EarlyStopping(patience=10, restore_best_weights=True)`.

### [Practical 3: Artificial Neural Network for Binary Classification](file:///c:/Users/akask/OneDrive/Desktop/college/ai%20ml%20lab/prac3.py)
- **Objective:** Train an ANN to predict customer churn based on account features.
- **Preprocessing:** Feature selection (dropping ID column), `LabelEncoder` for string features (`Gender`, `Contract`, `InternetService`), and `StandardScaler` for numeric normalization.
- **Architecture:** Sequential model with Dense(32) $\rightarrow$ Dense(16) $\rightarrow$ Dense(8) $\rightarrow$ Dense(1, activation='sigmoid').
- **Evaluation:** Evaluated with `binary_crossentropy`, accuracy curve, loss trajectory, precision, recall, F1-score, and confusion matrix.

### [Practical 4: Regularization Techniques (L1, L2, Dropout)](file:///c:/Users/akask/OneDrive/Desktop/college/ai%20ml%20lab/DL_week4.ipynb)
- **Objective:** Investigate and mitigate overfitting in deep regression networks.
- **Models Benchmarked:**
  1. *Baseline Model:* Standard fully connected network without penalties.
  2. *L1 + L2 Model:* Dense layers constrained by weight penalties using `kernel_regularizer=tf.keras.regularizers.l1_l2()`.
  3. *L1 + L2 + Dropout Model:* Combining weight decay with active dropout regularization (`Dropout(0.2)`).
- **Metrics Tracked:** Train $R^2$ vs Test $R^2$, MSE, and RMSE.

### [Practical 5: Hyperparameter Tuning in Deep Neural Networks](file:///c:/Users/akask/OneDrive/Desktop/college/ai%20ml%20lab/prac5.ipynb)
- **Objective:** Systematically analyze the sensitivity of deep learning models to hyperparameter choices.
- **Experiments Conducted:**
  - Depth variation: 2, 3, and 4 hidden layers.
  - Activation functions: Rectified Linear Unit (`relu`) vs Hyperbolic Tangent (`tanh`).
  - Optimizers: Adaptive Moment Estimation (`adam`) vs Root Mean Square Propagation (`rmsprop`).
  - Tuned parameters: Learning rates (e.g., $5 \times 10^{-4}$), batch sizes (16, 32, 64), and training epochs.

### [Practical 6: Convolutional Neural Network (CNN) on MNIST](file:///c:/Users/akask/OneDrive/Desktop/college/ai%20ml%20lab/prac6.ipynb)
- **Objective:** Image classification on handwritten digit dataset using 2D spatial feature extraction.
- **Architecture:**
  - Input: $28 \times 28 \times 1$ normalized grayscale images.
  - Layer 1: `Conv2D(32, kernel_size=(3,3), activation='relu')` $\rightarrow$ `MaxPooling2D(pool_size=(2,2))`
  - Layer 2: `Conv2D(64, kernel_size=(3,3), activation='relu')` $\rightarrow$ `MaxPooling2D(pool_size=(2,2))`
  - Classification Head: `Flatten()` $\rightarrow$ `Dense(128, activation='relu')` $\rightarrow$ `Dense(10, activation='softmax')`
- **Results:** High accuracy (>98%) on test set with confusion analysis and visualization of predictions.

### [Practical 7: Transfer Learning & Fine-Tuning with MobileNetV2](file:///c:/Users/akask/OneDrive/Desktop/college/ai%20ml%20lab/prac7.ipynb)
- **Objective:** Classify 120 breeds of dogs from the Stanford Dogs dataset using pre-trained computer vision architectures.
- **Workflow:**
  1. Pre-trained `MobileNetV2` base model initialized with ImageNet weights (`include_top=False`).
  2. Base frozen: Train custom classification head with `GlobalAveragePooling2D`, `Dense(128)`, `Dropout(0.2)`, and `Dense(120, activation='softmax')`.
  3. Fine-tuning stage: Unfreeze top layers of MobileNetV2 with low learning rate to adapt features to domain-specific dog breed attributes.

### [Practical 8: Stock Market Forecasting with LSTM](file:///c:/Users/akask/OneDrive/Desktop/college/ai%20ml%20lab/prac8.ipynb)
- **Objective:** Time-series sequence forecasting of NASDAQ index prices (`^IXIC`).
- **Data Engineering:** Automated historical download via `yfinance`, sequence generation via temporal sliding window lookback, and normalization with `MinMaxScaler([0, 1])`.
- **Model:** Recurrent Neural Network with `LSTM` layers followed by `Dropout` and `Dense(1)` linear output.
- **Visualization:** Time-series alignment plot showing predicted closing prices overlaid against actual market trajectories.

### [Practical 9: Slot-Filling & Named Entity Recognition (NER) with LSTM](file:///c:/Users/akask/OneDrive/Desktop/college/ai%20ml%20lab/p9%20(1).ipynb)
- **Objective:** Extract origin and destination entities (`B-from_city`, `B-to_city`, `O`) from natural language flight booking sentences.
- **NLP Pipeline:**
  - Tokenizer text-to-integer vectorization.
  - Uniform post-sequence padding with `pad_sequences`.
  - BIO tagging scheme encoding tokens.
  - Network: `Embedding` $\rightarrow$ `LSTM` $\rightarrow$ `Dense` sequence classifier.

### [Practical 10: Credit Card Fraud Detection with SMOTE & DNN](file:///c:/Users/akask/OneDrive/Desktop/college/ai%20ml%20lab/p10%20(1).ipynb)
- **Objective:** Binary classification on an extreme class imbalance dataset (284,807 transactions, only 0.17% fraud).
- **Data Safeguard:** Stratified train-test split (`test_size=0.2`), feature standardization on train split, and `SMOTE` oversampling applied **strictly to training data** to eliminate data leakage.
- **Evaluation:** Precision, Recall, Macro/Weighted F1-score, and Confusion Matrix demonstrating high fraud capture without overwhelming false alarms.

### [Practical 11: Convolutional Autoencoder for Image Reconstruction](file:///c:/Users/akask/OneDrive/Desktop/college/ai%20ml%20lab/p11%20(1).ipynb)
- **Objective:** Unsupervised spatial compression and image reconstruction of MNIST digits.
- **Architecture:**
  - *Encoder:* `Conv2D(16, 3, strides=2, padding='same', activation='relu')` $\rightarrow$ `Conv2D(8, 3, strides=2, padding='same', activation='relu')`
  - *Decoder:* `Conv2DTranspose(8, 3, strides=2, padding='same', activation='relu')` $\rightarrow$ `Conv2DTranspose(16, 3, strides=2, padding='same', activation='relu')` $\rightarrow$ `Conv2D(1, 3, activation='sigmoid', padding='same')`
- **Evaluation:** Reconstruction loss visualization comparing 5 test digits with their autoencoded outputs.

### [Practical 12: Generative Adversarial Network (GAN) on MNIST](file:///c:/Users/akask/OneDrive/Desktop/college/ai%20ml%20lab/DL_PRACTICAL_12.ipynb)
- **Objective:** Generate realistic synthetic handwritten digits from Gaussian noise.
- **Components:**
  - *Generator ($G$):* Dense(128, activation='relu') $\rightarrow$ Dense(784, activation='tanh'), mapping $z \sim \mathcal{N}(0, I_{100})$ to 784-dimensional normalized pixels.
  - *Discriminator ($D$):* Dense(128, activation='relu') $\rightarrow$ Dense(1, activation='sigmoid'), classifying inputs as real ($1$) or synthetic ($0$).
- **Training Strategy:** Custom TensorFlow gradient descent using `@tf.function`, dual `tf.GradientTape` contexts, binary cross-entropy loss, and independent Adam optimizers ($lr=0.0002$). Generates a $4 \times 4$ grid of generated digit samples.

---

## 🛠️ Tech Stack & Requirements

### Languages & Frameworks
- **Python 3.9+ / 3.10 / 3.11**
- **TensorFlow & Keras 2.x / 3.x**
- **Scikit-Learn**
- **Imbalanced-Learn (`imblearn`)**
- **NumPy & Pandas**
- **Matplotlib**
- **YFinance**
- **KaggleHub**

### Installation

To set up a local virtual environment and install the required dependencies:

```bash
# Clone or navigate to the repository folder
cd "path/to/ai ml lab"

# Create a virtual environment
python -m venv .venv

# Activate the virtual environment
# On Windows (PowerShell):
.venv\Scripts\Activate.ps1
# On Linux / macOS:
source .venv/bin/activate

# Install required packages
pip install tensorflow keras scikit-learn imbalanced-learn numpy pandas matplotlib yfinance kagglehub jupyter
```

---

## 🚀 How to Run

### Running Python Scripts (`.py`)
Run practical scripts directly in your terminal:
```bash
python prac1.py
python prac2.py
python prac3.py
```

### Running Jupyter Notebooks (`.ipynb`)
Launch the Jupyter Notebook or JupyterLab interface:
```bash
jupyter notebook
# or
jupyter lab
```
Alternatively, upload and open any `.ipynb` file directly in [Google Colaboratory](https://colab.research.google.com/).

---

## 📂 Repository Structure

```plaintext
ai ml lab/
│
├── README.md               # Comprehensive documentation & practical index
├── prac1.py                # Practical 1: Simple Linear Regression from scratch
├── prac2.py                # Practical 2: MLP for Housing Regression (ColumnTransformer + Keras)
├── prac3.py                # Practical 3: ANN for Churn Classification
├── DL_week4.ipynb          # Practical 4: L1/L2 Regularization & Dropout Comparison
├── prac5.ipynb             # Practical 5: Hyperparameter Tuning (Layers, Activations, Optimizers)
├── prac6.ipynb             # Practical 6: CNN for MNIST Digit Classification
├── prac7.ipynb             # Practical 7: Transfer Learning with MobileNetV2 (Stanford Dogs)
├── prac8.ipynb             # Practical 8: Stock Market Forecasting with LSTM (yfinance)
├── p9 (1).ipynb            # Practical 9: Named Entity Recognition (NER) / Slot-Filling with LSTM
├── p10 (1).ipynb           # Practical 10: Credit Card Fraud Detection with SMOTE & DNN
├── p11 (1).ipynb           # Practical 11: Convolutional Autoencoder for Image Reconstruction
└── DL_PRACTICAL_12.ipynb   # Practical 12: Generative Adversarial Network (GAN) on MNIST
```

---

## 📜 Academic Disclaimer
This codebase is developed for academic, educational, and laboratory demonstration purposes as part of the AI & ML degree curriculum. Pre-trained weights, sample datasets, and benchmark data belong to their respective authors and publishers.
