# Student-Grade-Prediction

This project builds a model capable of predicting the grades (A, B, C, D, Fail) of university students based on their lifestyle factors and habits. It includes a simple baseline model and a hybrid deep learning framework based on a state-of-the-art paper.

# Data Selection
The data selected for this project comes from kaggle, created by Sarvesh Chhetri and available in the following link: 
https://www.kaggle.com/datasets/sarveshchhetri/student-lifestyle-vs-academic-performance-dataset?resource=download


This dataset captures a simulation of 8,000 university students with a wide variety of academic, behavioral and socioeconomic attributes.

The dataset has two versions: The one that classifies grades in a range from A-C, and the second one that has a numeric float value in the grade column.

This project uses the version with the categorical target in the `Grade` column (`student_performance_grade.csv`), so the task is framed as a classification problem.

# Train/Test Split
The dataset is split into training and test sets with an 80/20 ratio.

The split is stratified by the encoded target so that the class distribution is preserved in both subsets. This is important because the target is imbalanced, and without stratification the test set could end up overrepresenting or underrepresenting some grades.

# Preprocessing
The exploratory analysis and preprocessing follows these decisions:

- `Student_ID` is dropped — no predictive value.
- `Grade` is encoded with `LabelEncoder` to convert string labels into numeric classes.
- The dataset is split 80/20 (train/test), stratified by class to preserve the grade distribution in both sets.
- Numeric features are normalized with `MinMaxScaler` and categorical features are one-hot encoded.
- The preprocessor is fit only on the training set to prevent data leakage.
- The dataset has no missing values, so no imputation is needed.
The dataset is **class-imbalanced**: grade A appears in ~56% of samples, while Fail represents less than 1%. This affects model training and is addressed in the framework model with SMOTE.
 
---


# Baseline Model

A Logistic Regression trained with scikit-learn serves as a transparent benchmark.
 
Logistic Regression is a linear model, meaning it can only learn straight-line decision boundaries and is sensitive to class imbalance, tending to over-predict the majority class (A).
 
| Metric | Score |
|---|---|
| Accuracy | 76.19% |
| Precision | 74.73% |
| Recall | 76.19% |
| F1-Score | 74.91% |
| AUC-ROC | 0.9195 |
 
---

# Framework-Based Model

The notebook then introduces a hybrid deep learning framework based on the paper by Ma & Xiao (2026) - [Application of deep learning to the development of a prediction model for college students’ learning outcomes](https://www.scopus.com/pages/publications/105027887338?origin=resultslist). The architecture is adapted to the classification setting used in this project:

Raw Data -> ResNet Encoder + Autoencoder -> LSTM Feature Selection -> Softmax Classifier

The idea is to combine three complementary stages:

- ResNet blocks learn non-linear feature interactions.
- The autoencoder compresses the input into a latent space.
- The LSTM treats the latent dimensions as a sequence and learns which features are most useful for prediction.

The paper originally solves a regression problem (predicting continuous exam scores). This project adapts it to **multi-class classification** (grades A–F):
 
| Paper (Regression) | This project (Classification) |
|---|---|
| MSE loss | Categorical Cross-Entropy |
| Linear output | Softmax output (one node per class) |
| RMSE, MAE, R² | Accuracy, Precision, Recall, F1, AUC-ROC |

From the exploratory results already documented in the notebook, the main interpretation is that class imbalance is one of the biggest challenges in the dataset. The hybrid model is included to test whether a deeper non-linear architecture can improve on the baseline and handle the minority classes more effectively.

### Iterative refinement
 
The final architecture was reached through several iterations:
 
| Iteration | Architecture | Accuracy |
|---|---|---|
| 1 | Full paper: ResNet + Autoencoder + LSTM | 66% |
| 2 | ResNet + Autoencoder + Dense (no LSTM) | 66% |
| 3 | Direct Dense on features (no AE, no ResNet) | 74% |
| 4 | Dense + L2=0.001, 128 units | 74.7% |
| **5** | **Dense + L2=0.01, 64 units** | **75.6%** |
 
The autoencoder and ResNet were discarded after finding they introduced an information bottleneck, compressing features to a 32-dimensional latent space was discarding information that simpler models used directly. The LSTM was discarded because the dataset is a static snapshot (one row per student), with no temporal dimension to model.

### Final architecture
 
```
Raw Data → MinMaxScaler + OHE → SMOTE → Dense(64) → Dense(32) → Dense(16) → Softmax
```
 
Each Dense layer uses BatchNormalization, Dropout, and L2 regularization.

### Addressing class imbalance
 
**SMOTE (Synthetic Minority Over-sampling Technique)** is applied only to the training set. It generates synthetic examples for minority classes by interpolating between real neighbors in the feature space, so all classes have equal representation during training. The test set is never modified.
 
`class_weight` was considered but discarded since SMOTE already balances the classes, applying both simultaneously is redundant.
 
### Regularization
 
Three techniques are combined to control overfitting:
 
- **BatchNormalization**: stabilizes training by normalizing layer inputs
- **Dropout (0.3)**: randomly deactivates neurons during training so the model can't memorize specific paths
- **L2 regularization (0.01)**: penalizes large weights: forcing the model to prefer simpler, more general solutions

Reducing the network capacity from 128 to 64 units in the first layer was also key — larger networks memorized the training data without improving validation performance.
