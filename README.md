# Student-Grade-Prediction

This project's aim is to build a model capable of predicting the grades of university students based on their lifestyle factors and habits.

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

- `Student_ID` is removed because it does not provide predictive value.
- `Grade` is encoded with `LabelEncoder` because it is the target variable and needs to be converted into numeric labels for classification.
- Numeric features are identified and normalized with `StandardScaler`.
- Categorical feature columns are identified and transformed with `OneHotEncoder`.
- The preprocessing is fit only on the training set and then applied to the test set to avoid data leakage.
- The dataset has no missing values, so no imputation step is needed.
- The target distribution is imbalanced, with grade A appearing much more often than the lower-frequency classes.

This imbalance helps explain why the baseline model can perform better on the majority class while struggling with harder classes. The notebook also notes that some grades are frequently confused with one another, especially pairs such as Fail and C.

This design keeps the preprocessing consistent, prevents leakage, and ensures the model receives numeric inputs in a suitable format for training.

# Baseline Model

The first model used in the project is Logistic Regression. It serves as a transparent benchmark because it is easy to train, fast to evaluate, and simple to interpret.

The baseline evaluation in the notebook focuses on:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion matrix

This gives a clear view of how the model behaves across all grade classes instead of relying on accuracy alone.

# Framework-Based Model

The notebook then introduces a hybrid deep learning framework based on the paper by Ma & Xiao (2026) - [Application of deep learning to the development of a prediction model for college students’ learning outcomes](https://www.scopus.com/pages/publications/105027887338?origin=resultslist). The architecture is adapted to the classification setting used in this project:

Raw Data -> ResNet Encoder + Autoencoder -> LSTM Feature Selection -> Softmax Classifier

The idea is to combine three complementary stages:

- ResNet blocks learn non-linear feature interactions.
- The autoencoder compresses the input into a latent space.
- The LSTM treats the latent dimensions as a sequence and learns which features are most useful for prediction.

Because the original paper focuses on regression, the notebook adapts the output layer and training objective for multi-class classification. The classification version uses softmax activation and categorical cross-entropy.

From the exploratory results already documented in the notebook, the main interpretation is that class imbalance is one of the biggest challenges in the dataset. The hybrid model is included to test whether a deeper non-linear architecture can improve on the baseline and handle the minority classes more effectively.