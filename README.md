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
The preprocessing consists follows these decisions:

- `Student_ID` is removed because it does not provide predictive value.
- `Grade` is encoded with `LabelEncoder` because it is the target variable and needs to be converted into numeric labels for classification.
- Numeric features are identified and normalized with `StandardScaler`.
- Categorical feature columns are identified and transformed with `OneHotEncoder`.
- The preprocessing is fit only on the training set and then applied to the test set to avoid data leakage.

This design keeps the preprocessing consistent, prevents leakage, and ensures the model receives numeric inputs in a suitable format for training.