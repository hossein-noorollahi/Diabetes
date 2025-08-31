---

# Diabetes Prediction using Machine Learning

This repository contains a Jupyter Notebook that demonstrates a complete machine learning workflow for predicting diabetes using the Pima Indians Diabetes Dataset. It implements and evaluates Logistic Regression, Random Forest, and Decision Tree algorithms to identify the most effective model for this classification task.

## Table of Contents
- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Main Function Points](#main-function-points)
- [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
- [Data Preprocessing](#data-preprocessing)
- [Machine Learning Models](#machine-learning-models)
- [Model Evaluation & Comparison](#model-evaluation--comparison)
- [Key Findings](#key-findings)
- [Technology Stack](#technology-stack)
- [How to Run](#how-to-run)

## Project Overview

Diabetes is a global health concern, and predictive modeling plays a crucial role in early diagnosis and management. This project leverages the Pima Indians Diabetes Dataset to build and compare various machine learning models (Logistic Regression, Random Forest, Decision Tree) for diabetes prediction. The goal is to provide a robust solution and identify key factors contributing to diabetes.

## Dataset

The **Pima Indians Diabetes Dataset**, sourced from the UCI Machine Learning Repository and originally from the National Institute of Diabetes and Digestive and Kidney Diseases, is a benchmark dataset for diabetes research. It comprises **768 records** of female patients of Pima Indian heritage, with 8 diagnostic measures and one target variable (`Outcome`).

**Features:**
*   `Pregnancies`: Number of times pregnant.
*   `Glucose`: Plasma glucose concentration after 2-hour oral glucose tolerance test.
*   `BloodPressure`: Diastolic blood pressure (mm Hg).
*   `SkinThickness`: Triceps skin fold thickness (mm).
*   `Insulin`: 2-Hour serum insulin (mu U/ml).
*   `BMI`: Body mass index (weight in kg/(height in m)^2).
*   `DiabetesPedigreeFunction`: A function that scores the likelihood of diabetes based on family history.
*   `Age`: Age of the individual (years).
*   `Outcome`: Class variable (0: No Diabetes, 1: Diabetes).

Approximately 34.9% of the patients in this dataset are diagnosed with diabetes.

## Main Function Points

*   **Data Preprocessing:** Handled missing values (identified as zeros in certain columns) and duplicates.
*   **Exploratory Data Analysis (EDA):** Visualized data distributions, correlations, and performed statistical tests.
*   **Feature Scaling:** Applied `StandardScaler` to standardize numerical features.
*   **Data Splitting:** Divided the dataset into training and testing sets (80/20 split).
*   **Model Training:** Trained Logistic Regression, Decision Tree, and Random Forest classifiers.
*   **Hyperparameter Tuning:** Optimized Decision Tree using `GridSearchCV`.
*   **Model Evaluation:** Assessed model performance using accuracy, classification reports, confusion matrices, ROC curves, AUC scores, and cross-validation.
*   **Feature Importance Analysis:** Identified significant features for diabetes prediction using model coefficients and feature importances.
*   **Model Comparison:** Compared the performance of all three models to determine the best-performing one.

## Exploratory Data Analysis (EDA)

The EDA phase involved a thorough examination of the dataset to understand its structure, distributions, and relationships between features and the target variable.

*   **Data Summary:** `df.describe()`, `df.info()`, `df.dtypes`, `df.shape` provided initial insights.
*   **Target Variable Distribution:** A `countplot` showed the class imbalance (65.1% no diabetes, 34.9% diabetes).
*   **Feature Distributions:** Histograms and box plots were used to visualize individual feature distributions and identify potential outliers or skewed data.
*   **Feature Means by Outcome:** Calculated and visualized the average values of features for both diabetic and non-diabetic groups.
*   **Statistical Significance Testing:** Welch's t-test was performed on each feature grouped by `Outcome` to identify statistically significant differences (p-value < 0.05).
    *   **Significant Features:** `Pregnancies`, `Glucose`, `SkinThickness`, `Insulin`, `BMI`, `DiabetesPedigreeFunction`, and `Age` showed significant differences. `BloodPressure` did not.
*   **Pairplot:** Visualized relationships between all pairs of features, colored by `Outcome`, to observe potential separation patterns.
*   **Correlation Heatmap:** Displayed the correlation matrix to understand linear relationships between features. No strong multicollinearity (correlation > 0.9) was observed.

## Data Preprocessing

*   **Missing Values:** The dataset was checked for null values using `df.isnull().sum()`. No explicit nulls were found, but zero values in `Glucose`, `BloodPressure`, `SkinThickness`, `Insulin`, and `BMI` were implicitly treated as missing or indicative of a specific condition in some contexts (though not imputed in this specific notebook as per the provided code).
*   **Duplicate Rows:** Checked for and removed any duplicate entries using `df.duplicated()` and `df.drop_duplicates()`. No duplicates were found in this dataset.
*   **Feature-Target Split:** Features (`x`) and the target variable (`y`) were separated.
*   **Train-Test Split:** The data was split into 80% training and 20% testing sets using `train_test_split` with `random_state=14` and `stratify=y` for consistent and balanced splits.
*   **Feature Scaling:** `StandardScaler` was applied to `x_train` and `x_test` to standardize the features, which is crucial for distance-based algorithms like Logistic Regression and helps improve model performance.

## Machine Learning Models

Three popular classification algorithms were implemented and evaluated:

1.  ### Logistic Regression
    *   **Model:** `LogisticRegression(max_iter=1000)`
    *   **Performance (with scaling):**
        *   Accuracy: 78.57%
        *   Mean 5-Fold Cross-Validation Score: 77.22%
    *   **Feature Importance:** Coefficients were analyzed, with `DiabetesPedigreeFunction` and `BMI` showing the highest odds ratios, indicating their strong positive association with diabetes likelihood.

2.  ### Decision Tree Classifier
    *   **Model:** `DecisionTreeClassifier(max_depth=5, random_state=42)`
    *   **Performance (with scaling):**
        *   Accuracy: 73.38%
        *   Mean 5-Fold Cross-Validation Score: 74.88%
    *   **Hyperparameter Tuning:** `GridSearchCV` was used to find optimal parameters:
        *   `max_depth`: 5
        *   `min_samples_leaf`: 4
        *   `min_samples_split`: 2
        *   Best Cross-Validation Score after tuning: 75.42%

3.  ### Random Forest Classifier
    *   **Model:** `RandomForestClassifier(n_estimators=100, random_state=42)`
    *   **Performance (with scaling):**
        *   Accuracy: 77.92%
        *   Mean 5-Fold Cross-Validation Score: 76.70%
    *   **Feature Importance:** `Glucose` was identified as the most important feature, followed by `BMI` and `DiabetesPedigreeFunction`.

## Model Evaluation & Comparison

Each model was evaluated using:
*   **Accuracy Score:** Overall correct predictions.
*   **Classification Report:** Precision, Recall, F1-score for each class.
*   **Confusion Matrix:** Visualization of True Positives, True Negatives, False Positives, False Negatives.
*   **ROC Curve and AUC:** Assessment of classifier performance across various thresholds.
*   **5-Fold Cross-Validation:** To ensure model generalization and reduce overfitting bias.

| Model                 | Accuracy Score (%) | Cross-Validation Score (%) |
| :-------------------- | :----------------- | :------------------------- |
| Logistic Regression   | 78.57              | 77.22                      |
| Random Forest         | 77.92              | 76.70                      |
| Decision Tree         | 73.38              | 74.88                      |

## Key Findings

*   **Logistic Regression** achieved the highest accuracy and cross-validation score among the evaluated models, indicating its strong performance on this dataset after feature scaling.
*   **Random Forest** showed competitive performance, very close to Logistic Regression.
*   **Glucose** was consistently identified as a highly significant and important feature across both statistical tests and tree-based models. `BMI`, `DiabetesPedigreeFunction`, and `Age` also played crucial roles.
*   Feature scaling significantly improved the performance of Logistic Regression, highlighting its sensitivity to feature ranges.

## Technology Stack

*   **Python**
*   **Jupyter Notebook**
*   **Pandas** (for data manipulation and analysis)
*   **NumPy** (for numerical operations)
*   **Scikit-learn** (for machine learning algorithms, preprocessing, and model evaluation)
*   **Seaborn** (for statistical data visualization)
*   **Matplotlib** (for creating static, animated, and interactive visualizations)
*   **SciPy** (for statistical tests)
---