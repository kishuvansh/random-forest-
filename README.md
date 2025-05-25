#  **Diabetes Prediction Using Machine Learning**

This project implements a machine learning model to predict the likelihood of diabetes in patients based on various health metrics. Utilizing a Random Forest classifier, the model analyzes features such as age, BMI, HbA1c levels, and medical history to generate predictions. The workflow includes data preprocessing, feature engineering, model training, and evaluation, achieving robust performance in diabetes risk assessment.

# Table of Contents
#### [Project Overview](#project-overview)
#### [Dataset Description](#dataset-description)
#### [Installation](#installation)
#### [Usage](#usage)
#### [Methodology](#methodology)
#### [Results](#results)

# Project Overview
Diabetes prediction plays a crucial role in preventive healthcare. This project demonstrates a complete machine learning pipeline for binary classification of diabetes status using clinical data. The implementation focuses on:
Handling missing values and categorical data
Normalizing numerical features
Training a Random Forest classifier
Evaluating model performance using standard metrics
The model serves as a foundation for clinical decision support systems and risk stratification tools.

## Dataset Description
The dataset used contains 100,000 samples with the following features:

| Feature             | Type         | Description                                              | Example Value |
|---------------------|--------------|----------------------------------------------------------|--------------|
| gender              | Categorical  | Patient gender (encoded: Female=0, Male=1)               | Female       |
| age                 | Numerical    | Age of the patient (in years)                            | 54.0         |
| hypertension        | Binary       | 1 = Hypertension present, 0 = Not present                | 0            |
| heart_disease       | Binary       | 1 = Heart disease present, 0 = Not present               | 1            |
| smoking_history     | Categorical  | Smoking status (never, current, former, etc.)            | never        |
| bmi                 | Numerical    | Body Mass Index                                          | 25.19        |
| HbA1c_level         | Numerical    | Hemoglobin A1c level                                     | 6.6          |
| blood_glucose_level | Numerical    | Blood glucose level                                      | 140          |
| diabetes            | Binary       | Target: 1 = Diabetic, 0 = Non-diabetic                   | 0            |




# Installation
Clone this repository:</br>
git clone https://github.com/kishuvansh/random-forest-.git
cd random-forest-
install jupyter Notebook and required Python libraries:</br>
pip install jupyter pandas scikit-learn matplotlib

#Usage
# Usage

1. Start Jupyter Notebook:
    ```
    jupyter notebook
    ```
2. Open the main notebook file (e.g., `random_forest_diabetes.ipynb`).
3.Download the dataset
Make sure diabetes_prediction_dataset.csv is in the same directory as the notebook or update the path accordingly
 in the code cell:
</br>file_path = "diabetes_prediction_dataset.csv"


4. Follow the notebook cells in order:
    - Load and explore the dataset.
    - Preprocess the data as shown.
    - Train the Random Forest model.
    - Evaluate model performance.
    - Use the model to make predictions on new patient data.

5. To make a prediction with custom input, modify the input cell as shown in the notebook and run it:
    ```python
    # Example input for prediction
    new_data = [[6,148,72,35,0,33.6,0.627,50]]
    prediction = model.predict(new_data)
    print("Diabetes Prediction:", "Positive" if prediction[0]==1 else "Negative")
    ```


# Methodology
1.Data Loading and Exploration
  -The dataset (diabetes_prediction_dataset.csv) is loaded and the relevant clinical features are inspected.  
</br>
2.Data Preprocessing
  - Handling Categorical Variables:</br>
     - The gender column is encoded using label encoding (Male=1, Female=0).  </br>
      - The smoking_history column is transformed using one-hot encoding.  </br>

3.Handling Missing Values:
(If missing values exist, describe how they are handled. If not, this can be omitted or state that the dataset had no missing values.)
Feature Scaling:
  - Numerical columns (age, bmi, HbA1c_level, blood_glucose_level) are standardized using StandardScaler.
  -  Feature and Target Selection

  - Features (X) are selected by dropping the diabetes column.
The target variable (y) is the diabetes column.
Splitting the Dataset

## The data is split into training (80%) and testing (20%) sets using train_test_split.
Model Training

A Random Forest Classifier (RandomForestClassifier from scikit-learn) is initialized with 100 trees and trained on the training data.
Model Evaluation

The model’s performance is evaluated on the test set using:
Accuracy score
Confusion matrix
Classification report (precision, recall, f1-score)
Prediction

The trained model can be used to make predictions on new/unseen data by providing input in the same format as the features.


# Result
