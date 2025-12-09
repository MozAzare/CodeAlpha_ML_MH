# Project 1: Credit Amount Classification 

This project predicts whether a customer is likely to request a high or low credit amount based on demographic and financial information. It uses the German Credit dataset and applies beginner-friendly preprocessing steps, exploratory data analysis, and a Random Forest classifier.

## 🔍 1. Project Overview

Banks and financial institutions routinely evaluate clients to determine suitable credit amounts. This project simplifies that process by predicting **HighCredit** (above the median credit amount) vs. **LowCredit** (below or equal to the median) using machine learning.

This turns the problem into a binary classification task, making it suitable for beginners learning preprocessing, feature encoding, classification algorithms, and evaluation metrics.

## 📦 2. Dataset Description

- **Rows**: 1000
- **Features**: 9 input attributes
- **Target**:
  - `HighCredit = 1` → Credit amount above 2319.5
  - `HighCredit = 0` → Credit amount below or equal to 2319.5

### Class Balance

| Class | Count |
|-------|-------|
| LowCredit (0) | 500 |
| HighCredit (1) | 500 |

## 🧹 3. Preprocessing Steps

1. Filled missing values in `"Saving accounts"` and `"Checking account"` with `"unknown"`.
2. Label-encoded categorical variables:
   - `Sex`
   - `Housing`
   - `Saving accounts`
   - `Checking account`
   - `Purpose`
3. Converted the target variable into:
```python
   df["HighCredit"] = (df["Credit amount"] > median).astype(int)
```
4. Scaled numerical features using `StandardScaler`.

## 📊 4. Exploratory Data Analysis (EDA)

- Age and credit amount distributions were plotted using `seaborn`.
- A correlation heatmap was used to visualize feature relationships.
- Feature distributions helped identify skew (especially credit amount).
- Categorical data was explored after label encoding.

## 🤖 5. Model Used

### Random Forest Classifier

- Chosen for its robustness and ability to capture non-linear patterns.
- Trained using 200 decision trees (`n_estimators=200`).

## 🧪 6. Model Results

**Overall Accuracy**: 0.74

### Confusion Matrix
```
[[124  31]
 [ 46  99]]
```

### Classification Report

| Class | Precision | Recall | F1-Score |
|-------|-----------|--------|----------|
| LowCredit (0) | 0.73 | 0.80 | 0.76 |
| HighCredit (1) | 0.76 | 0.68 | 0.72 |

### ✔ Interpretation

- The model performs slightly better on **LowCredit** cases, with higher recall (0.80).
- For **HighCredit**, precision is strong (0.76) but recall drops to 0.68, meaning the model misses some high-credit applicants.
- Overall performance is balanced and suitable for beginner ML use-cases.

## 📌 7. Feature Importance

Random Forest identified the following top predictors:

| Feature | Importance |
|---------|------------|
| Duration | 0.346 |
| Unnamed: 0 (row index) | 0.195 |
| Age | 0.153 |
| Purpose | 0.077 |
| Job | 0.061 |
| Checking account | 0.053 |
| Saving accounts | 0.049 |
| Housing | 0.035 |
| Sex | 0.028 |

### ✔ Interpretation

- **Loan Duration** is the strongest predictor — longer durations typically correspond to larger credit requests.
- **Age** also plays a significant role (older applicants often request higher amounts).
- Having a **checking/saving account** contributes moderately.
- **Gender** and **housing type** provide minimal predictive power.

## 🚀 Getting Started

### Prerequisites
```bash
pip install pandas numpy scikit-learn seaborn matplotlib
```

### Usage
```python
# Import required libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load and preprocess data
df = pd.read_csv('german_credit.csv')

# Follow preprocessing steps outlined above
# Train the model and evaluate results
```

## 📈 Future Improvements

- Experiment with other algorithms (XGBoost, SVM, Neural Networks)
- Implement cross-validation for more robust evaluation
- Feature engineering to create new predictive variables
- Handle class imbalance techniques if needed
- Hyperparameter tuning using GridSearchCV or RandomizedSearchCV

## 📝 License

This project is open source and available under the [MIT License](LICENSE).

## 👥 Contributors

- Muhammad H - Initial work

## 🙏 Acknowledgments

- German Credit Dataset from UCI Machine Learning Repository
- scikit-learn documentation and community
