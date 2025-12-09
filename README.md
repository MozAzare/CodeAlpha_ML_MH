# Project 1: Credit Amount Classification 

This project predicts whether a customer is likely to request a high or low credit amount based on demographic and financial information. It uses the German Credit dataset and applies beginner-friendly preprocessing steps, exploratory data analysis, and a Random Forest classifier.

##  1. Project Overview

Banks and financial institutions routinely evaluate clients to determine suitable credit amounts. This project simplifies that process by predicting **HighCredit** (above the median credit amount) vs. **LowCredit** (below or equal to the median) using machine learning.

This turns the problem into a binary classification task, making it suitable for beginners learning preprocessing, feature encoding, classification algorithms, and evaluation metrics.

##  2. Dataset Description

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

##  3. Preprocessing Steps

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

## 4. Exploratory Data Analysis (EDA)

- Age and credit amount distributions were plotted using `seaborn`.
- A correlation heatmap was used to visualize feature relationships.
- Feature distributions helped identify skew (especially credit amount).
- Categorical data was explored after label encoding.

## 5. Model Used

### Random Forest Classifier

- Chosen for its robustness and ability to capture non-linear patterns.
- Trained using 200 decision trees (`n_estimators=200`).

## 6. Model Results

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

##  7. Feature Importance

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

##  Getting Started

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



-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Project 2: Heart Disease Prediction (Machine Learning model)

This project predicts whether a patient is likely to have **heart disease** based on medical and physiological attributes. It uses the well-known **UCI Heart Disease** dataset and implements a complete workflow: data preprocessing, exploratory data analysis, model training, evaluation, and feature interpretation.

---

##  1. Project Overview

Heart disease is one of the leading causes of death worldwide. Early prediction can help guide medical decisions and potentially save lives.  
In this project, we build a machine learning classifier to predict whether a patient has heart disease (`target = 1`) or not (`target = 0`).

This task is suitable for beginners and covers essential ML steps:
- preprocessing  
- scaling  
- train/test split  
- classification models  
- model evaluation  
- feature importance + medical interpretation  

---

##  2. Dataset Description

The dataset contains **303 rows** and **14 clinical features**, including:
- `age` – patient age  
- `sex` – 1 = male, 0 = female  
- `cp` – chest pain type  
- `trestbps` – resting blood pressure  
- `chol` – cholesterol level  
- `thalach` – maximum heart rate achieved  
- `exang` – exercise-induced angina  
- `oldpeak` – ST depression  
- `ca`, `thal`, `slope` – critical heart-related indicators  

**Target variable:**

| Value | Meaning        |
|-------|----------------|
| 0     | No heart disease |
| 1     | Heart disease present |

---

##  3. Preprocessing Steps

1. Verified dataset structure and missing values (none found).
2. Separated input features `X` and target `y`.
3. Standardized numerical features using `StandardScaler`.
4. Performed an **80/20** train-test split with class stratification.

---

##  4. Exploratory Data Analysis (EDA)

- Histograms for key variables such as age and cholesterol.
- Checked class distribution using countplots.
- Generated a **correlation heatmap**, showing relationships between medical features.
- Observed strong correlations with:
  - `cp` (chest pain type)
  - `thalach` (max heart rate)
  - `oldpeak` (ST depression)
  - `thal` (blood disorder category)

---

##  5. Models Used

Two models were trained and compared:

### **1. Logistic Regression**
A baseline linear model for binary classification.

### **2. Random Forest Classifier**
A tree-based ensemble model capable of capturing non-linear relationships.

---

##  6. Model Performance

###  Logistic Regression
- **Accuracy:** 0.803  
- **ROC-AUC:** 0.869  
- **Recall for class 1 (Disease):** 0.91

###  Random Forest (Best Model)
- **Accuracy:** 0.819  
- **ROC-AUC:** 0.905  
- **Recall for class 1 (Disease):** 0.97  
- **Precision for class 0 (No disease):** 0.95  


###  Interpretation
- The model is **very strong at identifying patients WITH heart disease**  
  (Recall = 97%), which is the most important clinical priority.
- Slightly lower recall for “no disease” cases (64%), meaning some false positives (acceptable for medical screening).
- Random Forest outperforms Logistic Regression in **accuracy and ROC-AUC**, indicating a better overall ranking ability.

---

##  7. Feature Importance (Random Forest)

| Feature   | Importance |
|-----------|------------|
| `cp` (chest pain type) | 0.152 |
| `thalach` (max heart rate) | 0.114 |
| `thal` | 0.112 |
| `oldpeak` | 0.107 |
| `ca` | 0.089 |
| `chol` | 0.083 |
| `age` | 0.079 |
| `trestbps` | 0.074 |
| `exang` | 0.068 |
| `slope` | 0.057 |
| `sex` | 0.028 |
| `restecg` | 0.022 |
| `fbs` | 0.009 |

### ✔ Medical Interpretation
- **Chest pain type (`cp`)** is the strongest predictor of heart disease.
- **Higher maximum heart rate (`thalach`)** and **lower ST depression (`oldpeak`)** strongly indicate better heart function.
- **`ca`**, representing major vessel blockage, is also a critical factor.

---

##  8. How to Run This Project

```bash
jupyter notebook Heart_Disease_Model.ipynb



## 📝 License

This project is open source and available under the [MIT License](LICENSE).

## 👥 Contributors

- Muhammad H - Initial work

## 🙏 Acknowledgments

- German Credit Dataset from UCI Machine Learning Repository
- scikit-learn documentation and community
