# Credit-Score-Classification-with-XGBoost-RandomForest

A machine learning project for predicting customer credit scores using Random Forest and XGBoost. The project focuses on cleaning a messy real-world-style dataset, performing exploratory data analysis, building reproducible preprocessing pipelines, and comparing tree-based classification models.

Both models achieve approximately **70% accuracy**, with **XGBoost performing slightly better overall**.

---

## 📌 Project Overview

The objective is to predict `Credit_Score` as a three-class classification problem.

The original dataset contains **28 variables** and several data-quality issues, including:

* Inconsistent representations of missing values
* Numeric values stored as text
* Invalid or unrealistic values
* Placeholder strings and stray characters
* Multi-value text fields
* Missing numerical and categorical values

A major focus of this project is the **data-cleaning and preprocessing process**, with each transformation selected based on the structure and meaning of the data.

---

## 📂 Dataset

**File:** `Credis_Score_Dataset_A.csv`

| Category           | Features                                                                                                       |
| ------------------ | -------------------------------------------------------------------------------------------------------------- |
| Identifiers        | `ID`, `Customer_ID`, `Name`, `SSN`, `Month`                                                                    |
| Personal           | `Age`, `Occupation`                                                                                            |
| Income             | `Annual_Income`, `Monthly_Inhand_Salary`                                                                       |
| Accounts & Cards   | `Num_Bank_Accounts`, `Num_Credit_Card`, `Interest_Rate`                                                        |
| Loans              | `Num_of_Loan`, `Type_of_Loan`                                                                                  |
| Payment History    | `Delay_from_due_date`, `Num_of_Delayed_Payment`, `Payment_of_Min_Amount`, `Payment_Behaviour`                  |
| Credit Profile     | `Changed_Credit_Limit`, `Num_Credit_Inquiries`, `Credit_Mix`, `Credit_Utilization_Ratio`, `Credit_History_Age` |
| Financial Position | `Outstanding_Debt`, `Total_EMI_per_month`, `Amount_invested_monthly`, `Monthly_Balance`                        |
| **Target**         | **`Credit_Score`**                                                                                             |

---

## 🧹 Data Preprocessing

The dataset required extensive cleaning before modeling.

| Issue                                                                                    | Approach                                            | Reason                                                                                    |
| ---------------------------------------------------------------------------------------- | --------------------------------------------------- | ----------------------------------------------------------------------------------------- |
| Inconsistent missing-value representations (`_`, `-`, `NA`, `null`, `nan`, blanks, etc.) | Converted to missing values                         | Standardizes different representations of missing data                                    |
| `ID`, `Customer_ID`, `Name`, `Month`                                                     | Dropped                                             | Identifiers do not provide meaningful predictive information                              |
| `SSN`                                                                                    | Dropped                                             | Provides no useful predictive information and could encourage memorization of individuals |
| Numeric columns stored as text                                                           | Removed invalid characters and converted to numeric | Ensures numerical features can be processed correctly                                     |
| Unrealistic values in `Age`, `Num_Bank_Accounts`, and `Num_Credit_Card`                  | Replaced with missing values                        | Prevents implausible observations from distorting the model                               |
| `Type_of_Loan`                                                                           | Converted into `Num_Loans`                          | Replaces a multi-value text field with a stable numerical feature                         |
| `Credit_History_Age`                                                                     | Converted into `Credit_History_Age_Months`          | Converts `"X Years and Y Months"` into a single numerical value                           |
| Invalid values in `Occupation`, `Payment_of_Min_Amount`, and `Payment_Behaviour`         | Replaced with missing values                        | Invalid strings were not meaningful categories                                            |
| Missing numerical values                                                                 | Median imputation                                   | Median is more robust to skewed distributions                                             |
| Missing categorical values                                                               | Mode imputation                                     | Preserves the most common category                                                        |
| Duplicate rows                                                                           | None found                                          | No duplicate observations required removal                                                |

---

## 📊 Exploratory Data Analysis

Several exploratory analyses were performed to understand the dataset and identify potential relationships between features and the target.

### Target Distribution

The `Credit_Score` target was label-encoded into three classes:

* **Class 0**
* **Class 1**
* **Class 2**

### Numerical Feature Analysis

The analysis included:

* Histograms of numerical features
* Correlation between numerical features and the target
* A full correlation heatmap
* Examination of feature distributions and potential outliers

The strongest individual correlation with the target was observed for `Changed_Credit_Limit`, although **no feature had a correlation above 0.5**.

---

## 🤖 Modeling

Two tree-based classification algorithms were trained and compared:

* **Random Forest**
* **XGBoost**

### Preprocessing & Pipeline

The modeling workflow uses a scikit-learn `Pipeline` with a `ColumnTransformer`.

* Numerical features are passed directly to the models.
* Categorical features are encoded using `OneHotEncoder`.
* Tree-based models do not require feature scaling.
* The target `Credit_Score` is label-encoded.

### Train/Test Split

The dataset was divided into:

* **80% training**
* **20% testing**
* Stratified by target class
* `random_state=42`

The held-out test set contains **3,365 samples**.

### Hyperparameter Tuning

`RandomizedSearchCV` was used with:

* **5 parameter combinations**
* **3-fold cross-validation**
* Accuracy as the scoring metric

A randomized search was selected instead of a full grid search to keep computation manageable.

| Model             | Hyperparameters                                                                           |
| ----------------- | ----------------------------------------------------------------------------------------- |
| **Random Forest** | `n_estimators`: 100, 200, 300 · `max_depth`: None, 10, 20 · `min_samples_split`: 2, 5, 10 |
| **XGBoost**       | `n_estimators`: 100, 200, 300 · `max_depth`: 3, 5, 7 · `learning_rate`: 0.05, 0.1, 0.2    |

---

## 📈 Results

Both models achieved similar performance on the held-out test set.

### Overall Performance

| Metric               | Random Forest |    XGBoost |
| -------------------- | ------------: | ---------: |
| Accuracy             |        0.7016 | **0.7049** |
| Precision (weighted) |        0.7002 | **0.7054** |
| Recall (weighted)    |        0.7016 | **0.7049** |
| F1-score (weighted)  |        0.7002 | **0.7046** |

### Per-Class F1-Score

| Class | Support | Random Forest |  XGBoost |
| ----- | ------: | ------------: | -------: |
| 0     |     585 |          0.55 | **0.60** |
| 1     |     981 |      **0.71** |     0.70 |
| 2     |   1,799 |          0.74 |     0.74 |

### Key Findings

* Both models achieve approximately **70% accuracy**.
* **XGBoost performs slightly better overall**.
* Class 0 is the most difficult class for both models.
* XGBoost performs better on the smallest class, with its recall improving from approximately **0.54 to 0.61** compared with Random Forest.
* Class 2, the largest class, is the easiest for both models.

---

## 🔍 Feature Importance

Feature importance from the XGBoost model shows that **Credit Mix** is particularly influential.

The most important features include:

1. `Credit_Mix_Good` — approximately 0.15
2. `Credit_Mix_Standard` — approximately 0.12
3. `Outstanding_Debt`
4. `Payment_of_Min_Amount_No`
5. `Interest_Rate`

The importance of the `Credit_Mix` categories stands out compared with most other features.

---

## 🛠️ Tech Stack

* **Python** — programming language
* **pandas** — data manipulation
* **NumPy** — numerical computing
* **Matplotlib** — data visualization
* **Seaborn** — statistical visualization
* **scikit-learn** — preprocessing, pipelines, model training, hyperparameter tuning, and evaluation
* **XGBoost** — gradient boosting classification
* **Jupyter Notebook** — development and analysis environment

---

## 📁 Project Structure

```text
├── CreditScore.ipynb
├── Credis_Score_Dataset_A.csv
└── README.md
```

---

## 🔮 Future Improvements

Potential improvements to the project include:

* Move imputation into the preprocessing pipeline and perform the train/test split before fitting preprocessing steps to further prevent data leakage.
* Perform a broader hyperparameter search using `GridSearchCV` or a larger `RandomizedSearchCV` search space.
* Investigate class imbalance using class weights or resampling techniques.
* Add a confusion matrix and additional class-level evaluation metrics.
* Compare additional classification algorithms.
* Perform more extensive feature engineering.
* Evaluate model performance using additional metrics beyond accuracy.

---

## 📚 References

* [scikit-learn Documentation](https://scikit-learn.org/)
* [XGBoost Documentation](https://xgboost.readthedocs.io/)
* [pandas Documentation](https://pandas.pydata.org/docs/)
* [NumPy Documentation](https://numpy.org/doc/)
