# Employee Turnover Prediction

This project demonstrates a complete end-to-end data analysis process for predicting employee turnover using machine learning models. The primary goal is to identify key factors influencing whether an employee will leave the company, and to build predictive models to forecast turnover with high accuracy.

## Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Data Preprocessing](#data-preprocessing)
- [Feature Selection](#feature-selection)
- [Modeling](#modeling)
- [Evaluation](#evaluation)
- [Feature Importance](#feature-importance)
- [Conclusion](#conclusion)

## Project Overview

In this project, we use the HR dataset to analyze employee turnover and build predictive models using Logistic Regression and Random Forest classifiers. The process includes data preprocessing, feature selection, model training, evaluation, and interpretation of results.

## Dataset

The dataset used in this project is `HR.csv`, which contains various features related to employee satisfaction, work conditions, and personal information.

### Columns in the dataset:

- `satisfaction_level`
- `last_evaluation`
- `number_project`
- `average_montly_hours`
- `time_spend_company`
- `Work_accident`
- `left` (target variable)
- `promotion_last_5years`
- `department`
- `salary`

## Data Preprocessing

1. **Rename Column**: The `sales` column was renamed to `department` for better clarity.
   ```python
   hr = hr.rename(columns={'sales': 'department'})
   ```

2. **Merge Categories**: The `department` column was simplified by merging similar categories.
   ```python
   hr['department'] = np.where(hr['department'] == 'support', 'technical', hr['department'])
   hr['department'] = np.where(hr['department'] == 'IT', 'technical', hr['department'])
   ```

3. **Create Dummy Variables**: Categorical variables (`department` and `salary`) were converted into dummy variables for modeling.
   ```python
   cat_vars = ['department', 'salary']
   for var in cat_vars:
       cat_list = pd.get_dummies(hr[var], prefix=var)
       hr = hr.join(cat_list)
   hr.drop(columns=cat_vars, axis=1, inplace=True)
   ```

## Feature Selection

We used Recursive Feature Elimination (RFE) with Logistic Regression to select the most relevant features for predicting employee turnover.

```python
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
rfe = RFE(model, 10)
rfe = rfe.fit(hr[X], hr[y])
```

## Modeling

Two machine learning models were trained:

1. **Logistic Regression**
   ```python
   from sklearn.linear_model import LogisticRegression
   logreg = LogisticRegression()
   logreg.fit(X_train, y_train)
   ```

2. **Random Forest Classifier**
   ```python
   from sklearn.ensemble import RandomForestClassifier
   rf = RandomForestClassifier()
   rf.fit(X_train, y_train)
   ```

## Evaluation

The models were evaluated using accuracy scores, confusion matrices, and ROC curves.

- **Logistic Regression Accuracy**: 77.1%
- **Random Forest Accuracy**: 97.8%

### Confusion Matrix and Classification Report

- **Random Forest**
  ```python
  sns.heatmap(forest_cm, annot=True, fmt='.2f', xticklabels=["Left", "Stayed"], yticklabels=["Left", "Stayed"])
  plt.title('Random Forest')
  ```

- **Logistic Regression**
  ```python
  sns.heatmap(logreg_cm, annot=True, fmt='.2f', xticklabels=["Left", "Stayed"], yticklabels=["Left", "Stayed"])
  plt.title('Logistic Regression')
  ```

### ROC Curve

Both models' performance was visualized using ROC curves to assess the trade-off between the true positive rate and false positive rate.

## Feature Importance

The Random Forest model's feature importance was calculated to understand which factors most influence employee turnover.

```python
for index in feature_indexes_by_importance:
    print('{}-{:.2f}%'.format(feature_labels[index], (importance[index] *100.0)))
```

Top features influencing turnover:
- `satisfaction_level` (50.65%)
- `time_spend_company` (25.73%)
- `last_evaluation` (19.19%)

## Conclusion

The Random Forest model outperformed Logistic Regression in predicting employee turnover. Key factors such as satisfaction level, time spent in the company, and last evaluation score are critical in determining whether an employee will leave. This analysis provides valuable insights for HR departments to proactively address employee retention.
