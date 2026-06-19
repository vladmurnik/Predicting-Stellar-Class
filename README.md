# Astronomical Object Classification with XGBoost

## 🏆 Best Public Leaderboard Score

**0.95769**

**Model:** XGBoost + Astronomical Feature Engineering + Polynomial Features

---

## Overview

This repository contains an end-to-end machine learning pipeline for astronomical object classification using photometric measurements, celestial coordinates, and redshift information. The project combines astrophysics-inspired feature engineering with GPU-accelerated XGBoost to achieve strong predictive performance.

The workflow includes:

* Exploratory Data Analysis (EDA)
* Feature Engineering
* Data Preprocessing
* Model Training
* Evaluation
* Model Saving
* Prediction Generation
* Submission Creation

---

## Dataset

Expected structure:

```text
data/
├── train.csv
└── test.csv
```

### Features

| Feature       | Description            |
| ------------- | ---------------------- |
| u, g, r, i, z | Photometric magnitudes |
| redshift      | Cosmological redshift  |
| alpha         | Right Ascension        |
| delta         | Declination            |
| id            | Object identifier      |

### Target

```text
class
```

---

## Feature Engineering

To improve model performance, several domain-specific features are generated.

### Color Indices

```text
u_g
g_r
r_i
i_z
u_r
g_z
```

These features capture spectral and physical properties of celestial objects.

### Redshift Features

```python
redshift_log = np.log1p(redshift)
redshift_sq = redshift ** 2
```

### Cyclic Coordinate Encoding

```python
alpha_sin
alpha_cos
delta_sin
delta_cos
```

This preserves angular relationships and removes coordinate discontinuities.

### Data Cleaning

* Safe numeric conversion
* Infinite value handling
* Automatic ID removal

---

## Preprocessing

The pipeline uses Scikit-Learn's `ColumnTransformer` and `Pipeline`.

### Numerical Features

```python
SimpleImputer(strategy="median")
```

### Categorical Features

```python
SimpleImputer(strategy="most_frequent")
OneHotEncoder(
    handle_unknown="ignore",
    min_frequency=0.01
)
```

### Polynomial Features

```python
PolynomialFeatures(
    degree=2,
    include_bias=False
)
```

This allows the model to learn higher-order feature interactions.

---

## Model

The final classifier is an XGBoost model trained on GPU:

```python
XGBClassifier(
    tree_method="hist",
    device="cuda",
    objective="multi:softprob",
    num_class=3,
    n_estimators=3000,
    learning_rate=0.02,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=2.0,
    random_state=42
)
```

### Why XGBoost?

* Excellent performance on tabular data
* Handles non-linear relationships
* Strong regularization
* GPU acceleration support
* Robust handling of missing values

---

## Evaluation

The framework automatically computes:

* Accuracy
* Precision
* Recall
* F1 Score (Macro & Weighted)
* ROC-AUC (OVR)
* Log Loss
* Classification Report
* Confusion Matrix

Reports are saved automatically for experiment tracking.

---

## Results

The final solution achieved a public leaderboard score of:

# **0.95769**

### Performance Summary

| Metric              | Value                        |
| ------------------- | ---------------------------- |
| Public Score        | **0.95769**                  |
| Model               | XGBoost                      |
| Feature Engineering | Custom Astronomical Features |
| Polynomial Features | Degree 2                     |
| Hardware            | CUDA GPU                     |

The strong result demonstrates the effectiveness of combining domain knowledge with gradient boosting techniques.

---

## Model Saving

Models are automatically stored using Joblib:

```python
joblib.dump(model, "models/XGB_v5.pkl")
```

Saved models can later be loaded for inference without retraining.

---

## Prediction & Submission

Generate predictions:

```python
preds = pipeline.predict(test_data)
```

Create submission:

```csv
id,class
0,GALAXY
1,STAR
2,QSO
```

The resulting `submission.csv` can be submitted directly to Kaggle.

---

## Project Structure

```text
project/
│
├── data/
│   ├── train.csv
│   └── test.csv
│
├── models/
│   └── XGB_v5.pkl
│
├── notebooks/
│   └── training.ipynb
│
├── submission.csv
│
└── README.md
```

---

## Technologies

* Python
* Pandas
* NumPy
* Scikit-Learn
* XGBoost
* Matplotlib
* Seaborn
* Joblib
