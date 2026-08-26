# Common Machine Learning Workflow Cheat Sheet

This file shows a reusable **tabular supervised-learning workflow**.

Example task:

> Predict whether a customer will churn.

Libraries:

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
```

ML:

```python
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    ConfusionMatrixDisplay,
    roc_auc_score,
)
from sklearn.model_selection import (
    cross_validate,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler,
)
from sklearn.linear_model import LogisticRegression
```

---

# 1. Example data

```python
df = pd.DataFrame({
    "age": [
        24, 45, 31, 52, 38, 29, 63, 41,
        35, 48, 27, 56, 33, 44, 60, 39,
    ],
    "monthly_spend": [
        35, 90, 55, 120, 72, 42, 150, 80,
        65, 105, 39, 130, 60, 88, 145, 76,
    ],
    "tenure_months": [
        2, 36, 10, 60, 24, 4, 72, 30,
        18, 48, 3, 66, 14, 42, 70, 28,
    ],
    "support_tickets": [
        4, 1, 3, 0, 2, 5, 0, 1,
        2, 1, 4, 0, 3, 1, 0, 2,
    ],
    "plan": [
        "basic", "premium", "basic", "premium",
        "standard", "basic", "premium", "standard",
        "standard", "premium", "basic", "premium",
        "standard", "standard", "premium", "standard",
    ],
    "region": [
        "south", "west", "south", "east",
        "west", "south", "east", "west",
        "south", "east", "west", "east",
        "south", "west", "east", "south",
    ],
    "churn": [
        1, 0, 1, 0,
        0, 1, 0, 0,
        1, 0, 1, 0,
        1, 0, 0, 0,
    ],
})
```

For learning purposes this tiny dataset is fine.

For meaningful model evaluation, use substantially more real observations.

---

# 2. Define the question before modeling

Identify:

```text
Prediction target:
    churn

Problem type:
    binary classification

One observation:
    one customer

Positive class:
    churn = 1

Metric:
    ROC AUC + classification metrics
```

Ask:

- What exactly are we predicting?
- At what point in time is the prediction made?
- Which information is actually available at prediction time?
- What mistake is more expensive: false positive or false negative?

This prevents many ML problems before coding starts.

---

# 3. Inspect data

```python
df.head()
df.shape
df.info()
df.describe(include="all")
```

Missing values:

```python
df.isna().sum()
```

Duplicates:

```python
df.duplicated().sum()
```

Target distribution:

```python
df["churn"].value_counts()
df["churn"].value_counts(normalize=True)
```

---

# 4. Quick EDA

Numeric distributions:

```python
numeric_features = [
    "age",
    "monthly_spend",
    "tenure_months",
    "support_tickets",
]

df[numeric_features].hist(
    bins=10,
    figsize=(10, 7),
)

plt.tight_layout()
plt.show()
```

Target balance:

```python
sns.countplot(
    data=df,
    x="churn",
)

plt.show()
```

Numeric feature vs target:

```python
sns.boxplot(
    data=df,
    x="churn",
    y="tenure_months",
)

plt.show()
```

Category vs target:

```python
sns.countplot(
    data=df,
    x="plan",
    hue="churn",
)

plt.show()
```

Correlation:

```python
corr = df.corr(numeric_only=True)

sns.heatmap(
    corr,
    annot=True,
    fmt=".2f",
)

plt.show()
```

---

# 5. Separate features and target

```python
X = df.drop(columns="churn")
y = df["churn"]
```

Never accidentally include the target inside `X`.

---

# 6. Split before fitting learned preprocessing

```python
from sklearn.model_selection import train_test_split
```

```python
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y,
)
```

Why `stratify=y`?

It attempts to preserve the target-class proportions in both splits.

Why split now?

Because preprocessing parameters such as means, medians, category mappings, and scaling statistics should be learned from training data rather than the test set.

---

# 7. Identify feature types

```python
numeric_features = [
    "age",
    "monthly_spend",
    "tenure_months",
    "support_tickets",
]

categorical_features = [
    "plan",
    "region",
]
```

---

# 8. Build preprocessing

```python
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
```

Numeric pipeline:

```python
numeric_transformer = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="median"),
        ),
        (
            "scaler",
            StandardScaler(),
        ),
    ]
)
```

Categorical pipeline:

```python
categorical_transformer = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="most_frequent"),
        ),
        (
            "onehot",
            OneHotEncoder(
                handle_unknown="ignore",
            ),
        ),
    ]
)
```

Combine them:

```python
preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            numeric_transformer,
            numeric_features,
        ),
        (
            "cat",
            categorical_transformer,
            categorical_features,
        ),
    ]
)
```

---

# 9. Build one pipeline: preprocessing + model

```python
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
```

```python
model = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor,
        ),
        (
            "classifier",
            LogisticRegression(
                max_iter=1000,
            ),
        ),
    ]
)
```

This is one of the most important scikit-learn patterns.

The pipeline prevents you from manually applying different preprocessing to train/test data.

---

# 10. Establish a baseline

Before celebrating a model, compare it with something trivial.

For a binary target:

```python
from sklearn.metrics import accuracy_score
```

```python
majority_class = y_train.mode()[0]

baseline_pred = np.full(
    shape=len(y_test),
    fill_value=majority_class,
)

baseline_accuracy = accuracy_score(
    y_test,
    baseline_pred,
)

print(baseline_accuracy)
```

A model that cannot beat a reasonable baseline is not useful just because it runs.

---

# 11. Cross-validate on training data

```python
from sklearn.model_selection import cross_validate
```

```python
cv_results = cross_validate(
    model,
    X_train,
    y_train,
    cv=3,
    scoring=[
        "accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
    ],
)

pd.DataFrame(cv_results)
```

Average:

```python
cv_summary = (
    pd.DataFrame(cv_results)
      .filter(like="test_")
      .mean()
)

print(cv_summary)
```

For a real dataset, `cv=5` is a common starting point.

Tiny example data may require fewer folds because each class must have enough examples.

---

# 12. Fit final model on training data

```python
model.fit(
    X_train,
    y_train,
)
```

---

# 13. Predict

Classes:

```python
y_pred = model.predict(X_test)
```

Probabilities:

```python
y_prob = model.predict_proba(X_test)[:, 1]
```

---

# 14. Evaluate classification

```python
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    roc_auc_score,
)
```

Accuracy:

```python
accuracy = accuracy_score(
    y_test,
    y_pred,
)

print(accuracy)
```

ROC AUC:

```python
auc = roc_auc_score(
    y_test,
    y_prob,
)

print(auc)
```

Precision / recall / F1:

```python
print(
    classification_report(
        y_test,
        y_pred,
        zero_division=0,
    )
)
```

---

# 15. Confusion matrix

```python
from sklearn.metrics import ConfusionMatrixDisplay
```

```python
ConfusionMatrixDisplay.from_predictions(
    y_test,
    y_pred,
)

plt.show()
```

Think in terms of:

```text
True Positive
False Positive
True Negative
False Negative
```

The business cost of these errors may matter more than overall accuracy.

---

# 16. Classification metrics: what they mean

### Accuracy

```text
correct predictions / all predictions
```

Useful when classes and error costs are reasonably balanced.

### Precision

```text
TP / (TP + FP)
```

Question:

> When the model predicts positive, how often is it correct?

### Recall

```text
TP / (TP + FN)
```

Question:

> Of the actual positives, how many did the model find?

### F1

Harmonic mean of precision and recall.

Useful when you care about both.

### ROC AUC

Measures ranking ability across classification thresholds.

Do not choose a metric just because it is popular. Choose one that represents the actual objective.

---

# 17. Adjust classification threshold

Default classification often uses roughly `0.5`.

You can choose another threshold:

```python
threshold = 0.35

y_pred_custom = (
    y_prob >= threshold
).astype(int)
```

Then re-evaluate:

```python
print(
    classification_report(
        y_test,
        y_pred_custom,
        zero_division=0,
    )
)
```

Threshold tuning is useful when false positives and false negatives have different costs.

Do not choose the final threshold by repeatedly optimizing against your final test set.

---

# 18. Inspect model coefficients

For logistic regression:

```python
feature_names = (
    model.named_steps["preprocessor"]
         .get_feature_names_out()
)

coefficients = (
    model.named_steps["classifier"]
         .coef_[0]
)

importance = (
    pd.DataFrame({
        "feature": feature_names,
        "coefficient": coefficients,
    })
    .assign(
        abs_coefficient=lambda d: d["coefficient"].abs()
    )
    .sort_values(
        "abs_coefficient",
        ascending=False,
    )
)

importance.head(20)
```

Be careful:

> A predictive coefficient is not automatically a causal effect.

---

# 19. Model-agnostic permutation importance

```python
from sklearn.inspection import permutation_importance
```

Permutation importance can be used with the complete pipeline:

```python
result = permutation_importance(
    model,
    X_test,
    y_test,
    scoring="roc_auc",
    n_repeats=10,
    random_state=42,
)
```

Create a table:

```python
importance = (
    pd.DataFrame({
        "feature": X_test.columns,
        "importance": result.importances_mean,
    })
    .sort_values(
        "importance",
        ascending=False,
    )
)

print(importance)
```

Interpretation:

> How much does model performance degrade when this feature is randomly shuffled?

Permutation importance still does **not** imply causality.

---

# 20. Error analysis

Create a table of wrong predictions:

```python
errors = X_test.copy()

errors["actual"] = y_test
errors["predicted"] = y_pred
errors["probability"] = y_prob

errors = errors[
    errors["actual"] != errors["predicted"]
]

errors.sort_values(
    "probability",
    ascending=False,
)
```

Look for patterns:

- a particular customer segment
- missing values
- extreme values
- mislabeled rows
- data unavailable at inference time
- systematic false positives/negatives

Error analysis often tells you what to improve next.

---

# 21. Try another model

Example tree model:

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
```

Reuse preprocessing:

```python
rf_model = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor,
        ),
        (
            "classifier",
            RandomForestClassifier(
                n_estimators=300,
                random_state=42,
            ),
        ),
    ]
)
```

Cross-validate it exactly the same way:

```python
rf_scores = cross_validate(
    rf_model,
    X_train,
    y_train,
    cv=3,
    scoring="roc_auc",
)

rf_scores["test_score"].mean()
```

Compare models using the **same splits and metric**.

---

# 22. Hyperparameter search

```python
from sklearn.model_selection import GridSearchCV
```

Example:

```python
param_grid = {
    "classifier__C": [
        0.01,
        0.1,
        1,
        10,
    ],
}

search = GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    scoring="roc_auc",
    cv=3,
    n_jobs=-1,
)

search.fit(
    X_train,
    y_train,
)
```

Results:

```python
search.best_params_
search.best_score_
```

Best pipeline:

```python
best_model = search.best_estimator_
```

Evaluate once on test data:

```python
test_prob = best_model.predict_proba(X_test)[:, 1]

roc_auc_score(
    y_test,
    test_prob,
)
```

For larger search spaces, consider `RandomizedSearchCV`.

---

# 23. Regression variation

If your target is continuous:

```python
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.pipeline import Pipeline
```

Pipeline:

```python
regression_model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("regressor", LinearRegression()),
    ]
)

regression_model.fit(
    X_train,
    y_train,
)

y_pred = regression_model.predict(X_test)
```

Evaluate:

```python
mae = mean_absolute_error(
    y_test,
    y_pred,
)

rmse = mean_squared_error(
    y_test,
    y_pred,
) ** 0.5

r2 = r2_score(
    y_test,
    y_pred,
)

print({
    "MAE": mae,
    "RMSE": rmse,
    "R2": r2,
})
```

---

# 24. Data leakage: critical examples

## Bad: scaling before split

```python
X_scaled = scaler.fit_transform(X)
X_train, X_test = train_test_split(X_scaled)
```

The scaler has already seen test data.

## Better

Put the scaler inside a pipeline and fit the pipeline only on training data.

---

## Bad: target-derived feature

Suppose you predict churn but create:

```python
df["cancelled_account"] = ...
```

If that field becomes known only after churn occurs, it leaks the answer.

---

## Bad: future information

Predicting a customer's status on January 1 using purchases made in February is leakage.

Always ask:

> Would this exact feature value be available at the moment this prediction is supposed to happen?

---

# 25. Time-series warning

For time-dependent data, random train/test splitting can be wrong.

Instead:

```python
train = df[df["date"] < "2026-01-01"]
test = df[df["date"] >= "2026-01-01"]
```

Or investigate:

```python
from sklearn.model_selection import TimeSeriesSplit
```

Do not let future observations influence training for past predictions.

---

# 26. Imbalanced classification

Inspect:

```python
y.value_counts(normalize=True)
```

If positive examples are rare, accuracy can be misleading.

Consider:

```text
precision
recall
F1
ROC AUC
precision-recall AUC
confusion matrix
cost-sensitive evaluation
```

Many classifiers also support:

```python
class_weight="balanced"
```

Example:

```python
LogisticRegression(
    class_weight="balanced",
    max_iter=1000,
)
```

Do not automatically rebalance. First define which mistakes matter.

---

# 27. Reproducibility

Set `random_state` where supported:

```python
random_state = 42
```

Examples:

```python
train_test_split(
    X,
    y,
    random_state=42,
)

RandomForestClassifier(
    random_state=42,
)
```

Also save:

- package versions
- data snapshot/version
- feature definitions
- train/test split policy
- model parameters
- evaluation metric definitions

---

# 28. Save trained pipeline

```python
import joblib
```

Save:

```python
joblib.dump(
    model,
    "churn_model.joblib",
)
```

Load:

```python
model = joblib.load(
    "churn_model.joblib",
)
```

Because preprocessing is inside the pipeline, the loaded object contains both preprocessing and the model.

Only load serialized model files from trusted sources.

---

# 29. Predict new data

```python
new_customers = pd.DataFrame({
    "age": [34],
    "monthly_spend": [58],
    "tenure_months": [8],
    "support_tickets": [3],
    "plan": ["basic"],
    "region": ["south"],
})
```

Probability:

```python
model.predict_proba(
    new_customers,
)[:, 1]
```

Class:

```python
model.predict(
    new_customers,
)
```

The incoming feature schema should match the training schema.

---

# 30. Large dataset strategy

Do **not** immediately load everything into RAM.

Options include:

### Read only needed columns

```python
pd.read_csv(
    "large.csv",
    usecols=required_columns,
)
```

### Use efficient dtypes

```python
pd.read_csv(
    "large.csv",
    dtype={
        "customer_id": "int32",
        "region": "category",
    },
)
```

### Chunk processing

```python
for chunk in pd.read_csv(
    "large.csv",
    chunksize=100_000,
):
    process(chunk)
```

### Incremental learning

Some scikit-learn estimators implement:

```python
model.partial_fit(X_batch, y_batch)
```

Pattern:

```python
for X_batch, y_batch in batches:
    model.partial_fit(
        X_batch,
        y_batch,
        classes=[0, 1],
    )
```

`partial_fit()` updates compatible models batch-by-batch instead of requiring all training data at once.

Not every estimator supports it.

---

# 31. ML workflow to remember

```text
1. DEFINE
   prediction target
   prediction time
   unit of observation
   success metric

2. LOAD
   data

3. VALIDATE
   schema
   types
   missing values
   duplicates
   impossible values

4. EXPLORE
   distributions
   target balance
   relationships
   outliers

5. SPLIT
   train / validation / test as appropriate

6. PREPROCESS
   numeric missing values
   categorical missing values
   scaling if needed
   categorical encoding

7. BASELINE
   simple rule / majority / mean

8. PIPELINE
   preprocessing + model

9. CROSS-VALIDATE
   training data

10. TUNE
    using CV / validation data

11. EVALUATE
    untouched test data

12. ERROR ANALYSIS
    inspect failures

13. INTERPRET
    feature effects carefully
    prediction != causation

14. SAVE
    complete pipeline

15. MONITOR
    input drift
    performance
    calibration
    failures
```

---

# 32. Compact classification template

```python
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


target = "churn"

X = df.drop(columns=target)
y = df[target]

numeric_features = [
    "age",
    "monthly_spend",
    "tenure_months",
    "support_tickets",
]

categorical_features = [
    "plan",
    "region",
]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)

numeric_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])

categorical_transformer = Pipeline([
    (
        "imputer",
        SimpleImputer(strategy="most_frequent"),
    ),
    (
        "onehot",
        OneHotEncoder(handle_unknown="ignore"),
    ),
])

preprocessor = ColumnTransformer([
    (
        "num",
        numeric_transformer,
        numeric_features,
    ),
    (
        "cat",
        categorical_transformer,
        categorical_features,
    ),
])

model = Pipeline([
    ("preprocessor", preprocessor),
    (
        "classifier",
        LogisticRegression(max_iter=1000),
    ),
])

model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

print(classification_report(y_test, y_pred))
print("ROC AUC:", roc_auc_score(y_test, y_prob))
```

---

# 33. What to memorize vs look up

## Memorize concepts

```text
train/test separation
data leakage
cross-validation
baseline
feature/target distinction
classification vs regression
precision vs recall
bias/variance
overfitting
regularization
pipeline
categorical encoding
scaling
missing-data strategy
```

## Look up syntax

```text
exact train_test_split arguments
ColumnTransformer syntax
OneHotEncoder options
metric function signatures
plotting arguments
hyperparameter names
```

That is what cheat sheets are for.

## Official docs

- NumPy: https://numpy.org/doc/stable/
- pandas: https://pandas.pydata.org/docs/
- Matplotlib: https://matplotlib.org/stable/
- Seaborn: https://seaborn.pydata.org/
- scikit-learn: https://scikit-learn.org/stable/
