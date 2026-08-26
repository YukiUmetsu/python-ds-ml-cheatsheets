# Seaborn Cheat Sheet

```python
import seaborn as sns
import matplotlib.pyplot as plt
```

Seaborn provides a high-level statistical plotting API and works especially well with pandas DataFrames.

A useful mental model:

- `x`, `y` -> axes
- `hue` -> another variable represented visually
- `col`, `row` -> facet into multiple plots
- `data=df` -> DataFrame containing those variables

---

## 1. Histogram

```python
sns.histplot(
    data=df,
    x="age",
    bins=20,
)

plt.show()
```

By category:

```python
sns.histplot(
    data=df,
    x="age",
    hue="segment",
)
```

---

## 2. KDE / density

```python
sns.kdeplot(
    data=df,
    x="income",
)

plt.show()
```

Compare groups:

```python
sns.kdeplot(
    data=df,
    x="income",
    hue="segment",
)
```

---

## 3. ECDF

```python
sns.ecdfplot(
    data=df,
    x="response_time",
    hue="service",
)
```

Very useful for comparing distributions without choosing histogram bins.

---

## 4. Scatter plot

```python
sns.scatterplot(
    data=df,
    x="age",
    y="income",
)

plt.show()
```

Add category:

```python
sns.scatterplot(
    data=df,
    x="age",
    y="income",
    hue="segment",
)
```

Add size:

```python
sns.scatterplot(
    data=df,
    x="age",
    y="income",
    hue="segment",
    size="orders",
)
```

---

## 5. Line plot

```python
sns.lineplot(
    data=df,
    x="date",
    y="sales",
)

plt.show()
```

Multiple groups:

```python
sns.lineplot(
    data=df,
    x="date",
    y="sales",
    hue="region",
)
```

---

## 6. Count plot

Count categorical values:

```python
sns.countplot(
    data=df,
    x="category",
)

plt.show()
```

By another category:

```python
sns.countplot(
    data=df,
    x="category",
    hue="converted",
)
```

---

## 7. Bar plot

Shows an **aggregated estimate** for numeric `y`, not raw row counts:

```python
sns.barplot(
    data=df,
    x="region",
    y="sales",
)
```

If you simply want category counts, use `countplot()`.

---

## 8. Box plot

```python
sns.boxplot(
    data=df,
    x="segment",
    y="income",
)

plt.show()
```

Useful for:

- median
- quartiles
- spread
- potential outliers

---

## 9. Violin plot

```python
sns.violinplot(
    data=df,
    x="segment",
    y="income",
)

plt.show()
```

Shows more distribution shape than a box plot.

---

## 10. Strip plot

Individual observations:

```python
sns.stripplot(
    data=df,
    x="segment",
    y="income",
)
```

---

## 11. Pair plot

Quick multivariate EDA:

```python
sns.pairplot(
    df,
    vars=["age", "income", "sales"],
)

plt.show()
```

Color by target:

```python
sns.pairplot(
    df,
    vars=["age", "income", "sales"],
    hue="churn",
)
```

Useful for quickly spotting:

- correlations
- clusters
- separation by target
- outliers

---

## 12. Joint plot

```python
sns.jointplot(
    data=df,
    x="age",
    y="income",
)
```

Regression:

```python
sns.jointplot(
    data=df,
    x="age",
    y="income",
    kind="reg",
)
```

---

## 13. Regression plot

```python
sns.regplot(
    data=df,
    x="ad_spend",
    y="sales",
)
```

---

## 14. `lmplot`

Regression + faceting:

```python
sns.lmplot(
    data=df,
    x="ad_spend",
    y="sales",
    hue="region",
)
```

---

## 15. Correlation heatmap

```python
corr = df.corr(numeric_only=True)

sns.heatmap(
    corr,
    annot=True,
    fmt=".2f",
)

plt.show()
```

For many columns, annotation may be too noisy:

```python
sns.heatmap(corr)
```

---

## 16. Confusion matrix heatmap

```python
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()
```

---

## 17. Faceting

Figure-level distribution:

```python
sns.displot(
    data=df,
    x="sales",
    col="region",
)
```

Rows + columns:

```python
sns.displot(
    data=df,
    x="sales",
    row="segment",
    col="region",
)
```

---

## 18. Figure-level vs axes-level functions

### Axes-level

Plot into an existing Matplotlib `Axes`:

```text
scatterplot
lineplot
histplot
kdeplot
boxplot
violinplot
barplot
countplot
heatmap
regplot
```

Example:

```python
fig, ax = plt.subplots(figsize=(8, 5))

sns.scatterplot(
    data=df,
    x="age",
    y="income",
    ax=ax,
)

ax.set_title("Income by Age")
plt.show()
```

### Figure-level

Create/manage their own figure:

```text
relplot
displot
catplot
lmplot
jointplot
pairplot
```

Useful for faceting.

---

## 19. `relplot`

Relational figure-level plot:

```python
sns.relplot(
    data=df,
    x="age",
    y="income",
    hue="segment",
    col="region",
    kind="scatter",
)
```

---

## 20. `displot`

Distribution figure-level plot:

```python
sns.displot(
    data=df,
    x="sales",
    hue="segment",
    col="region",
    kind="hist",
)
```

---

## 21. `catplot`

Categorical figure-level plot:

```python
sns.catplot(
    data=df,
    x="region",
    y="sales",
    col="segment",
    kind="box",
)
```

Possible `kind` values include commonly used categorical plot types such as:

```text
strip
swarm
box
violin
boxen
point
bar
count
```

---

## 22. Order categories

```python
sns.boxplot(
    data=df,
    x="risk",
    y="loss",
    order=["low", "medium", "high"],
)
```

---

## 23. Rotate labels

Use Matplotlib after the Seaborn call:

```python
ax = sns.countplot(
    data=df,
    x="category",
)

ax.tick_params(
    axis="x",
    labelrotation=45,
)
```

---

## 24. Titles and labels

Axes-level:

```python
fig, ax = plt.subplots()

sns.scatterplot(
    data=df,
    x="age",
    y="income",
    ax=ax,
)

ax.set(
    title="Age vs Income",
    xlabel="Age",
    ylabel="Income",
)
```

---

## 25. Quick EDA recipes

### Numeric distribution

```python
sns.histplot(
    data=df,
    x="feature",
    kde=True,
)
```

### Feature vs numeric target

```python
sns.scatterplot(
    data=df,
    x="feature",
    y="target",
)
```

### Category vs numeric target

```python
sns.boxplot(
    data=df,
    x="category",
    y="target",
)
```

### Category vs binary target

```python
sns.countplot(
    data=df,
    x="category",
    hue="target",
)
```

### Numeric correlations

```python
sns.heatmap(
    df.corr(numeric_only=True),
    annot=True,
    fmt=".2f",
)
```

---

## 26. High-value functions to remember exist

```text
histplot
kdeplot
ecdfplot

scatterplot
lineplot

countplot
barplot
boxplot
violinplot
stripplot

regplot
lmplot

heatmap
pairplot
jointplot

relplot
displot
catplot
```

## Official docs

- https://seaborn.pydata.org/
