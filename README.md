# Python Data Science & ML Cheatsheets

Practical, example-first references for everyday data science and machine learning work.

The goal is **not to memorize APIs**. Remember the concepts and patterns, then search these files for the syntax you need.

## Cheat sheets

| File | What it covers |
|---|---|
| [`numpy-cheatsheet.md`](numpy-cheatsheet.md) | Arrays, indexing, shapes, vectorization, broadcasting, statistics, random numbers, linear algebra |
| [`pandas-cheatsheet.md`](pandas-cheatsheet.md) | Loading, inspecting, selecting, filtering, cleaning, grouping, joining, reshaping, dates, strings |
| [`matplotlib-cheatsheet.md`](matplotlib-cheatsheet.md) | Figures, axes, line/scatter/bar/hist plots, labels, annotations, layouts, saving |
| [`seaborn-cheatsheet.md`](seaborn-cheatsheet.md) | Statistical plots, distributions, categorical plots, regression, heatmaps, faceting |
| [`ml-workflow-cheatsheet.md`](ml-workflow-cheatsheet.md) | End-to-end tabular ML workflow with example data and scikit-learn |

## Typical imports

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
```

For ML:

```python
from sklearn.model_selection import train_test_split
```

## Suggested workflow

1. Load data with pandas.
2. Inspect shape, types, missing values, and target balance.
3. Explore distributions and relationships with Matplotlib/Seaborn.
4. Split train/test **before learning preprocessing statistics**.
5. Build preprocessing and model steps into a scikit-learn `Pipeline`.
6. Cross-validate on training data.
7. Evaluate once on the untouched test set.
8. Inspect errors and iterate.

## Official references

- NumPy: https://numpy.org/doc/stable/
- pandas: https://pandas.pydata.org/docs/
- Matplotlib: https://matplotlib.org/stable/
- Seaborn: https://seaborn.pydata.org/
- scikit-learn: https://scikit-learn.org/stable/

## Philosophy

Use these sheets as a fast lookup layer:

> **Concepts are worth memorizing. Function signatures are worth searching.**
