# Matplotlib Cheat Sheet

```python
import matplotlib.pyplot as plt
import numpy as np
```

For reusable code, prefer the **object-oriented API**:

```python
fig, ax = plt.subplots()
ax.plot(x, y)
plt.show()
```

---

## 1. Basic line plot

```python
fig, ax = plt.subplots()

ax.plot(x, y)

plt.show()
```

---

## 2. Figure size

```python
fig, ax = plt.subplots(figsize=(8, 5))
```

---

## 3. Multiple lines

```python
fig, ax = plt.subplots()

ax.plot(x, y1, label="Train")
ax.plot(x, y2, label="Validation")

ax.legend()
plt.show()
```

---

## 4. Scatter plot

```python
fig, ax = plt.subplots()

ax.scatter(df["age"], df["income"])

plt.show()
```

Map another variable to point size:

```python
ax.scatter(
    df["age"],
    df["income"],
    s=df["customers"] * 5,
)
```

---

## 5. Bar chart

```python
fig, ax = plt.subplots()

ax.bar(categories, values)

plt.show()
```

Horizontal:

```python
ax.barh(categories, values)
```

---

## 6. Histogram

```python
fig, ax = plt.subplots()

ax.hist(df["age"], bins=20)

plt.show()
```

---

## 7. Box plot

```python
fig, ax = plt.subplots()

ax.boxplot(df["sales"].dropna())

plt.show()
```

---

## 8. Titles and labels

```python
ax.set_title("Monthly Sales")
ax.set_xlabel("Month")
ax.set_ylabel("Revenue ($)")
```

---

## 9. Axis limits

```python
ax.set_xlim(0, 100)
ax.set_ylim(0, 1000)
```

---

## 10. Grid

```python
ax.grid(True)
```

---

## 11. Legend

```python
ax.plot(x, y1, label="Model A")
ax.plot(x, y2, label="Model B")

ax.legend()
```

Position:

```python
ax.legend(loc="best")
```

---

## 12. Reference lines

Horizontal:

```python
ax.axhline(y=0)
```

Vertical:

```python
ax.axvline(x=50)
```

Span:

```python
ax.axvspan(20, 30, alpha=0.2)
```

---

## 13. Text / annotation

Simple text:

```python
ax.text(10, 50, "Interesting point")
```

Annotation with arrow:

```python
ax.annotate(
    "Peak",
    xy=(x_peak, y_peak),
    xytext=(x_peak + 5, y_peak + 10),
    arrowprops={"arrowstyle": "->"},
)
```

---

## 14. Ticks

Set tick positions:

```python
ax.set_xticks([0, 10, 20, 30])
```

Set labels:

```python
ax.set_xticks(
    [0, 1, 2],
    labels=["Low", "Medium", "High"],
)
```

Rotate:

```python
ax.tick_params(axis="x", labelrotation=45)
```

---

## 15. Log scale

```python
ax.set_xscale("log")
ax.set_yscale("log")
```

---

## 16. Multiple plots

```python
fig, axes = plt.subplots(
    nrows=2,
    ncols=2,
    figsize=(10, 8),
)
```

Access:

```python
axes[0, 0].plot(x, y)
axes[0, 1].scatter(x, y)
axes[1, 0].hist(y)
```

---

## 17. Share axes

```python
fig, axes = plt.subplots(
    2,
    1,
    sharex=True,
)
```

---

## 18. Tight layout

```python
fig.tight_layout()
```

Or:

```python
fig, ax = plt.subplots(constrained_layout=True)
```

---

## 19. Plot pandas data

```python
fig, ax = plt.subplots()

ax.plot(df["date"], df["sales"])

plt.show()
```

---

## 20. Plot a mathematical function

```python
x = np.linspace(0, 2 * np.pi, 200)
y = np.sin(x)

fig, ax = plt.subplots()
ax.plot(x, y)
plt.show()
```

---

## 21. Error bars

```python
ax.errorbar(
    x,
    y,
    yerr=errors,
    fmt="o",
)
```

---

## 22. Fill area

```python
ax.fill_between(x, lower, upper, alpha=0.2)
```

Useful for confidence intervals.

---

## 23. Image / matrix

```python
fig, ax = plt.subplots()

im = ax.imshow(matrix)
fig.colorbar(im, ax=ax)

plt.show()
```

---

## 24. Save figure

```python
fig.savefig(
    "plot.png",
    dpi=300,
    bbox_inches="tight",
)
```

SVG:

```python
fig.savefig(
    "plot.svg",
    bbox_inches="tight",
)
```

---

## 25. Close figures in loops/scripts

```python
plt.close(fig)
```

Useful when generating many figures.

---

## 26. Common ML plots

### Actual vs predicted

```python
fig, ax = plt.subplots()

ax.scatter(y_true, y_pred)
ax.set_xlabel("Actual")
ax.set_ylabel("Predicted")
ax.set_title("Actual vs Predicted")

plt.show()
```

### Training curve

```python
fig, ax = plt.subplots()

ax.plot(history["train_loss"], label="Train")
ax.plot(history["val_loss"], label="Validation")

ax.set_xlabel("Epoch")
ax.set_ylabel("Loss")
ax.legend()

plt.show()
```

### Feature importance

```python
order = np.argsort(importances)

fig, ax = plt.subplots()

ax.barh(
    feature_names[order],
    importances[order],
)

ax.set_title("Feature Importance")
plt.show()
```

---

## 27. State-based pyplot shorthand

Fine for notebooks and quick exploration:

```python
plt.figure(figsize=(8, 5))
plt.plot(x, y)
plt.title("Example")
plt.xlabel("X")
plt.ylabel("Y")
plt.show()
```

For larger codebases, prefer:

```python
fig, ax = plt.subplots()
...
```

---

## 28. High-value methods to remember exist

```text
plt.subplots

ax.plot
ax.scatter
ax.bar / barh
ax.hist
ax.boxplot
ax.imshow

ax.set_title
ax.set_xlabel / set_ylabel
ax.set_xlim / set_ylim
ax.set_xscale / set_yscale
ax.legend
ax.grid
ax.axhline / axvline
ax.annotate

fig.tight_layout
fig.savefig

plt.show
plt.close
```

## Official docs

- https://matplotlib.org/stable/
