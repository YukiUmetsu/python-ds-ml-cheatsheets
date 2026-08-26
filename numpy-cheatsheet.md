# NumPy Cheat Sheet

```python
import numpy as np
```

NumPy is mainly about **fast N-dimensional arrays and vectorized numerical operations**.

---

## 1. Create arrays

```python
a = np.array([1, 2, 3])
b = np.array([[1, 2], [3, 4]])

np.zeros((3, 4))
np.ones((2, 3))
np.full((2, 3), 7)

np.arange(0, 10, 2)      # 0, 2, 4, 6, 8
np.linspace(0, 1, 5)     # 5 evenly spaced values

np.eye(3)                 # identity matrix
```

### Create from existing data

```python
values = [10, 20, 30]
a = np.asarray(values)

a = np.array(values, dtype=np.float64)
```

---

## 2. Inspect an array

```python
a.shape        # dimensions, e.g. (100, 5)
a.ndim         # number of dimensions
a.size         # total number of elements
a.dtype        # element type
a.itemsize     # bytes per element
```

---

## 3. Change dtype

```python
a.astype(np.float64)
a.astype(np.int64)
a.astype(bool)
```

---

## 4. Index and slice

```python
a = np.array([10, 20, 30, 40, 50])

a[0]           # first
a[-1]          # last
a[1:4]         # positions 1,2,3
a[:3]
a[::2]         # every second element
```

### 2D

```python
m = np.array([
    [1, 2, 3],
    [4, 5, 6],
])

m[0, 1]        # row 0, column 1
m[0, :]        # first row
m[:, 1]        # second column
m[:, :2]       # first two columns
```

---

## 5. Boolean filtering

```python
a = np.array([1, 5, 8, 12, 20])

a[a > 10]
a[(a >= 5) & (a <= 12)]
a[(a < 5) | (a > 15)]
```

### Create a mask

```python
mask = a > 10
a[mask]
```

### Replace conditionally

```python
a[a < 0] = 0
```

---

## 6. Fancy indexing

```python
a = np.array([10, 20, 30, 40])

a[[0, 2, 3]]
```

---

## 7. Reshape

```python
a = np.arange(12)

a.reshape(3, 4)
a.reshape(2, 2, 3)

a.reshape(-1, 1)   # column vector
a.reshape(1, -1)   # row vector

a.ravel()          # flatten, view when possible
a.flatten()        # flatten, copy
```

---

## 8. Add/remove dimensions

```python
a = np.array([1, 2, 3])

a[:, np.newaxis]       # shape (3, 1)
a[np.newaxis, :]       # shape (1, 3)

np.expand_dims(a, axis=0)
np.squeeze(np.array([[[1], [2]]]))
```

---

## 9. Transpose / swap axes

```python
m.T
np.transpose(m)
```

---

## 10. Combine arrays

```python
a = np.array([1, 2])
b = np.array([3, 4])

np.concatenate([a, b])
np.stack([a, b])            # new axis
np.vstack([a, b])           # vertical
np.hstack([a, b])           # horizontal
```

For 2D arrays:

```python
np.concatenate([x, y], axis=0)  # add rows
np.concatenate([x, y], axis=1)  # add columns
```

---

## 11. Split arrays

```python
np.split(a, 2)
np.array_split(a, 3)        # permits uneven sizes
```

---

## 12. Arithmetic

```python
a + b
a - b
a * b          # elementwise
a / b
a ** 2

np.sqrt(a)
np.exp(a)
np.log(a)
np.abs(a)
```

### Matrix multiplication

```python
A @ B
np.matmul(A, B)
```

`*` is elementwise. `@` is matrix multiplication.

---

## 13. Broadcasting

```python
a = np.array([1, 2, 3])
a + 10
```

2D + 1D:

```python
X = np.array([
    [1, 2, 3],
    [4, 5, 6],
])

offset = np.array([10, 20, 30])

X + offset
```

Column-wise broadcasting:

```python
a = np.array([1, 2, 3])
b = np.array([10, 20])

a[:, None] + b
# shape: (3, 2)
```

### Rule of thumb

Compare shapes from the **right**. Dimensions are compatible when they are:

- equal, or
- one of them is `1`.

---

## 14. Aggregations

```python
a.sum()
a.mean()
a.median()             # ERROR: ndarray has no median method
np.median(a)

a.min()
a.max()
a.std()
a.var()

a.argmin()
a.argmax()
```

### By axis

```python
X.sum(axis=0)      # aggregate down rows -> one value per column
X.sum(axis=1)      # aggregate across columns -> one value per row

X.mean(axis=0)
X.std(axis=0)
```

---

## 15. Cumulative operations

```python
np.cumsum(a)
np.cumprod(a)
```

---

## 16. Unique values and counts

```python
np.unique(a)

values, counts = np.unique(a, return_counts=True)
```

---

## 17. Sort

```python
np.sort(a)

indices = np.argsort(a)
a[indices]
```

---

## 18. Conditional values

```python
np.where(a > 0, 1, 0)

indices = np.where(a > 10)
```

Clip:

```python
np.clip(a, 0, 100)
```

---

## 19. Missing / infinite values

```python
np.isnan(a)
np.isfinite(a)
np.isinf(a)

np.nanmean(a)
np.nansum(a)
np.nanmedian(a)
```

Replace NaN:

```python
np.nan_to_num(a, nan=0.0)
```

---

## 20. Compare floating-point values

Avoid:

```python
a == b
```

Prefer:

```python
np.isclose(a, b)
np.allclose(A, B)
```

---

## 21. Random numbers

Use the modern generator API:

```python
rng = np.random.default_rng(42)

rng.random(5)
rng.integers(0, 10, size=5)
rng.normal(loc=0, scale=1, size=1000)
rng.choice([10, 20, 30], size=5)

rng.shuffle(a)
```

Reproducibility:

```python
rng = np.random.default_rng(42)
```

---

## 22. Statistics

```python
np.mean(a)
np.median(a)
np.std(a)
np.var(a)

np.percentile(a, [25, 50, 75])
np.quantile(a, [0.25, 0.5, 0.75])

np.corrcoef(x, y)
```

---

## 23. Linear algebra

```python
A @ B

np.linalg.solve(A, b)
np.linalg.inv(A)       # usually avoid when solve() is enough
np.linalg.det(A)
np.linalg.eig(A)
np.linalg.svd(A)
np.linalg.norm(a)
```

Prefer:

```python
x = np.linalg.solve(A, b)
```

over:

```python
x = np.linalg.inv(A) @ b
```

---

## 24. Save/load

Binary NumPy format:

```python
np.save("array.npy", a)
a = np.load("array.npy")
```

Multiple arrays:

```python
np.savez("data.npz", x=x, y=y)

data = np.load("data.npz")
x = data["x"]
```

Text:

```python
np.savetxt("data.csv", X, delimiter=",")
X = np.loadtxt("data.csv", delimiter=",")
```

---

## 25. Vectorization

Avoid Python loops when a vectorized operation expresses the same calculation.

Instead of:

```python
result = []
for x in a:
    result.append(x * 2 + 1)
```

Use:

```python
result = a * 2 + 1
```

---

## 26. Copies vs views

Slices often share memory:

```python
a = np.array([1, 2, 3])
b = a[:2]

b[0] = 999
# a may also change
```

Explicit copy:

```python
b = a[:2].copy()
```

---

## 27. Useful ML patterns

### Standardize manually

```python
X_scaled = (X - X.mean(axis=0)) / X.std(axis=0)
```

For real ML pipelines, prefer `StandardScaler` so preprocessing is fitted only on training data.

### Min-max scale manually

```python
X_scaled = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
```

### Euclidean distance

```python
distance = np.linalg.norm(a - b)
```

### Mean squared error

```python
mse = np.mean((y_true - y_pred) ** 2)
```

### Accuracy

```python
accuracy = np.mean(y_true == y_pred)
```

---

## 28. High-value functions to remember exist

```text
array
asarray
arange
linspace
zeros / ones / full
reshape
concatenate / stack
where
unique
sort / argsort
mean / median / std / percentile
sum / min / max / argmin / argmax
isnan / isfinite
clip
all / any
isclose / allclose
random.default_rng
linalg.solve / norm
```

## Official docs

- https://numpy.org/doc/stable/
