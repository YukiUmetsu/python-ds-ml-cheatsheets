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

np.zeros((2, 3))         # [[0., 0., 0.], [0., 0., 0.]]
np.ones((2, 3))          # [[1., 1., 1.], [1., 1., 1.]]
np.full((2, 3), 7)       # [[7, 7, 7], [7, 7, 7]]

np.arange(5)             # [0, 1, 2, 3, 4]     stop only
np.arange(2, 8)          # [2, 3, 4, 5, 6, 7]  start, stop
np.arange(0, 10, 2)      # [0, 2, 4, 6, 8]     start, stop, step
# stop is exclusive (like range)

np.linspace(0, 1, 5)     # 5 evenly spaced values from 0 to 1 inclusive

np.eye(3)                # [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
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

`reshape` rearranges the same elements into a new shape. Product of dims must equal `size`.

```python
a = np.arange(12)
# [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
```

2 args → 2D `(rows, cols)`:

```python
a.reshape(3, 4)
# [[ 0,  1,  2,  3],
#  [ 4,  5,  6,  7],
#  [ 8,  9, 10, 11]]
```

3 args → 3D `(depth, rows, cols)` (or any 3 dims):

```python
a.reshape(2, 2, 3)
# [[[ 0,  1,  2],
#   [ 3,  4,  5]],
#
#  [[ 6,  7,  8],
#   [ 9, 10, 11]]]
# shape (2, 2, 3): 2 blocks, each 2x3
```

`-1` means “infer this dimension”:

```python
a.reshape(-1, 1)   # (12, 1) column vector
a.reshape(1, -1)   # (1, 12) row vector
a.reshape(2, -1)   # (2, 6)
a.reshape(2, 2, -1)  # (2, 2, 3)
```

### ravel vs flatten

Both turn the array into 1D. Difference: view vs copy.

```python
m = np.array([
    [1, 2, 3],
    [4, 5, 6],
])

m.ravel()
# [1, 2, 3, 4, 5, 6]   # usually a view (shares memory)

m.flatten()
# [1, 2, 3, 4, 5, 6]   # always a copy
```

Changing a ravel result can change `m`; changing a flatten result does not:

```python
r = m.ravel()
r[0] = 99
m
# [[99, 2, 3],
#  [ 4, 5, 6]]

f = m.flatten()
f[0] = 0
m
# [[99, 2, 3],
#  [ 4, 5, 6]]   # unchanged by f
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

### Transpose (2D)

Rows become columns:

```python
m = np.array([
    [1, 2, 3],
    [4, 5, 6],
])
# shape (2, 3)

m.T
# [[1, 4],
#  [2, 5],
#  [3, 6]]   shape (3, 2)

np.transpose(m)   # same as m.T for 2D
```

### Swap axes (any ndim)

`swapaxes(i, j)` exchanges two axes. For 2D, `swapaxes(0, 1)` == transpose:

```python
m.swapaxes(0, 1)   # same as m.T
```

3D example — think `(batch, rows, cols)`:

```python
x = np.arange(24).reshape(2, 3, 4)
# shape (2, 3, 4)

x.swapaxes(1, 2)
# shape (2, 4, 3)  — rows <-> cols within each batch

np.transpose(x, axes=(0, 2, 1))
# same result: reorder axes explicitly
```

`T` / default `transpose` reverse all axes. Prefer `swapaxes` / `transpose(..., axes=...)` when you only want specific axes moved.

---

## 10. Combine arrays

### Concatenate (join along an existing axis)

```python
a = np.array([1, 2])
b = np.array([3, 4])

np.concatenate([a, b])
# [1, 2, 3, 4]
```

2D:

```python
x = np.array([[1, 2], [3, 4]])
y = np.array([[5, 6], [7, 8]])

np.concatenate([x, y], axis=0)  # add rows  -> (4, 2)
np.concatenate([x, y], axis=1)  # add cols  -> (2, 4)
```

### Stack (join along a *new* axis)

```python
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

np.stack([a, b])
# [[1, 2, 3],
#  [4, 5, 6]]          shape (2, 3)  — new axis=0

np.stack([a, b], axis=1)
# [[1, 4],
#  [2, 5],
#  [3, 6]]              shape (3, 2)  — new axis=1
```

### vstack / hstack

```python
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

np.vstack([a, b])
# [[1, 2, 3],
#  [4, 5, 6]]          # stack as rows

np.hstack([a, b])
# [1, 2, 3, 4, 5, 6]   # side by side (1D)
```

2D:

```python
x = np.array([[1, 2], [3, 4]])
y = np.array([[5, 6], [7, 8]])

np.vstack([x, y])   # same as concatenate(..., axis=0)
# [[1, 2],
#  [3, 4],
#  [5, 6],
#  [7, 8]]

np.hstack([x, y])   # same as concatenate(..., axis=1)
# [[1, 2, 5, 6],
#  [3, 4, 7, 8]]
```

Rule of thumb:

- `concatenate` — same ndim, join on existing axis
- `stack` — create a new axis
- `vstack` / `hstack` — shortcuts for vertical / horizontal join

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

Reduce many values to a summary. Methods like `a.sum()` exist on arrays; some only as functions (`np.median`).

```python
a = np.array([1., 2., 3., 4.])

a.sum()       # 10.0   total
a.mean()      # 2.5    average
np.median(a)  # 2.5    middle value (no a.median())
a.min()       # 1.0
a.max()       # 4.0
a.std()       # spread (population by default for ndarray)
a.var()       # variance

a.argmin()    # index of min -> 0
a.argmax()    # index of max -> 3
```

Also useful:

```python
a.prod()                    # product of all elements
np.percentile(a, [25, 50, 75])
np.quantile(a, [0.25, 0.5, 0.75])
```

### By axis

`axis` is the dimension you collapse:

```python
X = np.array([
    [1, 2, 3],
    [4, 5, 6],
])

X.sum(axis=0)   # [5, 7, 9]  down rows -> one per column
X.sum(axis=1)   # [6, 15]    across cols -> one per row

X.mean(axis=0)
X.std(axis=0)
X.argmax(axis=1)  # [2, 2]  index of max in each row
```

`keepdims=True` keeps the reduced axis as size 1 (handy for broadcasting):

```python
X.mean(axis=1, keepdims=True)
# [[2.],
#  [5.]]   shape (2, 1)
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
a = np.array([3, 1, 2, 1, 3])

np.unique(a)
# [1, 2, 3]

values, counts = np.unique(a, return_counts=True)
# values: [1, 2, 3]
# counts: [2, 1, 2]
```

---

## 17. Set operations

Treat 1D arrays like sets (results are sorted unique):

```python
a = np.array([1, 2, 3, 4])
b = np.array([3, 4, 5, 6])

np.intersect1d(a, b)   # in both -> [3, 4]
np.union1d(a, b)       # in either -> [1, 2, 3, 4, 5, 6]
np.setdiff1d(a, b)     # in a but not b -> [1, 2]
np.setdiff1d(b, a)     # in b but not a -> [5, 6]
```

---

## 18. Sort

`np.sort` returns a sorted **copy**. `argsort` returns the indices that would sort.

```python
a = np.array([30, 10, 20])

np.sort(a)
# [10, 20, 30]

a.sort()              # in-place; returns None
```

### Argsort (sort by index)

```python
a = np.array([30, 10, 20])

idx = np.argsort(a)
# [1, 2, 0]           positions of 10, 20, 30

a[idx]
# [10, 20, 30]
```

Use argsort to reorder a related array the same way:

```python
names = np.array(["c", "a", "b"])
scores = np.array([30, 10, 20])

names[np.argsort(scores)]
# ['a', 'b', 'c']
```

### Sort along an axis (2D)

```python
m = np.array([
    [3, 1, 2],
    [9, 7, 8],
])

np.sort(m, axis=1)    # sort within each row
# [[1, 2, 3],
#  [7, 8, 9]]

np.sort(m, axis=0)    # sort within each column
# [[3, 1, 2],
#  [9, 7, 8]]

np.argsort(m, axis=1) # indices within each row
```

Descending:

```python
np.sort(a)[::-1]
a[np.argsort(-a)]     # also works for numeric arrays
```

Partial / nth element:

```python
np.partition(a, 1)    # element at index 1 is in sorted position;
                      # left side smaller, right side larger (unordered)
```

---

## 19. Conditional values

```python
np.where(a > 0, 1, 0)

indices = np.where(a > 10)
```

Clip:

```python
np.clip(a, 0, 100)
```

---

## 20. Missing / infinite values

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

## 21. Compare floating-point values

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

## 22. Random numbers

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

## 23. Statistics

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

## 24. Linear algebra

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

## 25. Save/load

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

## 26. Vectorization

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

## 27. Copies vs views

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

## 28. Useful ML patterns

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

## 29. High-value functions to remember exist

```text
array
asarray
arange
linspace
zeros / ones / full
reshape
transpose / swapaxes
concatenate / stack / vstack / hstack
where
unique
intersect1d / union1d / setdiff1d
sort / argsort / partition
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
