# PyTorch Cheat Sheet

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
```

PyTorch is mainly about **tensors, automatic differentiation, and neural network modules**.

### What is a tensor?

A **tensor** is a multi-dimensional array of numbers (like a NumPy `ndarray`), plus:

- a **dtype** (e.g. `float32`) and **device** (`cpu` / `cuda` / `mps`)
- optional **autograd** tracking (`requires_grad=True`) so gradients can flow through ops

Scalars, vectors, matrices, and higher-dim arrays are all tensors: shape `()` / `(n,)` / `(m, n)` / …

---

## 1. Create tensors

```python
torch.tensor([1, 2, 3])              # from Python list; copies data by default
torch.tensor([[1., 2.], [3., 4.]])   # infers dtype from values

torch.zeros(2, 3)                    # all zeros, float32 by default
torch.ones(2, 3)                     # all ones
torch.full((2, 3), 7)                # fill with constant
torch.empty(2, 3)                    # uninitialized memory (fast; values are garbage)

torch.arange(5)                      # [0, 1, 2, 3, 4] — stop exclusive
torch.arange(2, 8, 2)                # [2, 4, 6]
torch.linspace(0, 1, 5)              # 5 evenly spaced values, inclusive endpoints
torch.eye(3)                         # 3×3 identity matrix

torch.rand(2, 3)                     # uniform [0, 1)
torch.randn(2, 3)                    # standard normal
torch.randint(0, 10, (2, 3))         # random integers in [0, 10)

torch.zeros_like(x)                  # same shape/dtype/device as x
torch.ones_like(x)
```

### Create from NumPy

```python
import numpy as np

arr = np.array([1, 2, 3])

torch.from_numpy(arr)                # shares memory with NumPy (CPU only)
torch.as_tensor(arr)                 # shares memory when possible; prefers tensor dtype/device
torch.tensor(arr)                    # always copies data
```

### `tensor` vs `from_numpy` vs `as_tensor`

| Function | Copies data? | Shares NumPy memory? | Device | Typical use |
|---|---|---|---|---|
| `torch.tensor(data)` | Yes (default) | No | Any | Safe default when you want an independent tensor |
| `torch.from_numpy(ndarray)` | No | Yes (CPU) | CPU only | Zero-copy bridge from NumPy on CPU |
| `torch.as_tensor(data)` | Only if needed | Yes when possible | Any | Flexible; good when source may be list, array, or tensor |

**Performance:** `from_numpy` / `as_tensor` avoid copies — fastest for large CPU arrays.

**Error handling:** `from_numpy` raises if the array is not writable. Wrong dtype/device with shared memory can silently alias — mutate one, mutate both.

**Tip:** After `from_numpy`, changing the NumPy array changes the tensor. Use `torch.tensor(arr)` when you need a copy.

---

## 2. Inspect a tensor

```python
x.shape          # torch.Size — dimensions
x.size()         # same as shape
x.ndim           # number of dimensions
x.numel()        # total element count
x.dtype          # e.g. torch.float32, torch.int64
x.device         # e.g. cpu, cuda:0, mps
x.requires_grad  # True if tracked for autograd
x.is_cuda        # on GPU?
x.is_contiguous()  # memory layout OK for view?
```

---

## 3. Dtype and device

```python
x = torch.randn(3, 4, dtype=torch.float64)   # explicit dtype
x.float()                                    # cast to float32
x.double()                                   # cast to float64
x.int()                                      # cast to int32
x.long()                                     # cast to int64
x.bool()                                     # cast to bool

x.to(torch.float32)                          # cast dtype
x.to("cuda")                                 # move to GPU
x.to(device="cuda", dtype=torch.float16)     # both at once
x.cuda()                                     # shortcut to default CUDA device
x.cpu()                                      # move to CPU
x.half()                                     # float16
```

### `.cuda()` vs `.to(device)`

| Method | Flexibility | CPU / MPS | Notes |
|---|---|---|---|
| `x.cuda()` | CUDA only | No | Fails if CUDA unavailable |
| `x.cpu()` | CPU only | — | Always safe |
| `x.to(device)` | Any device string or `torch.device` | Yes | Preferred — portable code |
| `x.to("mps")` | Apple Silicon GPU | Yes | macOS only |

**Tip:** Define once: `device = torch.device("cuda" if torch.cuda.is_available() else "cpu")`.

---

## 4. Index, slice, reshape

```python
x = torch.randn(4, 5)

x[0]             # first row
x[:, 0]          # first column
x[1:3, 2:4]      # sub-block
x[x > 0]         # boolean mask (1D result)

x.view(2, 10)    # reshape; requires contiguous memory
x.reshape(2, 10) # reshape; copies if needed
x.flatten()      # 1D; always contiguous copy path
x.squeeze()      # remove size-1 dims
x.unsqueeze(0)   # add dim at axis 0 → shape (1, 4, 5)
x.unsqueeze(-1)  # add dim at end
x.T              # transpose (2D); for ndim>2 use permute
x.permute(2, 0, 1)  # reorder axes
x.contiguous()   # make memory contiguous (needed after transpose + view)
```

### `view` vs `reshape` vs `flatten`

| Function | Copies? | Requires contiguous? | Error if invalid shape |
|---|---|---|---|
| `view(*shape)` | No (view) | Yes | Raises `RuntimeError` |
| `reshape(*shape)` | Copies only when needed | No | Raises if total elements mismatch |
| `flatten(start, end)` | Usually copies to 1D | No | Rarely fails |

**Performance:** `view` is free (shared memory). `reshape` may copy after `transpose`/`permute`.

**Tip:** After `x = x.transpose(...)`, call `x.contiguous().view(...)` or just use `reshape`.

---

## 5. Combine and split tensors

```python
a = torch.tensor([1, 2, 3])
b = torch.tensor([4, 5, 6])

torch.cat([a, b], dim=0)       # [1,2,3,4,5,6] — join along existing dim
torch.stack([a, b], dim=0)     # shape (2, 3) — new dimension
torch.vstack([a, b])           # vertical stack (rows)
torch.hstack([a, b])           # horizontal stack

torch.split(x, 2, dim=1)       # equal chunks; raises if not divisible
torch.chunk(x, 3, dim=1)         # n chunks (last may be smaller)
```

### `cat` vs `stack`

| Function | Output ndim | Use when |
|---|---|---|
| `torch.cat(tensors, dim)` | Same as inputs | Join along existing axis (batch dim, seq dim) |
| `torch.stack(tensors, dim)` | ndim + 1 | Build a new batch/stack dimension |

---

## 6. Math and broadcasting

**Broadcasting** lets PyTorch apply an op to tensors of different shapes without copying data to match sizes. Smaller tensors are *virtually* expanded along missing or size-1 dimensions.

Compare shapes from the **right**: each dimension must be equal, or one of them must be `1`. If ndim differs, the shorter shape is padded with leading `1`s.

```python
x = torch.randn(3, 4)
x + 1                  # scalar broadcasts to every element
x + torch.tensor([1, 2, 3, 4])  # (4,) broadcasts across rows → still (3, 4)

# add a column vector to each row — new axis makes (3, 1) + (1, 4) → (3, 4)
col = torch.tensor([[0], [1], [2]])
x + col

a + b                # elementwise; shapes broadcast if compatible
a * b                # elementwise (not matrix multiply)
a @ b                # matrix multiply (2D+)
torch.matmul(a, b)   # batched matmul
torch.mm(a, b)       # strict 2D matmul
torch.bmm(a, b)      # batched 2D matmul, shapes (B,n,m) @ (B,m,p)

torch.sum(x)         # scalar sum
torch.mean(x, dim=0) # mean along axis 0
torch.std(x)           # standard deviation
torch.argmax(x, dim=1) # index of max per row

torch.exp(x)
torch.log(x)         # natural log; no log base 10 built-in
torch.sqrt(x)
torch.clamp(x, min=0, max=1)  # clip values
```

**Tip:** `*` is elementwise. Use `@` or `torch.matmul` for linear layers manually. Use `unsqueeze` / `[:, None]` when you need explicit control over which axis broadcasts.

---

## 7. Tensor ↔ Python / NumPy

```python
scalar = x.item()              # single-element tensor → Python number
lst = x.tolist()               # nested Python list (copies)
arr = x.detach().cpu().numpy() # tensor → NumPy (must be on CPU, detached)
```

### `item()` vs `tolist()` vs `numpy()`

| Method | Output | GPU OK? | Grad tracked? |
|---|---|---|---|
| `x.item()` | Python scalar | Yes (1 elem) | OK but usually `.detach().item()` in training |
| `x.tolist()` | Nested list | Yes | Includes grad history in graph if not detached |
| `x.numpy()` | `ndarray` | **No** — CPU only | Error if `requires_grad=True` |
| `x.detach().cpu().numpy()` | `ndarray` | Yes (moves first) | Safe for metrics/logging |

**Error handling:** Calling `.numpy()` on a CUDA tensor raises. Calling on a grad-tracked tensor raises unless detached.

---

## 8. Autograd basics

**Autograd** = PyTorch’s automatic differentiation engine. When tensors have `requires_grad=True`, every op builds a computation graph. From a scalar loss, it can compute ∂loss/∂param for every tracked parameter.

| Piece | What it does |
|---|---|
| **autograd** | Records ops on tracked tensors; knows how to differentiate them |
| **`loss.backward()`** | Walks the graph backward from `loss`; fills each leaf’s `.grad` with ∂loss/∂that_leaf |
| **`optimizer.zero_grad()`** | Clears stored `.grad` before the next step (grads **accumulate** by default) |

```python
x = torch.tensor([2.0, 3.0], requires_grad=True)
y = (x ** 2).sum()               # forward: build graph
y.backward()                     # reverse: fill x.grad with dy/dx
x.grad                           # e.g. [4., 6.] for y = x0² + x1²

# call every step *before* backward — otherwise grads stack up from prior batches
optimizer.zero_grad(set_to_none=True)  # set_to_none=True: free memory instead of writing zeros
```

**Tip:** Order in a training step: `zero_grad` → forward → `loss.backward()` → `optimizer.step()`.

### `backward()` vs `torch.autograd.grad`

| API | Stores `.grad`? | Multiple outputs | Typical use |
|---|---|---|---|
| `loss.backward()` | Yes, in leaf `.grad` | One scalar loss | Standard training loop |
| `torch.autograd.grad(outputs, inputs)` | No | Flexible | Custom grads, higher-order, meta-learning |

```python
grads = torch.autograd.grad(loss, model.parameters(), create_graph=False)
```

### `torch.no_grad()` vs `torch.inference_mode()`

| Context | Disables grad | Version counter | Speed | Use |
|---|---|---|---|---|
| `torch.no_grad()` | Yes | Still updated | Fast | Eval, inference, metric logging |
| `torch.inference_mode()` | Yes | Skipped | Faster | Pure inference (PyTorch 1.9+) |

**Tip:** Prefer `inference_mode()` for deployment/eval when you never need autograd inside the block.

---

## 9. `detach()` vs `clone()` vs `detach().clone()`

| Operation | Breaks grad graph? | Copies data? | Aliasing risk |
|---|---|---|---|
| `x.detach()` | Yes | No (view) | Mutations affect original |
| `x.clone()` | No | Yes | Independent copy, still in graph |
| `x.detach().clone()` | Yes | Yes | Safe independent snapshot |

**Use `detach().clone()`** when saving checkpoints mid-training or logging values without affecting backward.

---

## 10. Build a model (`nn.Module`)

```python
class MLP(nn.Module):
    def __init__(self, in_dim, hidden, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x):
        return self.net(x)

model = MLP(10, 64, 2).to(device)
```

### `nn.Module` vs `nn.Sequential`

| Style | When to use | Flexibility |
|---|---|---|
| Subclass `nn.Module` | Multi-input, branches, skip connections | Full control over `forward` |
| `nn.Sequential(...)` | Linear stack of layers | Simplest feed-forward stacks |

```python
# count trainable parameters
sum(p.numel() for p in model.parameters() if p.requires_grad)
```

### Common layers

```python
nn.Linear(in_features, out_features)     # fully connected: y = xW^T + b
nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)  # 2D convolution
nn.MaxPool2d(2)                          # 2×2 max pool, stride=kernel by default
nn.AvgPool2d(2)                          # average pool
nn.BatchNorm1d(num_features)             # normalize over batch (N) for (N,C) or (N,C,L)
nn.BatchNorm2d(num_features)             # normalize over N×H×W per channel
nn.LayerNorm(normalized_shape)           # normalize over last dim(s); batch-size independent
nn.GroupNorm(num_groups, num_channels)   # normalize within channel groups
nn.Dropout(p=0.5)                        # zero random activations during training
nn.Embedding(num_embeddings, embedding_dim)  # lookup table for token IDs
nn.LSTM(input_size, hidden_size, batch_first=True)
nn.GRU(input_size, hidden_size, batch_first=True)
nn.TransformerEncoderLayer(d_model, nhead, batch_first=True)
```

### Normalization layers compared

| Layer | Normalizes over | Batch size sensitive? | Typical use |
|---|---|---|---|
| `BatchNorm1d/2d` | Batch + spatial dims | Yes (bad for tiny batch) | CNNs, large batch training |
| `LayerNorm` | Feature dims per sample | No | Transformers, RNNs, small batch |
| `GroupNorm` | Channel groups per sample | No | Small batch CNNs, detection |

**Tip:** Call `model.train()` during training and `model.eval()` before validation — toggles Dropout and BatchNorm behavior.

---

## 11. Loss functions

```python
criterion = nn.CrossEntropyLoss()          # multiclass: raw logits in, class indices target
criterion = nn.NLLLoss()                   # needs log-prob input (often after LogSoftmax)
criterion = nn.BCEWithLogitsLoss()         # binary/multilabel: raw logits
criterion = nn.BCELoss()                   # needs probabilities in [0,1]
criterion = nn.MSELoss()                   # regression
criterion = nn.L1Loss()                    # MAE regression
criterion = nn.SmoothL1Loss()              # Huber-like; robust regression
```

### Classification losses compared

| Loss | Input to loss | Target shape | Softmax/sigmoid inside? | Common mistake |
|---|---|---|---|---|
| `CrossEntropyLoss` | Raw logits `(N,C)` | Class index `(N,)` long | Yes (log-softmax + NLL) | Applying softmax before loss |
| `NLLLoss` | Log-probabilities | Class index `(N,)` | No — pair with `LogSoftmax` | Feeding raw logits |
| `BCEWithLogitsLoss` | Raw logits | Float 0/1 same shape | Sigmoid inside | Using with already-sigmoided outputs |
| `BCELoss` | Probabilities [0,1] | Float 0/1 | No | Numerical instability without logits version |

**Performance:** `BCEWithLogitsLoss` and `CrossEntropyLoss` fuse activation + loss — faster and more stable than separate steps.

```python
# functional equivalents (same math, no module state)
loss = F.cross_entropy(logits, targets)
loss = F.binary_cross_entropy_with_logits(logits, targets.float())
```

### `F.cross_entropy` vs `nn.CrossEntropyLoss`

| | Module | Functional |
|---|---|---|
| State | Can hold `weight`, `ignore_index`, `label_smoothing` | Pass args each call |
| Use | Store as `self.criterion` in a class | One-off or custom loops |

---

## 12. Optimizers

```python
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)  # decoupled WD
```

### Optimizer comparison

| Optimizer | Memory | Convergence | Weight decay | Typical use |
|---|---|---|---|---|
| `SGD` + momentum | Low | Needs tuning | L2 in grad | CNNs at scale, fine-tuning |
| `Adam` | Higher (2 momentum buffers) | Fast default | Often conflated with L2 | General default, transformers |
| `AdamW` | Same as Adam | Stable with WD | Decoupled (correct) | Transformers, modern default |

**Tip:** `weight_decay` in Adam is not the same as AdamW — prefer `AdamW` when you want regularization.

### Learning rate schedulers

```python
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3)

# after each epoch:
scheduler.step()                    # StepLR, CosineAnnealingLR
scheduler.step(val_loss)            # ReduceLROnPlateau — pass metric
```

| Scheduler | Needs metric? | Behavior |
|---|---|---|
| `StepLR` | No | Drop LR every N epochs |
| `CosineAnnealingLR` | No | Smooth cosine decay to 0 |
| `ReduceLROnPlateau` | Yes (val loss) | Reduce when metric stalls |

---

## 13. Standard training loop

```python
model.train()
for epoch in range(num_epochs):
    for batch_x, batch_y in train_loader:
        batch_x = batch_x.to(device, non_blocking=True)
        batch_y = batch_y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)  # clear old gradients

        logits = model(batch_x)
        loss = criterion(logits, batch_y)

        loss.backward()                        # backprop
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # optional
        optimizer.step()
```

### Validation loop

```python
model.eval()
correct, total = 0, 0

with torch.inference_mode():
    for batch_x, batch_y in val_loader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

        logits = model(batch_x)
        preds = logits.argmax(dim=1)

        correct += (preds == batch_y).sum().item()
        total += batch_y.size(0)

acc = correct / total
```

### `model.train()` vs `model.eval()`

| Mode | Dropout | BatchNorm | `requires_grad` |
|---|---|---|---|
| `train()` | Active | Uses batch stats | Unchanged (still True for params) |
| `eval()` | Disabled | Uses running stats | Unchanged — pair with `no_grad` / `inference_mode` |

---

## 14. Dataset and DataLoader

```python
class TabularDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.as_tensor(X, dtype=torch.float32)
        self.y = torch.as_tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_loader = DataLoader(
    train_ds,
    batch_size=64,
    shuffle=True,           # shuffle training data each epoch
    num_workers=4,          # parallel loading (0 = main process only)
    pin_memory=True,        # faster CPU→GPU copy when using CUDA
    drop_last=False,        # drop incomplete last batch
)
```

### DataLoader options

| Argument | Effect | Performance tip |
|---|---|---|
| `batch_size` | Samples per step | Larger = better GPU use until OOM |
| `shuffle=True` | Random order each epoch | Training only |
| `num_workers>0` | Subprocess prefetch | 2–8 often good; too many hurts on macOS/Windows |
| `pin_memory=True` | Page-locked host memory | Use with CUDA + `non_blocking=True` |
| `persistent_workers=True` | Keep workers alive between epochs | Saves spawn cost when `num_workers>0` |

**Error handling:** `num_workers>0` on notebooks/Windows can cause pickling errors — set `num_workers=0` to debug.

---

## 15. Save and load

```python
# recommended: state dict only (portable, smaller)
torch.save(model.state_dict(), "model_weights.pt")
model.load_state_dict(torch.load("model_weights.pt", map_location=device))

# full checkpoint (weights + optimizer + epoch)
checkpoint = {
    "epoch": epoch,
    "model_state": model.state_dict(),
    "optimizer_state": optimizer.state_dict(),
}
torch.save(checkpoint, "checkpoint.pt")

ckpt = torch.load("checkpoint.pt", map_location=device)
model.load_state_dict(ckpt["model_state"])
optimizer.load_state_dict(ckpt["optimizer_state"])
```

### Save strategies compared

| Method | Saves | Load requirement | Risk |
|---|---|---|---|
| `state_dict()` | Weights only | Same model architecture code | Low — preferred |
| Full `model` object | Architecture + weights | Same class definition + version | Breaks across refactors |
| `torch.jit.script` / `trace` | TorchScript graph | Different API for inference | Deployment |

**Tip:** Always use `map_location=device` when loading on a machine without GPU.

```python
# strict=False allows partial load (e.g. transfer learning)
model.load_state_dict(torch.load("weights.pt"), strict=False)
```

---

## 16. GPU, mixed precision, compile

```python
# automatic mixed precision (AMP) — faster on modern NVIDIA GPUs
scaler = torch.cuda.amp.GradScaler()

model.train()
for batch_x, batch_y in train_loader:
    batch_x, batch_y = batch_x.to(device), batch_y.to(device)
    optimizer.zero_grad(set_to_none=True)

    with torch.cuda.amp.autocast():          # forward in float16 where safe
        loss = criterion(model(batch_x), batch_y)

    scaler.scale(loss).backward()            # scaled backward
    scaler.step(optimizer)
    scaler.update()
```

### `DataParallel` vs `DistributedDataParallel`

| API | Processes | Performance | Use |
|---|---|---|---|
| `nn.DataParallel` | Single process, multiple GPUs | GIL bottleneck, slower | Legacy / quick test |
| `DistributedDataParallel` | One process per GPU | Best multi-GPU training | Production multi-GPU |

**Tip:** On Apple Silicon use `device = "mps"` when available; AMP API differs — check current PyTorch docs for `mps`.

```python
model = torch.compile(model)   # PyTorch 2+ graph optimization; first epoch slower
```

---

## 17. `nn.functional` vs `nn.Module` layers

Many ops exist in both forms:

```python
# Module: has learnable params / state, register in __init__
self.conv = nn.Conv2d(3, 16, 3)

# Functional: stateless call, good for one-off ops
out = F.relu(x)
out = F.max_pool2d(x, kernel_size=2)
out = F.interpolate(x, scale_factor=2, mode="bilinear")
```

| Pattern | Example | When |
|---|---|---|
| `nn.*` module | `nn.ReLU()`, `nn.Conv2d` | Layers with weights or state in `Sequential` |
| `F.*` functional | `F.relu(x)`, `F.cross_entropy` | No params; flexible control flow in `forward` |

---

## 18. Useful debugging checks

```python
torch.isnan(x).any()              # NaN check
torch.isinf(x).any()              # Inf check
torch.allclose(a, b, rtol=1e-5)   # approximate equality

# reproduce experiments
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

# see GPU memory
torch.cuda.memory_summary()

# anomaly detection (slow — debug only)
with torch.autograd.detect_anomaly():
    loss.backward()
```

**Tip:** Exploding loss → lower LR, check normalization, try `clip_grad_norm_`, inspect for NaN inputs.

---

## 19. Minimal end-to-end example

```python
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# toy data: 100 samples, 10 features, binary labels
X = torch.randn(100, 10)
y = (X[:, 0] + X[:, 1] > 0).long()

ds = TensorDataset(X, y)
loader = DataLoader(ds, batch_size=16, shuffle=True)

model = nn.Sequential(
    nn.Linear(10, 32),
    nn.ReLU(),
    nn.Linear(32, 2),
).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

for epoch in range(5):
    model.train()
    for batch_x, batch_y in loader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)

        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(batch_x), batch_y)
        loss.backward()
        optimizer.step()

    print(f"epoch {epoch+1} loss {loss.item():.4f}")
```

`TensorDataset` wraps tensors that are already in memory — quick for prototypes.

---

## 20. What to memorize vs look up

### Memorize concepts

```text
tensor shape / dtype / device
requires_grad and the train/eval + no_grad pattern
loss input format (logits vs probabilities)
optimizer.zero_grad before backward
train/val split before fitting normalization stats
state_dict for saving weights
batch dimension convention (N, C, H, W for images)
```

### Look up syntax

```text
exact Conv2d padding/stride formulas
scheduler constructor args
DistributedDataParallel setup
custom Dataset __getitem__ details
torch.compile and AMP API changes by version
```

## Official docs

- PyTorch: https://pytorch.org/docs/stable/
- torchvision: https://pytorch.org/vision/stable/
- torchaudio: https://pytorch.org/audio/stable/
