# pandas Cheat Sheet

```python
import pandas as pd
import numpy as np
```

pandas is mainly about **labeled tabular data** using `DataFrame` and `Series`.

---

## 1. Create data

```python
df = pd.DataFrame({
    "name": ["Aki", "Ben", "Cara"],
    "age": [29, 35, 41],
    "sales": [120.5, 98.0, 150.2],
})
```

Series:

```python
s = pd.Series([10, 20, 30], name="score")
```

---

## 2. Read data

CSV:

```python
df = pd.read_csv("data.csv")
```

Useful CSV options:

```python
df = pd.read_csv(
    "data.csv",
    usecols=["id", "date", "sales"],
    parse_dates=["date"],
    dtype={"id": "int64"},
)
```

JSON:

```python
df = pd.read_json("data.json")
```

Parquet:

```python
df = pd.read_parquet("data.parquet")
```

Excel:

```python
df = pd.read_excel("data.xlsx", sheet_name="Sheet1")
```

---

## 3. Quick inspection

```python
df.head()
df.tail()

df.shape
df.columns
df.index
df.dtypes

df.info()
df.describe()
df.describe(include="all")
```

Random rows:

```python
df.sample(5, random_state=42)
```

---

## 4. Select columns

One column -> Series:

```python
df["sales"]
```

Multiple columns -> DataFrame:

```python
df[["name", "sales"]]
```

---

## 5. Select rows with `loc`

Label-based:

```python
df.loc[0]
df.loc[0:3]
df.loc[:, ["name", "sales"]]
```

Filter rows + choose columns:

```python
df.loc[df["age"] >= 30, ["name", "sales"]]
```

---

## 6. Select by integer position with `iloc`

```python
df.iloc[0]
df.iloc[:5]
df.iloc[:, 0]
df.iloc[:5, :3]
```

---

## 7. Filter rows

```python
df[df["age"] > 30]
```

Multiple conditions:

```python
df[(df["age"] > 30) & (df["sales"] > 100)]
```

OR:

```python
df[(df["age"] < 25) | (df["age"] > 60)]
```

NOT:

```python
df[~(df["age"] > 30)]
```

Use parentheses around each condition.

---

## 8. Membership filtering

```python
df[df["state"].isin(["OK", "TX", "KS"])]
```

Not in:

```python
df[~df["state"].isin(["OK", "TX"])]
```

---

## 9. Range filtering

```python
df[df["age"].between(30, 50)]
```

---

## 10. Query syntax

```python
df.query("age > 30 and sales > 100")
```

With Python variable:

```python
minimum = 100
df.query("sales > @minimum")
```

---

## 11. Sort

```python
df.sort_values("sales")
df.sort_values("sales", ascending=False)

df.sort_values(
    ["state", "sales"],
    ascending=[True, False],
)
```

Sort index:

```python
df.sort_index()
```

---

## 12. Rename

```python
df = df.rename(columns={
    "old_name": "new_name",
})
```

Clean all column names:

```python
df.columns = (
    df.columns
      .str.strip()
      .str.lower()
      .str.replace(" ", "_")
)
```

---

## 13. Add / modify columns

```python
df["revenue"] = df["price"] * df["quantity"]
df["age_next_year"] = df["age"] + 1
```

Conditional:

```python
df["high_value"] = df["sales"] >= 100
```

With NumPy:

```python
df["tier"] = np.where(df["sales"] >= 100, "high", "low")
```

---

## 14. Drop columns / rows

Columns:

```python
df = df.drop(columns=["unused_column"])
```

Rows:

```python
df = df.drop(index=[0, 2])
```

---

## 15. Missing values

Count:

```python
df.isna().sum()
```

Percentage:

```python
df.isna().mean().sort_values(ascending=False)
```

Rows containing any missing value:

```python
df[df.isna().any(axis=1)]
```

Drop:

```python
df.dropna()
df.dropna(subset=["target"])
```

Fill:

```python
df["age"] = df["age"].fillna(df["age"].median())
df["city"] = df["city"].fillna("Unknown")
```

Forward/backward fill:

```python
df.ffill()
df.bfill()
```

---

## 16. Replace values

```python
df["status"] = df["status"].replace({
    "Y": "Yes",
    "N": "No",
})
```

---

## 17. Change types

```python
df["age"] = df["age"].astype("int64")
df["price"] = df["price"].astype("float64")
df["category"] = df["category"].astype("category")
```

Safer numeric conversion:

```python
df["price"] = pd.to_numeric(df["price"], errors="coerce")
```

---

## 18. String operations

```python
df["name"].str.lower()
df["name"].str.upper()
df["name"].str.strip()

df["email"].str.contains("@gmail.com", na=False)
df["name"].str.startswith("A", na=False)

df["full_name"].str.split(" ", expand=True)
```

Regex replacement:

```python
df["phone"] = df["phone"].str.replace(r"\D", "", regex=True)
```

---

## 19. Dates and times

Convert:

```python
df["date"] = pd.to_datetime(df["date"], errors="coerce")
```

Extract components:

```python
df["year"] = df["date"].dt.year
df["month"] = df["date"].dt.month
df["day"] = df["date"].dt.day
df["weekday"] = df["date"].dt.day_name()
```

Filter:

```python
df[df["date"] >= "2026-01-01"]
```

Time difference:

```python
df["days_since"] = (
    pd.Timestamp("2026-08-26") - df["date"]
).dt.days
```

---

## 20. Duplicate data

Find:

```python
df.duplicated()
df[df.duplicated()]
```

Based on selected columns:

```python
df[df.duplicated(subset=["customer_id"], keep=False)]
```

Remove:

```python
df = df.drop_duplicates()
df = df.drop_duplicates(subset=["customer_id"], keep="last")
```

---

## 21. Value counts

```python
df["category"].value_counts()
df["category"].value_counts(normalize=True)
```

Include missing:

```python
df["category"].value_counts(dropna=False)
```

---

## 22. Unique values

```python
df["category"].unique()
df["category"].nunique()
```

---

## 23. Basic statistics

```python
df["sales"].mean()
df["sales"].median()
df["sales"].std()
df["sales"].min()
df["sales"].max()
df["sales"].quantile([0.25, 0.5, 0.75])
```

Correlation:

```python
df.corr(numeric_only=True)
```

---

## 24. GroupBy

Basic:

```python
df.groupby("category")["sales"].mean()
```

Multiple statistics:

```python
df.groupby("category")["sales"].agg([
    "count",
    "mean",
    "median",
    "sum",
])
```

Named aggregations:

```python
summary = (
    df.groupby("category", as_index=False)
      .agg(
          orders=("order_id", "count"),
          avg_sales=("sales", "mean"),
          total_sales=("sales", "sum"),
      )
)
```

Multiple groups:

```python
df.groupby(["state", "category"])["sales"].sum()
```

---

## 25. `transform` vs `agg`

`agg()` reduces each group:

```python
df.groupby("category")["sales"].mean()
```

`transform()` returns one value per original row:

```python
df["category_avg"] = (
    df.groupby("category")["sales"].transform("mean")
)
```

Useful for comparing each row with its group:

```python
df["above_category_avg"] = (
    df["sales"] > df["category_avg"]
)
```

---

## 26. Merge / join

SQL-style merge:

```python
result = pd.merge(
    customers,
    orders,
    on="customer_id",
    how="left",
)
```

Common `how` values:

```text
inner
left
right
outer
```

Different key names:

```python
pd.merge(
    left,
    right,
    left_on="customer_id",
    right_on="id",
    how="left",
)
```

Useful validation:

```python
pd.merge(
    customers,
    orders,
    on="customer_id",
    how="left",
    validate="one_to_many",
)
```

---

## 27. Concatenate

Rows:

```python
df = pd.concat([df1, df2], ignore_index=True)
```

Columns:

```python
df = pd.concat([df1, df2], axis=1)
```

---

## 28. Pivot table

```python
pd.pivot_table(
    df,
    index="state",
    columns="category",
    values="sales",
    aggfunc="mean",
)
```

---

## 29. Pivot

When combinations are unique:

```python
df.pivot(
    index="date",
    columns="product",
    values="sales",
)
```

---

## 30. Melt: wide -> long

```python
long_df = df.melt(
    id_vars=["customer_id"],
    value_vars=["jan", "feb", "mar"],
    var_name="month",
    value_name="sales",
)
```

---

## 31. Crosstab

```python
pd.crosstab(df["segment"], df["converted"])
```

Proportions:

```python
pd.crosstab(
    df["segment"],
    df["converted"],
    normalize="index",
)
```

---

## 32. Apply / map

Map dictionary:

```python
df["state_name"] = df["state"].map({
    "OK": "Oklahoma",
    "TX": "Texas",
})
```

Apply a function to a Series:

```python
df["name_length"] = df["name"].apply(len)
```

Prefer built-in vectorized operations over `apply()` when possible.

Instead of:

```python
df["double_sales"] = df["sales"].apply(lambda x: x * 2)
```

Prefer:

```python
df["double_sales"] = df["sales"] * 2
```

---

## 33. Index

Set:

```python
df = df.set_index("customer_id")
```

Reset:

```python
df = df.reset_index()
```

---

## 34. Sample

```python
df.sample(n=100, random_state=42)
df.sample(frac=0.1, random_state=42)
```

---

## 35. Rank

```python
df["sales_rank"] = df["sales"].rank(
    ascending=False,
    method="dense",
)
```

---

## 36. Rolling windows

```python
df["rolling_7"] = df["sales"].rolling(7).mean()
```

Group-specific rolling calculations often require sorting by time first.

---

## 37. Shift / lag

```python
df["previous_sales"] = df["sales"].shift(1)
df["change"] = df["sales"] - df["previous_sales"]
df["pct_change"] = df["sales"].pct_change()
```

---

## 38. Export

CSV:

```python
df.to_csv("output.csv", index=False)
```

Parquet:

```python
df.to_parquet("output.parquet", index=False)
```

Excel:

```python
df.to_excel("output.xlsx", index=False)
```

JSON:

```python
df.to_json("output.json", orient="records")
```

---

## 39. Large data: memory basics

Inspect memory:

```python
df.info(memory_usage="deep")
```

Load only required columns:

```python
df = pd.read_csv(
    "large.csv",
    usecols=["id", "date", "target"],
)
```

Specify efficient dtypes:

```python
df = pd.read_csv(
    "large.csv",
    dtype={
        "id": "int32",
        "category": "category",
    },
)
```

Process CSV in chunks:

```python
for chunk in pd.read_csv("large.csv", chunksize=100_000):
    process(chunk)
```

Prefer Parquet for analytical datasets when possible.

---

## 40. Quick EDA checklist

```python
df.shape
df.head()
df.info()

df.isna().sum()
df.duplicated().sum()

df.describe()
df.describe(include="object")

df["target"].value_counts(dropna=False)
df["target"].value_counts(normalize=True)

df.corr(numeric_only=True)
```

---

## 41. ML-friendly split into X / y

```python
target = "churn"

X = df.drop(columns=[target])
y = df[target]
```

---

## 42. High-value things to remember exist

```text
read_csv / read_parquet
head / info / describe
loc / iloc
query / isin / between
isna / fillna / dropna
astype / to_numeric / to_datetime
sort_values
value_counts / unique / nunique
groupby / agg / transform
merge / concat
pivot_table / pivot / melt
drop_duplicates
str.*
dt.*
rolling / shift / pct_change
sample
to_csv / to_parquet
```

## Official docs

- https://pandas.pydata.org/docs/
