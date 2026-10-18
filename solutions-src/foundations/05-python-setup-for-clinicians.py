# %% Step 1
# The version-check cell is in the worked example above. Any recent versions work; if an import fails
# in a local environment, install the missing package with pip and restart the kernel.
import sys
print(f"Running Python {sys.version.split()[0]}")

# %% Step 2
# `df` was loaded from the synthetic CSV in the worked example. Reload it, so this step does not depend
# on the columns the worked example added.
df = pd.read_csv(StringIO(csv_text))
n_rows, n_cols = df.shape
print(f"{n_rows} rows, {n_cols} columns")
print("Missing values per column:")
print(df.isna().sum()[df.isna().sum() > 0])

# %% Step 3
df["tested"] = df.hba1c.notna()
# pandas' nullable boolean type can hold True, False and <NA>, so "not measured" stays distinct from "no".
df["hba1c_above_7"] = (df.hba1c > 7.0).astype("boolean").mask(df.hba1c.isna())
print(df[["patient_id", "hba1c", "tested", "hba1c_above_7"]])
print("\nCounts, including missing:")
print(df.hba1c_above_7.value_counts(dropna=False))

# %% Step 4
summary = df.groupby("sex").agg(
    hba1c_mean=("hba1c", "mean"),
    hba1c_n=("hba1c", "count"),           # count() skips missing values
    sbp_mean=("systolic_bp", "mean"),
    sbp_n=("systolic_bp", "count"),
    patients=("patient_id", "size"),      # size() counts every row
)
print(summary.round(1))

# %% Note
# The men's mean HbA1c is based on two values, not three, because one man was not tested. Reporting the
# count next to each mean makes that visible; with numbers this small, neither mean says much.

# %% Step 5
import matplotlib.pyplot as plt

shown = df.dropna(subset=["hba1c", "age"])
fig, ax = plt.subplots(figsize=(5, 3.5))
for sex, group in shown.groupby("sex"):
    ax.scatter(group.age, group.hba1c, label=sex)
ax.axhline(7.0, linestyle="--", linewidth=1, color="grey")
ax.set_xlabel("Age (years)")
ax.set_ylabel("HbA1c (%)")
ax.legend(title="Sex")
caption = (f"Synthetic data. {len(shown)} of {len(df)} patients shown; "
           f"{len(df) - len(shown)} excluded because HbA1c was not measured.")
fig.text(0.01, 0.01, caption, fontsize=8)
fig.tight_layout(rect=(0, 0.06, 1, 1))
plt.show()
print(caption)
