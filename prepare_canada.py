from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

CARAVAN = Path(r"C:\Users\nyk750\Caravan-nc\Caravan-nc")
REPO = Path(r"C:\Users\nyk750\flood-forecasting")

# --------------------------------------------------
# 1. Find HYSETS metadata
# --------------------------------------------------
matches = list(CARAVAN.rglob("attributes_other_hysets.csv"))

if not matches:
    raise FileNotFoundError(
        f"Could not find attributes_other_hysets.csv under {CARAVAN}"
    )

meta_file = matches[0]
print("HYSETS metadata:", meta_file)

df = pd.read_csv(meta_file)

print("\nColumns:")
print(df.columns.tolist())

# Caravan standard attribute files use gauge_id as first column.
id_col = "gauge_id" if "gauge_id" in df.columns else df.columns[0]

# Find country column robustly.
country_candidates = [
    c for c in df.columns
    if "country" in c.lower()
]

if not country_candidates:
    raise ValueError("Could not find a country column.")

country_col = country_candidates[0]

print("\nID column:", id_col)
print("Country column:", country_col)

# --------------------------------------------------
# 2. Filter Canada
# --------------------------------------------------
canada = df[
    df[country_col]
    .astype(str)
    .str.strip()
    .str.lower()
    .eq("canada")
].copy()

canada_ids = canada[id_col].astype(str).tolist()

print("\nCanadian HYSETS basins in Caravan:", len(canada_ids))

# --------------------------------------------------
# 3. Only keep basins available in Google's MultiMet list
# --------------------------------------------------
multimet_file = (
    REPO
    / "example-configs"
    / "multimet-basins-list-without-chirps.txt"
)

multimet_ids = {
    x.strip()
    for x in multimet_file.read_text().splitlines()
    if x.strip()
}

usable = sorted(set(canada_ids) & multimet_ids)

print("Canadian basins available for MultiMet:", len(usable))

if len(usable) < 20:
    raise RuntimeError(
        f"Only {len(usable)} usable Canadian basins found. Check IDs/data."
    )

# --------------------------------------------------
# 4. Random spatial split
# --------------------------------------------------
rng = np.random.default_rng(42)

ids = np.array(usable)
rng.shuffle(ids)

n = len(ids)

n_train = int(n * 0.80)
n_val = int(n * 0.10)

train_ids = ids[:n_train]
val_ids = ids[n_train:n_train + n_val]
test_ids = ids[n_train + n_val:]

# --------------------------------------------------
# 5. Save
# --------------------------------------------------
out = REPO / "canada_experiment" / "basins"
out.mkdir(parents=True, exist_ok=True)

np.savetxt(out / "all_canada.txt", ids, fmt="%s")
np.savetxt(out / "train.txt", train_ids, fmt="%s")
np.savetxt(out / "validation.txt", val_ids, fmt="%s")
np.savetxt(out / "test.txt", test_ids, fmt="%s")

# Keep metadata only for basins that made the MultiMet filter + split.
canada_usable = canada[canada[id_col].astype(str).isin(usable)].copy()
canada_usable.to_csv(out / "canada_metadata.csv", index=False)

print("\nSPLIT")
print("Train:", len(train_ids))
print("Validation:", len(val_ids))
print("Test:", len(test_ids))

print("\nExample TEST basin:")
print(test_ids[0])

# --------------------------------------------------
# 6. Map of train / validation / test gauges
# --------------------------------------------------
split_lookup = {
    gid: "train" for gid in train_ids.tolist()
} | {
    gid: "validation" for gid in val_ids.tolist()
} | {
    gid: "test" for gid in test_ids.tolist()
}
canada_usable["split"] = canada_usable[id_col].astype(str).map(split_lookup)

colors = {
    "train": "#1f77b4",
    "validation": "#ff7f0e",
    "test": "#2ca02c",
}
fig, ax = plt.subplots(figsize=(10, 8))
for name in ["train", "validation", "test"]:
    d = canada_usable[canada_usable["split"] == name]
    ax.scatter(
        d["gauge_lon"],
        d["gauge_lat"],
        s=12,
        alpha=0.75,
        c=colors[name],
        label=f"{name} (n={len(d)})",
    )
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")
ax.set_title("Canada MultiMet basins: train / validation / test")
ax.legend(loc="lower left")
ax.grid(True, alpha=0.3)
ax.set_aspect("equal", adjustable="datalim")
fig.tight_layout()
map_path = out / "canada_splits_map.png"
fig.savefig(map_path, dpi=160)
plt.close(fig)
print("\nSaved map:", map_path)