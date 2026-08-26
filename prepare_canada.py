from pathlib import Path
import json
import urllib.request

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection

CARAVAN = Path(r"C:\Users\nyk750\Caravan-nc\Caravan-nc")
REPO = Path(r"C:\Users\nyk750\flood-forecasting")

# 1. Find HYSETS metadata
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

# 2. Filter Canada
canada = df[
    df[country_col]
    .astype(str)
    .str.strip()
    .str.lower()
    .eq("canada")
].copy()

canada_ids = canada[id_col].astype(str).tolist()

print("\nCanadian HYSETS basins in Caravan:", len(canada_ids))

# 3. Only keep basins available in Google's MultiMet list
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

# 4. Random spatial split
rng = np.random.default_rng(42)

ids = np.array(usable)
rng.shuffle(ids)

n = len(ids)

n_train = int(n * 0.80)
n_val = int(n * 0.10)

train_ids = ids[:n_train]
val_ids = ids[n_train:n_train + n_val]
test_ids = ids[n_train + n_val:]

# 5. Save
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

# 6. Map of train / validation / test gauges
split_lookup = {
    gid: "train" for gid in train_ids.tolist()
} | {
    gid: "validation" for gid in val_ids.tolist()
} | {
    gid: "test" for gid in test_ids.tolist()
}
canada_usable["split"] = canada_usable[id_col].astype(str).map(split_lookup)

# Canada / province outlines (cached locally after first download).
border_path = out / "canada_provinces.geojson"
border_url = (
    "https://raw.githubusercontent.com/codeforamerica/"
    "click_that_hood/master/public/data/canada.geojson"
)
if not border_path.exists():
    print("\nDownloading Canada province borders...")
    with urllib.request.urlopen(border_url, timeout=60) as resp:
        border_path.write_bytes(resp.read())


def _geojson_line_segments(geojson: dict) -> list[np.ndarray]:
    """Extract lon/lat rings from a GeoJSON FeatureCollection as line segments."""
    segments = []

    def add_ring(ring):
        coords = np.asarray(ring, dtype=float)
        if coords.ndim == 2 and coords.shape[0] >= 2:
            segments.append(coords[:, :2])

    def add_geometry(geom):
        if geom is None:
            return
        gtype = geom.get("type")
        coords = geom.get("coordinates")
        if gtype == "Polygon":
            for ring in coords:
                add_ring(ring)
        elif gtype == "MultiPolygon":
            for polygon in coords:
                for ring in polygon:
                    add_ring(ring)
        elif gtype == "GeometryCollection":
            for child in geom.get("geometries", []):
                add_geometry(child)

    for feature in geojson.get("features", []):
        add_geometry(feature.get("geometry"))
    return segments


with border_path.open(encoding="utf-8") as f:
    border_geojson = json.load(f)
border_segments = _geojson_line_segments(border_geojson)

colors = {
    "train": "#1f77b4",
    "validation": "#ff7f0e",
    "test": "#2ca02c",
}
fig, ax = plt.subplots(figsize=(10, 8))
if border_segments:
    ax.add_collection(
        LineCollection(
            border_segments,
            colors="#444444",
            linewidths=0.6,
            alpha=0.85,
            zorder=0,
        )
    )
for name in ["train", "validation", "test"]:
    d = canada_usable[canada_usable["split"] == name]
    ax.scatter(
        d["gauge_lon"],
        d["gauge_lat"],
        s=12,
        alpha=0.75,
        c=colors[name],
        label=f"{name} (n={len(d)})",
        zorder=2,
    )
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")
ax.set_title("Canada MultiMet basins: train / validation / test")
ax.legend(loc="lower left")
ax.grid(True, alpha=0.3)
# Fit view to gauges, with a little padding so borders stay visible.
pad_lon = 3.0
pad_lat = 2.0
ax.set_xlim(
    canada_usable["gauge_lon"].min() - pad_lon,
    canada_usable["gauge_lon"].max() + pad_lon,
)
ax.set_ylim(
    canada_usable["gauge_lat"].min() - pad_lat,
    canada_usable["gauge_lat"].max() + pad_lat,
)
ax.set_aspect("equal", adjustable="box")
fig.tight_layout()
map_path = out / "canada_splits_map.png"
fig.savefig(map_path, dpi=160)
plt.close(fig)
print("\nSaved map:", map_path)