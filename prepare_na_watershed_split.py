"""
Create US + Canada train/val/test splits by HydroBASINS lev06 HYBAS_ID.

Each gauge is assigned to its lev06 subbasin polygon (HYBAS_ID). Whole
HYBAS_ID groups go to only one of train / validation / test so gauges in
the same watershed block are not split across folds.

Includes HYSETS MultiMet basins in Canada and the United States
(~12k gauges). Does not add camels_* IDs (those largely duplicate USGS
gauges already present as hysets_*).

Outputs:
  canada_experiment/basins/train_hybas06.txt
  canada_experiment/basins/validation_hybas06.txt
  canada_experiment/basins/test_hybas06.txt
  canada_experiment/basins/na_metadata_hybas06.csv
  canada_experiment/basins/na_splits_hybas06_map.png
"""

from __future__ import annotations

import json
import urllib.request
import zipfile
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection

CARAVAN = Path("/u2/users/nyk750/Caravan-nc")
REPO = Path("/u2/users/nyk750/flood-forecasting")
OUT = REPO / "canada_experiment" / "basins"
CACHE = OUT / "hydrobasins_cache"

HYBAS_URLS = {
    "na": (
        "https://data.hydrosheds.org/file/hydrobasins/standard/"
        "hybas_na_lev06_v1c.zip"
    ),
    "ar": (
        "https://data.hydrosheds.org/file/hydrobasins/standard/"
        "hybas_ar_lev06_v1c.zip"
    ),
}
# North America Albers (meters) for nearest-polygon fallback.
NA_ALBERS = (
    "+proj=aea +lat_0=40 +lon_0=-96 +lat_1=20 +lat_2=60 "
    "+x_0=0 +y_0=0 +datum=NAD83 +units=m +no_defs"
)
SEED = 42
TRAIN_FRAC = 0.80
VAL_FRAC = 0.10
COUNTRIES = {"canada", "united states of america"}
GROUP_COL = "HYBAS_ID"


def _load_us_canada_multimet_meta() -> pd.DataFrame:
    meta_file = CARAVAN / "attributes" / "hysets" / "attributes_other_hysets.csv"
    df = pd.read_csv(meta_file)
    country_col = next(c for c in df.columns if "country" in c.lower())

    multimet_ids = {
        x.strip()
        for x in (
            REPO / "example-configs" / "multimet-basins-list-without-chirps.txt"
        )
        .read_text()
        .splitlines()
        if x.strip()
    }

    country = df[country_col].astype(str).str.strip().str.lower()
    keep = df[country.isin(COUNTRIES) & df["gauge_id"].astype(str).isin(multimet_ids)].copy()
    keep["gauge_id"] = keep["gauge_id"].astype(str)
    keep["country_norm"] = country.loc[keep.index].values
    return keep.reset_index(drop=True)


def _download_zip(url: str, zip_path: Path) -> None:
    print(f"Downloading {url} -> {zip_path}")
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "Mozilla/5.0 (compatible; flood-forecasting)"},
    )
    with urllib.request.urlopen(req, timeout=300) as resp, open(zip_path, "wb") as out:
        out.write(resp.read())


def _ensure_hydrobasins() -> gpd.GeoDataFrame:
    CACHE.mkdir(parents=True, exist_ok=True)
    frames = []
    for region, url in HYBAS_URLS.items():
        zip_path = CACHE / f"hybas_{region}_lev06_v1c.zip"
        if not zip_path.exists():
            _download_zip(url, zip_path)
        shp_matches = list(CACHE.rglob(f"hybas_{region}_lev06*.shp"))
        if not shp_matches:
            print(f"Extracting {zip_path}")
            with zipfile.ZipFile(zip_path, "r") as zf:
                zf.extractall(CACHE)
            shp_matches = list(CACHE.rglob(f"hybas_{region}_lev06*.shp"))
        if not shp_matches:
            raise FileNotFoundError(f"No shapefile for region={region}")
        gdf = gpd.read_file(shp_matches[0])
        for col in ("MAIN_BAS", "HYBAS_ID"):
            if col not in gdf.columns:
                raise ValueError(f"{col} missing in {shp_matches[0]}")
        frames.append(gdf[["MAIN_BAS", "HYBAS_ID", "geometry"]].copy())
        print(f"Loaded {region}: {len(gdf)} polygons from {shp_matches[0].name}")

    hybas = gpd.GeoDataFrame(pd.concat(frames, ignore_index=True), crs="EPSG:4326")
    hybas["geometry"] = hybas.geometry.buffer(0)
    return hybas


def _assign_hybas(meta: pd.DataFrame, hybas: gpd.GeoDataFrame) -> pd.DataFrame:
    gauges = gpd.GeoDataFrame(
        meta.copy(),
        geometry=gpd.points_from_xy(meta["gauge_lon"], meta["gauge_lat"]),
        crs=4326,
    )
    joined = gpd.sjoin(gauges, hybas, how="left", predicate="within")

    missing = joined[GROUP_COL].isna()
    if missing.any():
        print(f"Point-in-polygon misses: {int(missing.sum())}; using nearest.")
        miss = joined.loc[missing, ["gauge_id", "geometry"]].copy()
        miss_p = miss.to_crs(NA_ALBERS)
        hybas_p = hybas.to_crs(NA_ALBERS)
        nearest = gpd.sjoin_nearest(
            miss_p[["gauge_id", "geometry"]],
            hybas_p,
            how="left",
            distance_col="dist_m",
        )
        nearest = nearest.set_index("gauge_id").loc[miss["gauge_id"]].reset_index()
        joined.loc[missing, "MAIN_BAS"] = nearest["MAIN_BAS"].values
        joined.loc[missing, "HYBAS_ID"] = nearest["HYBAS_ID"].values
        print(
            "Nearest-match distance m: "
            f"median={nearest['dist_m'].median():.0f}, "
            f"max={nearest['dist_m'].max():.0f}"
        )

    joined = (
        joined.sort_values(["gauge_id", GROUP_COL])
        .drop_duplicates(subset=["gauge_id"], keep="first")
        .reset_index(drop=True)
    )
    if joined[GROUP_COL].isna().any():
        n = int(joined[GROUP_COL].isna().sum())
        raise RuntimeError(f"{n} gauges still missing {GROUP_COL}")

    joined["MAIN_BAS"] = joined["MAIN_BAS"].astype(np.int64)
    joined["HYBAS_ID"] = joined["HYBAS_ID"].astype(np.int64)
    return joined


def _split_groups_by_size(
    group_sizes: pd.Series,
    train_frac: float,
    val_frac: float,
    seed: int,
) -> dict[str, set]:
    """Assign groups so no group is split across train/val/test."""
    rng = np.random.default_rng(seed)
    total = int(group_sizes.sum())
    n_train_target = int(total * train_frac)
    n_val_target = int(total * val_frac)

    assign = {"train": set(), "validation": set(), "test": set()}
    counts = {"train": 0, "validation": 0, "test": 0}

    remaining = []
    max_holdout = max(n_val_target, total - n_train_target - n_val_target)
    for g, size in group_sizes.items():
        g = int(g)
        size = int(size)
        if size > max_holdout:
            assign["train"].add(g)
            counts["train"] += size
        else:
            remaining.append((g, size))

    order = np.arange(len(remaining))
    rng.shuffle(order)
    remaining = [remaining[i] for i in order]

    for g, size in remaining:
        if counts["validation"] + size <= n_val_target:
            bucket = "validation"
        elif counts["test"] + size <= (total - n_train_target - n_val_target):
            bucket = "test"
        else:
            val_room = n_val_target - counts["validation"]
            test_room = (total - n_train_target - n_val_target) - counts["test"]
            if size <= val_room and val_room >= test_room:
                bucket = "validation"
            elif size <= test_room:
                bucket = "test"
            else:
                bucket = "train"
        assign[bucket].add(g)
        counts[bucket] += size

    return assign


def _geojson_line_segments(geojson: dict) -> list[np.ndarray]:
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


def _load_borders() -> list[np.ndarray]:
    """Optional outlines for the map. Missing downloads are skipped."""
    borders = []
    sources = {
        "canada_provinces.geojson": (
            "https://raw.githubusercontent.com/codeforamerica/"
            "click_that_hood/master/public/data/canada.geojson"
        ),
        # Public-domain US state outlines (simplified).
        "us_states.geojson": (
            "https://raw.githubusercontent.com/PublicaMundi/"
            "MappingAPI/master/data/geojson/us-states.json"
        ),
    }
    for name, url in sources.items():
        path = OUT / name
        try:
            if not path.exists():
                print(f"Downloading {name}...")
                req = urllib.request.Request(
                    url,
                    headers={
                        "User-Agent": "Mozilla/5.0 (compatible; flood-forecasting)"
                    },
                )
                with urllib.request.urlopen(req, timeout=60) as resp:
                    path.write_bytes(resp.read())
            with path.open(encoding="utf-8") as f:
                borders.extend(_geojson_line_segments(json.load(f)))
        except Exception as exc:  # noqa: BLE001 — map borders are optional
            print(f"Skipping border {name}: {exc}")
    return borders


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    meta = _load_us_canada_multimet_meta()
    print(f"US + Canada MultiMet gauges: {len(meta)}")
    print(meta["country_norm"].value_counts().to_string())

    hybas = _ensure_hydrobasins()
    print(f"HydroBASINS polygons: {len(hybas)}")

    assigned = _assign_hybas(meta, hybas)
    group_sizes = assigned.groupby(GROUP_COL).size().sort_values(ascending=False)
    print(f"Unique {GROUP_COL} (lev06) groups: {len(group_sizes)}")
    print(f"Largest groups ({GROUP_COL}, n_gauges):\n", group_sizes.head(10).to_string())

    group_to_split = _split_groups_by_size(group_sizes, TRAIN_FRAC, VAL_FRAC, SEED)

    def gauges_for(split: str) -> np.ndarray:
        mask = assigned[GROUP_COL].isin(group_to_split[split])
        return assigned.loc[mask, "gauge_id"].astype(str).sort_values().to_numpy()

    train_ids = gauges_for("train")
    val_ids = gauges_for("validation")
    test_ids = gauges_for("test")

    for a, b in (("train", "validation"), ("train", "test"), ("validation", "test")):
        overlap = group_to_split[a] & group_to_split[b]
        if overlap:
            raise RuntimeError(f"{GROUP_COL} overlap {a}/{b}: {overlap}")

    np.savetxt(OUT / "train_hybas06.txt", train_ids, fmt="%s")
    np.savetxt(OUT / "validation_hybas06.txt", val_ids, fmt="%s")
    np.savetxt(OUT / "test_hybas06.txt", test_ids, fmt="%s")
    np.savetxt(OUT / "all_us_canada.txt", assigned["gauge_id"].astype(str).sort_values(), fmt="%s")

    split_lookup = (
        {g: "train" for g in train_ids}
        | {g: "validation" for g in val_ids}
        | {g: "test" for g in test_ids}
    )
    assigned["split"] = assigned["gauge_id"].map(split_lookup)
    meta_out = assigned.drop(columns=["geometry"], errors="ignore")
    for c in ("index_right", "dist_deg", "dist_m"):
        if c in meta_out.columns:
            meta_out = meta_out.drop(columns=[c])
    meta_out.to_csv(OUT / "na_metadata_hybas06.csv", index=False)

    print(f"\nWATERSHED SPLIT (by {GROUP_COL} lev06)")
    print("Train:", len(train_ids), "groups:", len(group_to_split["train"]))
    print("Validation:", len(val_ids), "groups:", len(group_to_split["validation"]))
    print("Test:", len(test_ids), "groups:", len(group_to_split["test"]))
    print(
        "By country x split:\n",
        assigned.groupby(["country_norm", "split"]).size().unstack(fill_value=0).to_string(),
    )

    border_segments = _load_borders()
    colors = {"train": "#1f77b4", "validation": "#ff7f0e", "test": "#2ca02c"}
    fig, ax = plt.subplots(figsize=(12, 8))
    if border_segments:
        ax.add_collection(
            LineCollection(
                border_segments,
                colors="#444444",
                linewidths=0.35,
                alpha=0.7,
                zorder=0,
            )
        )
    for name in ["train", "validation", "test"]:
        d = assigned[assigned["split"] == name]
        ax.scatter(
            d["gauge_lon"],
            d["gauge_lat"],
            s=4,
            c=colors[name],
            label=f"{name} (n={len(d)})",
            alpha=0.55,
            zorder=2,
        )
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title("US + Canada MultiMet: train/val/test by HydroBASINS lev06 HYBAS_ID")
    ax.legend(loc="lower left")
    ax.grid(True, alpha=0.3)
    pad_lon, pad_lat = 3.0, 2.0
    ax.set_xlim(assigned["gauge_lon"].min() - pad_lon, assigned["gauge_lon"].max() + pad_lon)
    ax.set_ylim(assigned["gauge_lat"].min() - pad_lat, assigned["gauge_lat"].max() + pad_lat)
    ax.set_aspect("equal", adjustable="box")
    map_path = OUT / "na_splits_hybas06_map.png"
    fig.tight_layout()
    fig.savefig(map_path, dpi=180)
    plt.close(fig)
    print("Map:", map_path)


if __name__ == "__main__":
    main()
