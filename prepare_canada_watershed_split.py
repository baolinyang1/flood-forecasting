"""
Create Canada train/val/test splits by HydroBASINS terminal watershed (MAIN_BAS).

Caravan/HYSETS does not ship a terminal-watershed ID. Nature Flood Hub–style
hydrologic separation uses HydroBASINS: each gauge is assigned to MAIN_BAS
(the terminal / primary basin that drains to the ocean or an endorheic sink).
Whole MAIN_BAS groups are assigned to only one of train / validation / test
so nested or same-river gauges are not split across periods.

Outputs (does NOT overwrite the random seed-42 split files):
  canada_experiment/basins/train_watershed.txt
  canada_experiment/basins/validation_watershed.txt
  canada_experiment/basins/test_watershed.txt
  canada_experiment/basins/canada_metadata_watershed.csv
  canada_experiment/basins/canada_splits_watershed_map.png
"""

from __future__ import annotations

import zipfile
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

CARAVAN = Path(r"C:\Users\nyk750\Caravan-nc\Caravan-nc")
REPO = Path(r"C:\Users\nyk750\flood-forecasting")
OUT = REPO / "canada_experiment" / "basins"
CACHE = OUT / "hydrobasins_cache"

# Level-06 HydroBASINS. MAIN_BAS = terminal / primary basin ID.
# North America alone stops ~62.7N; Arctic layer covers northern Canada.
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
SEED = 42
TRAIN_FRAC = 0.80
VAL_FRAC = 0.10


def _load_canada_multimet_meta() -> pd.DataFrame:
    meta_file = next(CARAVAN.rglob("attributes_other_hysets.csv"))
    df = pd.read_csv(meta_file)
    id_col = "gauge_id"
    country_col = next(c for c in df.columns if "country" in c.lower())

    canada = df[
        df[country_col].astype(str).str.strip().str.lower().eq("canada")
    ].copy()

    multimet_ids = {
        x.strip()
        for x in (
            REPO / "example-configs" / "multimet-basins-list-without-chirps.txt"
        )
        .read_text()
        .splitlines()
        if x.strip()
    }
    canada = canada[canada[id_col].astype(str).isin(multimet_ids)].copy()
    canada[id_col] = canada[id_col].astype(str)
    return canada


def _download_zip(url: str, zip_path: Path) -> None:
    import urllib.request

    print(f"Downloading {url} -> {zip_path}")
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "Mozilla/5.0 (compatible; flood-forecasting)"},
    )
    with urllib.request.urlopen(req, timeout=300) as resp, open(
        zip_path, "wb"
    ) as out:
        out.write(resp.read())


def _ensure_hydrobasins() -> gpd.GeoDataFrame:
    """Load NA + Arctic HydroBASINS lev06 (MAIN_BAS = terminal watershed)."""
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
        if "MAIN_BAS" not in gdf.columns:
            raise ValueError(f"MAIN_BAS missing in {shp_matches[0]}")
        frames.append(gdf[["MAIN_BAS", "HYBAS_ID", "geometry"]].copy())
        print(f"Loaded {region}: {len(gdf)} polygons from {shp_matches[0].name}")

    hybas = gpd.GeoDataFrame(pd.concat(frames, ignore_index=True), crs="EPSG:4326")
    hybas["geometry"] = hybas.geometry.buffer(0)
    return hybas


def _assign_main_bas(meta: pd.DataFrame, hybas: gpd.GeoDataFrame) -> pd.DataFrame:
    gauges = gpd.GeoDataFrame(
        meta.copy(),
        geometry=gpd.points_from_xy(meta["gauge_lon"], meta["gauge_lat"]),
        crs=4326,
    )

    joined = gpd.sjoin(
        gauges,
        hybas,
        how="left",
        predicate="within",
    )

    # Coastal / border misses: nearest polygon fallback.
    missing = joined["MAIN_BAS"].isna()
    if missing.any():
        print(f"Point-in-polygon misses: {int(missing.sum())}; using nearest.")
        miss = joined.loc[missing, ["gauge_id", "geometry"]].copy()
        miss_p = miss.to_crs(3347)  # StatsCan Lambert
        hybas_p = hybas.to_crs(3347)
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

    # One row per gauge (sjoin can duplicate if on boundary).
    joined = (
        joined.sort_values(["gauge_id", "MAIN_BAS"])
        .drop_duplicates(subset=["gauge_id"], keep="first")
        .reset_index(drop=True)
    )

    if joined["MAIN_BAS"].isna().any():
        n = int(joined["MAIN_BAS"].isna().sum())
        raise RuntimeError(f"{n} gauges still missing MAIN_BAS")

    joined["MAIN_BAS"] = joined["MAIN_BAS"].astype(np.int64)
    return joined


def _split_groups_by_size(
    group_sizes: pd.Series,
    train_frac: float,
    val_frac: float,
    seed: int,
) -> dict[str, set]:
    """Assign MAIN_BAS groups so no group is split across train/val/test.

    Large terminal basins (bigger than the val/test budget) are forced into
    train, then remaining groups are shuffled and packed into val/test/train.
    """
    rng = np.random.default_rng(seed)
    total = int(group_sizes.sum())
    n_train_target = int(total * train_frac)
    n_val_target = int(total * val_frac)
    n_test_target = total - n_train_target - n_val_target

    assign = {"train": set(), "validation": set(), "test": set()}
    counts = {"train": 0, "validation": 0, "test": 0}

    # Force oversized groups into train (cannot fit in 10% buckets).
    remaining = []
    max_holdout = max(n_val_target, n_test_target)
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
        # Fill val then test toward targets; otherwise train.
        if counts["validation"] + size <= n_val_target:
            bucket = "validation"
        elif counts["test"] + size <= n_test_target:
            bucket = "test"
        else:
            # Prefer whichever holdout is farther below target if small leftover room.
            val_room = n_val_target - counts["validation"]
            test_room = n_test_target - counts["test"]
            if size <= val_room and val_room >= test_room:
                bucket = "validation"
            elif size <= test_room:
                bucket = "test"
            else:
                bucket = "train"
        assign[bucket].add(g)
        counts[bucket] += size

    return assign


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    meta = _load_canada_multimet_meta()
    print(f"Canada x MultiMet gauges: {len(meta)}")

    hybas = _ensure_hydrobasins()
    print(f"HydroBASINS polygons: {len(hybas)}")

    assigned = _assign_main_bas(meta, hybas)
    group_sizes = assigned.groupby("MAIN_BAS").size().sort_values(ascending=False)
    print(f"Unique MAIN_BAS (terminal) groups: {len(group_sizes)}")
    print(
        "Largest groups (MAIN_BAS, n_gauges):\n",
        group_sizes.head(10).to_string(),
    )

    group_to_split = _split_groups_by_size(
        group_sizes, TRAIN_FRAC, VAL_FRAC, SEED
    )

    def gauges_for(split: str) -> np.ndarray:
        mask = assigned["MAIN_BAS"].isin(group_to_split[split])
        return assigned.loc[mask, "gauge_id"].astype(str).sort_values().to_numpy()

    train_ids = gauges_for("train")
    val_ids = gauges_for("validation")
    test_ids = gauges_for("test")

    # Sanity: no MAIN_BAS shared across splits.
    for a, b in (("train", "validation"), ("train", "test"), ("validation", "test")):
        overlap = group_to_split[a] & group_to_split[b]
        if overlap:
            raise RuntimeError(f"MAIN_BAS overlap {a}/{b}: {overlap}")

    np.savetxt(OUT / "train_watershed.txt", train_ids, fmt="%s")
    np.savetxt(OUT / "validation_watershed.txt", val_ids, fmt="%s")
    np.savetxt(OUT / "test_watershed.txt", test_ids, fmt="%s")

    split_lookup = (
        {g: "train" for g in train_ids}
        | {g: "validation" for g in val_ids}
        | {g: "test" for g in test_ids}
    )
    assigned["split"] = assigned["gauge_id"].map(split_lookup)
    meta_out = assigned.drop(columns=["geometry"], errors="ignore")
    # Drop sjoin helper cols if present.
    for c in ("index_right", "dist_deg"):
        if c in meta_out.columns:
            meta_out = meta_out.drop(columns=[c])
    meta_out.to_csv(OUT / "canada_metadata_watershed.csv", index=False)

    print("\nWATERSHED SPLIT (by MAIN_BAS groups)")
    print("Train:", len(train_ids), "groups:", len(group_to_split["train"]))
    print("Validation:", len(val_ids), "groups:", len(group_to_split["validation"]))
    print("Test:", len(test_ids), "groups:", len(group_to_split["test"]))

    # Map with Canada / province outlines (reuse cache from prepare_canada.py).
    import json
    import urllib.request
    from matplotlib.collections import LineCollection

    border_path = OUT / "canada_provinces.geojson"
    border_url = (
        "https://raw.githubusercontent.com/codeforamerica/"
        "click_that_hood/master/public/data/canada.geojson"
    )
    if not border_path.exists():
        print("Downloading Canada province borders...")
        req = urllib.request.Request(
            border_url,
            headers={"User-Agent": "Mozilla/5.0 (compatible; flood-forecasting)"},
        )
        with urllib.request.urlopen(req, timeout=60) as resp:
            border_path.write_bytes(resp.read())

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

    with border_path.open(encoding="utf-8") as f:
        border_segments = _geojson_line_segments(json.load(f))

    colors = {"train": "#1f77b4", "validation": "#ff7f0e", "test": "#2ca02c"}
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
        d = assigned[assigned["split"] == name]
        ax.scatter(
            d["gauge_lon"],
            d["gauge_lat"],
            s=12,
            c=colors[name],
            label=f"{name} (n={len(d)})",
            alpha=0.75,
            zorder=2,
        )
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(
        "Canada MultiMet: train/val/test by HydroBASINS MAIN_BAS (terminal)"
    )
    ax.legend(loc="lower left")
    ax.grid(True, alpha=0.3)
    pad_lon, pad_lat = 3.0, 2.0
    ax.set_xlim(
        assigned["gauge_lon"].min() - pad_lon,
        assigned["gauge_lon"].max() + pad_lon,
    )
    ax.set_ylim(
        assigned["gauge_lat"].min() - pad_lat,
        assigned["gauge_lat"].max() + pad_lat,
    )
    ax.set_aspect("equal", adjustable="box")
    map_path = OUT / "canada_splits_watershed_map.png"
    fig.tight_layout()
    fig.savefig(map_path, dpi=200)
    plt.close(fig)
    print("Map:", map_path)
    print("\nPoint configs at *_watershed.txt to use this split.")


if __name__ == "__main__":
    main()
