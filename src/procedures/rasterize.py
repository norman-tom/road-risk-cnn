from typing import Literal

import os, glob
from tqdm import tqdm
from pathlib import Path

import numpy as np
import geopandas as gpd

import rasterio
import rasterio.features
from rasterio.enums import MergeAlg

from src import custom_types

ROOT_DIR = Path(os.path.abspath(__file__)).parents[2]
BLOB_DIR = os.path.join(ROOT_DIR, "data", "blob")


def rasterize(
    label: str,
    grid: custom_types.Grid,
    gdf_data: gpd.GeoDataFrame,
    idx_: str,
    value: str = None,
    onehot: str = None,
    merge_alg: Literal["add", "replace"] = "add",
    dtype: np.dtype = np.single,
    samples: int = None,
    replace: bool = False,
) -> None:
    """Rasterize gdf_data.
    :param label:str
    :param grid: custom_types.Grid
    :param gdf_data: gdf_data
    :param idx_: identifier of blob, e.g., "crash"
    :param value: value burned into grid, defaults to None
    :param onehot: value to seperate into channels, defaults to None
    :param merge_alg: Literal["add", "replace"], defaults to "add"
    :param dtype: dtype, defaults to np.single
    :param samples: only generate samples cells, defaults to None
    :param replace:  removes all files, defaults to False
    """

    path = os.path.join(BLOB_DIR, label)
    if not os.path.exists(path):
        os.makedirs(path)

    if replace:
        for f in glob.glob(f"{path}/*"):
            os.remove(f)

    merge_alg = MergeAlg.add if merge_alg == "add" else MergeAlg.replace

    if onehot is not None:
        assert onehot in gdf_data
        channels = [gdf_data[gdf_data[onehot] == c] for c in sorted(gdf_data[onehot].unique())]
    else:
        channels = [gdf_data]

    if value is not None:
        assert value in gdf_data
        channels = [[(g, v) for g, v in zip(c.geometry, c[value].astype(dtype))] for c in channels]
    else:
        channels = [c.geometry for c in channels]

    out = np.zeros((grid.shape[0], len(channels), *grid[0].pixels), dtype=dtype)

    pbar = tqdm(total=len(grid))
    for cell_idx, cell in enumerate(grid):
        if cell_idx == samples:
            break

        transform = rasterio.Affine.translation(
            cell.origin[0], cell.box[1]
        ) * rasterio.Affine.scale(cell.res, -cell.res)

        for channel_idx, channel in enumerate(channels):
            rst = rasterio.features.rasterize(
                shapes=channel,
                fill=0,
                default_value=1,
                transform=transform,
                all_touched=True,
                merge_alg=merge_alg,
                out_shape=grid[0].pixels,
                dtype=dtype,
            )
            out[cell_idx, channel_idx] = rst

        pbar.update(1)

    np.save(os.path.join(path, f"{idx_}_{label}"), out)
    pbar.close()
