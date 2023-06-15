import os
import geopandas as gpd
from pathlib import Path

from src import utils

ROOT_DIR = Path(os.path.abspath(__file__)).parents[2]


def intersect(
    gdf_polytope: gpd.GeoDataFrame,
    gdf_data: gpd.GeoDataFrame,
    how="inner",
    predicate="intersects",
    PATH: str = None,
) -> gpd.GeoDataFrame:
    """Spatially joins gdf_polytope and gdf_data and removes according records from gdf_data.

    :param gdf_polytope: gpd.GeoDataFrame
    :param gdf_data: gpd.GeoDataFrame
    :param how: defaults to "inner"
    :param predicate: defaults to "intersects"
    :param PATH: save gdf_data, defaults to None
    :return: Filtered gdf_data
    """
    gdf_data["index_"] = gdf_data.index
    gdf_intersect = gdf_polytope.sjoin(gdf_data, how=how, predicate=predicate)["index_"]
    gdf_data = gdf_data[gdf_data.index_.isin(gdf_intersect)]

    if PATH:
        gdf_data.to_file(PATH, encoding="utf-8")

    return gdf_data


def get_bounds(gdf: gpd.GeoDataFrame):
    """Returns origin, height, width for a given gdf."""
    minx, miny, maxx, maxy = gdf.total_bounds
    origin = (minx, miny)
    height = maxy - miny
    width =  maxx - minx
    return origin, height, width
