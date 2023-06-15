import os
import numpy as np
from pathlib import Path

import torch
from ray import tune
import geopandas as gpd

ROOT_DIR = Path(os.path.abspath(__file__)).parents[2]
GDF_DIR = os.path.join(ROOT_DIR, "data", "raw")
BLOB_DIR = os.path.join(ROOT_DIR, "data", "blob")
MODEL_DIR = os.path.join(ROOT_DIR, "models")
RAY_DIR = os.path.join(ROOT_DIR, "ray")


def get_gdf(dataset_label: str, id_: str, dir_=GDF_DIR, epsg=3308) -> gpd.GeoDataFrame:
    """
    :param dataset_label: str
    :param id_: identifier, e.g., "crashes"
    :param dir_: defaults to GDF_DIR
    :param epsg: defaults to 3308
    :return: gpd.GeoDataFrame
    """
    path = os.path.join(dir_, dataset_label, id_)
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    gdf = gpd.read_file(path)
    if gdf.crs is None or gdf.crs.to_epsg() != epsg:
        gdf = gdf.to_crs(epsg=epsg)
    return gdf


def get_blob(dataset_label: str, id_: list, dir_=BLOB_DIR) -> np.ndarray:
    """
    :param dataset_label: str
    :param id_: identifier, e.g., "['X_p1', 'X_p2']" for partition 1 and 2 of the examples set. 
    :param dir_: defaults to BLOB_DIR
    :return: np.ndarray
    """
    data = []
    for id in id_:
        path = os.path.join(dir_, dataset_label, f"{id}_dataset.npy")
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        data.append(np.load(path))
    return np.concatenate(data)


def save_net(net, opt, fname: str):
    torch.save(
        {
            "net_state_dict": net.state_dict(),
            "opt_state_dict": opt.state_dict(),
        },
        os.path.join(MODEL_DIR, f"{fname}.pt"),
    )

def get_net(net_cls, cfg, fname: str):
    """
    :param net_cls: net class
    :param cfg: net configuration
    :param fname: _description_
    :return: net
    """
    net = net_cls(**cfg)
    checkpoint = torch.load(os.path.join(MODEL_DIR, f"{fname}.pt"))
    net.load_state_dict(checkpoint["net_state_dict"])
    return net
