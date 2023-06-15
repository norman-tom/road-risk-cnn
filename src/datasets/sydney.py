import os
from pathlib import Path

import numpy as np

from src import custom_types, utils, procedures
from src.custom_types import Grid

ROOT_DIR = Path(os.path.abspath(__file__)).parents[2]
BLOB_DIR = os.path.join(ROOT_DIR, "data", "blob")

from sklearn.model_selection import train_test_split


class Sydney(custom_types.Dataset):
    def __init__(self):
        label = "sydney"
        cell_dim = 384

        gdf = utils.get_gdf(label, "area")
        origin, h, w = utils.get_bounds(gdf)

        gridX = Grid(
            origin=origin,
            label=label,
            res=4,
            height=h,
            width=w,
            cell_dim=cell_dim,
            pad=True,
        )
        gridY = Grid(
            origin=origin,
            label=label,
            res=128,
            height=h,
            width=w,
            cell_dim=cell_dim,
            pad=True,
        )

        if not os.path.isdir(os.path.join(BLOB_DIR, label)):
            print(f"{os.path.join(BLOB_DIR, label)} not found; generating ...")
            gdf_speed = utils.get_gdf(label, "speedzones")
            gdf_speed["Speed"] = gdf_speed["Speed"].str.replace(" km/h", "")
            procedures.rasterize(
                label,
                gridX,
                gdf_speed,
                idx_="speed",
                value="Speed",
                merge_alg="replace",
            )

            gdf_crashes = utils.get_gdf(label, "crashes")
            procedures.rasterize(label, gridY, gdf_crashes, idx_="crash")

        X = utils.get_blob(label, ["X_p1", "X_p2", "X_p3", "X_p4"])
        Y = utils.get_blob(label, ["Y_p1", "Y_p2", "Y_p3", "Y_p4"])
        X = np.delete(X, np.s_[1:], axis=1)                    #Delete the traffic lights and road hierarchy features.           
        X[:, 0] = (X[:, 0] - X[:, 0].mean()) / X[:, 0].std()   #Normalise the speed channel across the whole dataset.
        
        maskY = np.zeros(9)
        maskY[4] = 1

        Y = Y.reshape((Y.shape[0], -1)) @ maskY.T
        
        # Define classes
        idx_cls_high_risk = Y > 2
        idx_cls_med_risk = np.where(np.logical_and(Y >= 1, Y <= 2))
        Y[idx_cls_high_risk] = 2
        Y[idx_cls_med_risk] = 1
        # cls_low_risk: 0

        super().__init__(label, gridX, X, gridY, Y)

    def split_dataset(
        self, include_outer=True, seed=0
    ) -> tuple[custom_types.Dataset, custom_types.Dataset, custom_types.Dataset]:
        X_tr, X_ts, Y_tr, Y_ts = train_test_split(
            self.X,
            self.Y,
            test_size=0.2,
            stratify=self.Y,
            shuffle=True,
            random_state=seed,
        )
        X_tr, X_val, Y_tr, Y_val = train_test_split(
            X_tr, Y_tr, test_size=0.25, stratify=Y_tr, shuffle=True, random_state=seed
        )
        if include_outer:
            return (
                custom_types.Dataset(self.label, self.gridX, X_tr, self.gridY, Y_tr),
                custom_types.Dataset(self.label, self.gridX, X_val, self.gridY, Y_val),
                custom_types.Dataset(self.label, self.gridX, X_ts, self.gridY, Y_ts),
            )
        else:
            return (
                custom_types.Dataset(self.label, self.gridX, X_tr[:, :, 32:64, 32:64], self.gridY, Y_tr),
                custom_types.Dataset(self.label, self.gridX, X_val[:, :, 32:64, 32:64], self.gridY, Y_val),
                custom_types.Dataset(self.label, self.gridX, X_ts[:, :, 32:64, 32:64], self.gridY, Y_ts),
            )
            