import os
import pathlib

import torch
from ray import tune
from ray import shutdown

import torchvision

from src import datasets, procedures, utils
from src.models.vgg16 import VGG16
from src.models.resnet18 import RESNET18
from src.report import Reporter

ROOT_DIR = pathlib.Path(__file__).parent.resolve()
RAY_DIR = os.path.join(ROOT_DIR, "ray")


def main():
    # this part of the code trim input data to model area

    # name = "sydney"
    # RAY_DIR = os.path.join(ROOT_DIR, "data/raw/")

    # dir_ = os.path.join(ROOT_DIR, "data/raw")
    # gdf_area = utils.get_gdf("nsw", "area", dir_=dir_)

    # gdf_speed = utils.get_gdf("nsw", "speed", dir_=dir_)
    # gdf_crashes = utils.get_gdf("nsw", "crashes", dir_=dir_)
    # gdf_traffic_lights = utils.get_gdf("nsw", "traffic_lights", dir_=dir_)
    # gdf_functional_hierachy = utils.get_gdf("nsw", "functional_hierachy", dir_=dir_)

    # dir_ = os.path.join(ROOT_DIR, "data/raw/sydney")
    # utils.geo.intersect(gdf_area, gdf_speed, PATH=f"{dir_}/speed")
    # utils.geo.intersect(gdf_area, gdf_crashes, PATH=f"{dir_}/crashes")
    # utils.geo.intersect(gdf_area, gdf_traffic_lights, PATH=f"{dir_}/traffic_lights")
    # utils.geo.intersect(
    #     gdf_area, gdf_functional_hierachy, PATH=f"{dir_}/functional_hierachy"
    # )

    # this part of the code rasterises the data
    # dataset = datasets.SydneyExtended()

    """    name = "sydney"

    dataset_tr = datasets.Sydney

    name = "sydney"
    for architecture in ["VGG16", "RESNET18"]:
        if architecture == "VGG16":
            max_num_epochs = 300
            batch_size = 256
            lr = 1e-5
        else:
            max_num_epochs = 50
            batch_size = 128
            lr = 1e-4

        procedures.tune(
            datasets.SydneyExtended,
            name=name,
            config={
                "features": tune.grid_search(
                    ["glh", "gslh"]
                ),
                "include_outer": True,
                "max_num_epochs": max_num_epochs,
                "architecture": architecture,
                "batch_size": batch_size,
                "weights": [0.42, 2.39, 4.36],
                "lr": lr,
            },
            num_samples=1,
        )
        shutdown() """

    """weights: tune.grid_search(
        [
            [0.42, 2.39, 4.36],  # Lin  1.97x + 0.42
            #[0.42, 2.91, 4.36],  # Log  3.5863 log(x + 1) + 0.42
            #[0.42, 1.35, 4.36],  # Geo  (0.42)(3.2221)^{x} 
        ])"""


    # Reporting
    import warnings
    warnings.simplefilter(action='ignore')
    
    reporter = Reporter(VGG16, "acc_val_macro")
    _, __, dataset_ts = datasets.SydneyExtended(reporter.features).split_dataset()
    reporter.load_dataset(dataset_ts)
    print("_____VGG: TEST______")
    print(f"loss: {reporter.loss()}")
    print(f"acc_macro: {reporter.accuracy('macro')}")
    print(f"acc_micro: {reporter.accuracy('micro')}")
    print(f"recall: {reporter.recall()}")
    print(f"precision: {reporter.precision()}")
    print(reporter.confusion())
    print(reporter.report())
    print("_____RESNET: TEST______")
    print(f"loss: {reporter.loss()}")
    print(f"acc_macro: {reporter.accuracy('macro')}")
    print(f"acc_micro: {reporter.accuracy('micro')}")
    print(f"recall: {reporter.recall()}")
    print(f"precision: {reporter.precision()}")
    print(reporter.confusion())
    print(reporter.report())

    reporter = Reporter(VGG16, "acc_val_macro")
    dataset_tr, dataset_val, _ = datasets.SydneyExtended(reporter.features).split_dataset()
    reporter.load_dataset(dataset_tr)
    print("_____VGG: TRAINING______")
    print(f"loss: {reporter.loss()}")
    print(f"acc_macro: {reporter.accuracy('macro')}")
    print(f"acc_micro: {reporter.accuracy('micro')}")
    print(f"recall: {reporter.recall()}")
    print(f"precision: {reporter.precision()}")
    print(reporter.confusion())
    print(reporter.report())
    reporter.load_dataset(dataset_val)
    print("_____VGG: VALIDATION______")
    print(f"loss: {reporter.loss()}")
    print(f"acc_macro: {reporter.accuracy('macro')}")
    print(f"acc_micro: {reporter.accuracy('micro')}")
    print(f"recall: {reporter.recall()}")
    print(f"precision: {reporter.precision()}")
    print(reporter.confusion())
    print(reporter.report())
    del dataset_tr
    del dataset_val
    del _

    reporter = Reporter(RESNET18, "acc_val_macro")
    dataset_tr, dataset_val, _ = datasets.SydneyExtended(reporter.features).split_dataset()
    reporter.load_dataset(dataset_tr)
    print("_____RESNET: TRAINING______")
    print(f"loss: {reporter.loss()}")
    print(f"acc_macro: {reporter.accuracy('macro')}")
    print(f"acc_micro: {reporter.accuracy('micro')}")
    print(f"recall: {reporter.recall()}")
    print(f"precision: {reporter.precision()}")
    print(reporter.confusion())
    print(reporter.report())
    reporter.load_dataset(dataset_val)
    print("_____RESNET: VALIDATION______")
    print(f"loss: {reporter.loss()}")
    print(f"acc_macro: {reporter.accuracy('macro')}")
    print(f"acc_micro: {reporter.accuracy('micro')}")
    print(f"recall: {reporter.recall()}")
    print(f"precision: {reporter.precision()}")
    print(reporter.confusion())
    print(reporter.report())
    del dataset_tr
    del dataset_val
    del _ 

    reporter = Reporter(VGG16, "acc_val_macro")
    sydney = datasets.SydneyExtended(reporter.features)
    mean = sydney.mean
    std = sydney.std
    del sydney
    dataset_ts = datasets.Newcastle(reporter.features, (mean, std))
    reporter.load_dataset(dataset_ts)
    print("_____VGG: NEWCASTLE______")
    print(f"loss: {reporter.loss()}")
    print(f"acc_macro: {reporter.accuracy('macro')}")
    print(f"acc_micro: {reporter.accuracy('micro')}")
    print(f"recall: {reporter.recall()}")
    print(f"precision: {reporter.precision()}")
    print(reporter.confusion())
    print(reporter.report())
    del dataset_ts

    reporter = Reporter(RESNET18, "acc_val_macro")
    sydney = datasets.SydneyExtended(reporter.features)
    mean = sydney.mean
    std = sydney.std
    del sydney
    dataset_ts = datasets.Newcastle(reporter.features, (mean, std))
    reporter.load_dataset(dataset_ts)
    print("_____RESNET: NEWCASTLE______")
    print(f"loss: {reporter.loss()}")
    print(f"acc_macro: {reporter.accuracy('macro')}")
    print(f"acc_micro: {reporter.accuracy('micro')}")
    print(f"recall: {reporter.recall()}")
    print(f"precision: {reporter.precision()}")
    print(reporter.confusion())
    print(reporter.report())


if __name__ == "__main__":
    main()
