import os
from pathlib import Path

import torch
from ray import tune
from torch.utils.data import DataLoader

from src import models, procedures

ROOT_DIR = Path(os.path.abspath(__file__)).parents[1]
LOG_DIR = os.path.join(ROOT_DIR, "log")


def train(config):
    seed = config["seed"]
    torch.manual_seed(seed)

    dataset = config["dataset"](config["features"])
    batch_size = config["batch_size"]

    include_outer = config["include_outer"]
    dataset_tr, dataset_val, _ = dataset.split_dataset(include_outer, seed)
    loader_tr = DataLoader(dataset_tr, batch_size=batch_size, shuffle=True)
    loader_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=False)

    n_in_channels = dataset_tr.X.shape[1]
    n_classes = int((torch.max(dataset_tr.Y) + 1).item())

    if config["architecture"] == "VGG16":
        net = models.VGG16(n_in_channels, n_classes)
    elif config["architecture"] == "RESNET18":
        net = models.RESNET18(n_in_channels, n_classes)
    elif config["architecture"] == "CUSTOM":
        net = models.CustomNet(n_in_channels, n_classes)

    # net = models.Net(n_in_channels, config["architecture"], n_classes=n_classes)

    if config["cuda"]:
        net.to("cuda")

    device = next(net.parameters()).device

    optimizer = torch.optim.Adam(net.parameters(), lr=config["lr"])

    # criterion = torch.nn.BCELoss()
    # weights = [0.42, 2.18, 4.36]
    # class_weights = torch.FloatTensor(weights).to(device)

    iteration = 0

    class_weights = torch.FloatTensor(config["weights"]).to(device)
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
    for i in range(config["max_num_epochs"]):
        loss_tr = 0
        net.train()
        n_batch = 0

        for n_batch, (x_batch, y_batch) in enumerate(loader_tr, 1):
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            optimizer.zero_grad()
            out = net(x_batch)
            loss = criterion(out, y_batch)
            loss.backward()
            loss_tr += loss.item()
            optimizer.step()

        loss_val, acc_val_micro, acc_val_macro = procedures.test(
            net, loader_val, criterion, device, n_classes
        )
        with tune.checkpoint_dir(i) as checkpoint_dir:
            path = os.path.join(checkpoint_dir, "checkpoint")
            torch.save((net.state_dict(), optimizer.state_dict()), path)
        tune.report(
            loss_tr=loss_tr / n_batch,
            loss_val=loss_val,
            acc_val_micro=acc_val_micro,
                acc_val_macro=acc_val_macro,
        )
        iteration += 1