import torch
import numpy as np
from torch.utils.data import DataLoader
from torcheval.metrics.functional import multiclass_accuracy


def test(net, data_loader, criterion, device, n_classes):
    net.eval()

    with torch.no_grad():
        n_batches = 0
        epoch_loss = 0

        targets = []
        outputs = []

        for x_batch, y_batch in data_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            out = net(x_batch)

            loss = criterion(out, y_batch)
            y_pred = torch.softmax(out, dim=1).argmax(dim=1)

            outputs.append(y_pred.cpu())
            targets.append(y_batch.cpu())

            epoch_loss += loss.item()

            n_batches += 1

        epoch_loss /= n_batches

        outputs = torch.cat(outputs)
        targets = torch.cat(targets)

        acc_micro = multiclass_accuracy(
            outputs, targets, average="micro", num_classes=n_classes
        ).item()
        acc_macro = multiclass_accuracy(
            outputs, targets, average="macro", num_classes=n_classes
        ).item()

    return epoch_loss, acc_micro, acc_macro
