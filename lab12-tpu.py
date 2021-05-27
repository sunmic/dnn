import torchvision.transforms as transforms
import torchvision.datasets as datasets


from clearml import Task


task = Task.init(project_name="DNNPWr", task_name="lab12-tpu")
task.execute_remotely("tpu")

# imports the torch_xla package
import torch_xla
import torch_xla.core.xla_model as xm
dev = xm.xla_device()

transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

root = './root'
train_ds = datasets.STL10(root, split='train', transform=transform, download=True)
test_ds = datasets.STL10(root, split='test', transform=transform, download=True)
unlabeled_ds = datasets.STL10(root, split='unlabeled', transform=transform, download=True)

from torch.utils.data import DataLoader
train_dl = DataLoader(train_ds, batch_size=32, shuffle=True, pin_memory=True)
test_dl = DataLoader(test_ds, batch_size=32, pin_memory=True)

import torchvision
import torchvision.transforms as transforms
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Tuple
from tqdm import tqdm

from sklearn.metrics import f1_score
import numpy as np
import time
from torch.utils.tensorboard import SummaryWriter

def count_correct(
        y_pred: torch.Tensor, y_true: torch.Tensor
) -> torch.Tensor:
    preds = torch.argmax(y_pred, dim=1)
    return (preds == y_true).float().sum()

def validate(
        model: nn.Module,
        loss_fn: torch.nn.CrossEntropyLoss,
        dataloader: DataLoader
) -> Tuple[torch.Tensor, torch.Tensor]:
    loss = 0
    correct = 0
    all = 0
    for X_batch, y_batch in dataloader:
        y_pred = model(X_batch.to(dev))
        all += len(y_pred)
        loss += loss_fn(y_pred, y_batch.to(dev)).sum()
        correct += count_correct(y_pred, y_batch.to(dev))
    return loss / all, correct / all

def fit(
        model: nn.Module, optimiser: optim.Optimizer,
        loss_fn: torch.nn.CrossEntropyLoss, train_dl: DataLoader,
        val_dl: DataLoader, epochs: int,
        writer : SummaryWriter,
        print_metrics: str = True,
        epoch_offset : int = 0
):
    for epoch in range(epoch_offset, epoch_offset + epochs):
        epoch_start = time.time()
        model.train()
        for X_batch, y_batch in tqdm(train_dl):
            y_pred = model(X_batch.to(dev))
            loss = loss_fn(y_pred, y_batch.to(dev))
            loss = loss_fn(y_pred, y_batch.to(dev))

            loss.backward()
            optimiser.step()
            optimiser.zero_grad()

        if print_metrics:
            model.eval()
            with torch.no_grad():
                train_loss, train_acc = validate(
                    model=model, loss_fn=loss_fn, dataloader=train_dl
                )
                val_loss, val_acc = validate(
                    model=model, loss_fn=loss_fn, dataloader=val_dl
                )
                elapsed = time.time() - epoch_start

                for tag, value in [
                    ('Train/Loss', train_loss),
                    ('Val/Loss', val_loss),
                    ('Train/Accuracy', train_acc),
                    ('Val/Accuracy', val_acc)
                ]:
                    writer.add_scalar(tag=tag, scalar_value=value, global_step=epoch)


                print(
                    f"Epoch {epoch}: "
                    f"({round(elapsed)}s.) "
                    f"train loss = {train_loss:.3f} (acc: {train_acc:.3f}), "
                    f"validation loss = {val_loss:.3f} (acc: {val_acc:.3f})"
                )

import socket
import os
hostname = socket.gethostname()
print(hostname)
wkdir = '.'
# if hostname != 'LAPTOP-LKDD3MT2':
#     from google.colab import drive
#     drive.mount("/content/drive/")
#     wkdir = '/content/drive/MyDrive/'
print(f"Work directory: {wkdir}")

from torch.utils.tensorboard import SummaryWriter

log_dir = f'{wkdir}/lab12/logs/'
os.makedirs(log_dir, exist_ok=True)

def desc(name: str, value: bool):
    return name if value else f'no-{name}'

def get_writer(experiment_name: str):
    from datetime import datetime
    date = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    writer = SummaryWriter(f'{log_dir}/{date}-{experiment_name}', flush_secs=2)
    return writer

import torchvision.models as models

class LabVGG19(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = torch.nn.Sequential()
        # dokończ konstruktor
        vgg19 = models.vgg19(pretrained=True)
        self.part0 = vgg19.features[0:4]
        self.part1 = vgg19.features[4:9]
        self.part2 = vgg19.features[9:18]

    def forward(self, x):
        # dokończ forward
        out0 = self.part0(x)
        out1 = self.part1(out0)
        out2 = self.part2(out1)
        return out0, out1, out2

class CustomVGG19C10(torch.nn.Module):
    def __init__(self, labels, pool_size = 7, freeze: bool = False):
        super().__init__()
        self.labvgg19 = LabVGG19()
        self.adaptive_avg_pool = nn.AdaptiveAvgPool2d(output_size=(pool_size, pool_size))
        self.classifier = torch.nn.Sequential()
        self.classifier.add_module("flatten", torch.nn.Flatten())
        self.classifier.add_module("linear_1", torch.nn.Linear(in_features=(64 + 128 + 256)*pool_size*pool_size, out_features=labels))

        if freeze:
            for param in self.labvgg19.parameters():
                param.requires_grad = False

    def forward(self, x):
        out0, out1, out2 = self.labvgg19(x)
        out0, out1, out2 = map(self.adaptive_avg_pool, (out0, out1, out2))
        out = torch.cat([out0, out1, out2], dim=1)
        return self.classifier(out)


#Eksperymenty
epochs = 50
size = 1

for freeze in [False, True]:
    experiment_name = f"CustVGG19-size-{size}-{desc('freeze', freeze)}"
    writer = get_writer(experiment_name)

    model = CustomVGG19C10(labels=10, pool_size=size, freeze=freeze).to(dev)

    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())
    fit(model, optimizer, loss_fn, train_dl, test_dl, epochs=epochs, writer=writer)

    writer.close()

