import torchvision.transforms as transforms
import torchvision.datasets as datasets

from clearml import Task


task = Task.init(project_name="DNNPWr", task_name="lab12")
task.execute_remotely("default")


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
        y_pred = model(X_batch.cuda())
        all += len(y_pred)
        loss += loss_fn(y_pred, y_batch.cuda()).sum()
        correct += count_correct(y_pred, y_batch.cuda())
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
            y_pred = model(X_batch.cuda())
            loss = loss_fn(y_pred, y_batch.cuda())

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

epochs = 50

for pretrained in [True, False]:
    experiment_name = f"VGG19-{desc('pretrained', pretrained)}"
    writer = get_writer(experiment_name)

    model = models.vgg19(pretrained=pretrained).cuda()

    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())
    fit(model, optimizer, loss_fn, train_dl, test_dl, epochs=epochs, writer=writer)

    writer.close()

