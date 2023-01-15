from dataclasses import dataclass, InitVar
from config import Config, Trainer, Model, DatasetBuilder, Evaluator
from pathlib import Path
from typing import Any

import torch
from torch.nn.modules import Module

from torchvision.datasets import MNIST
from torchvision import transforms

class MnistModule(Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 8, kernel_size=(3, 3))
        self.bn1 = torch.nn.BatchNorm2d(8)
        self.conv2 = torch.nn.Conv2d(8, 16, kernel_size=(3, 3), stride=(2, 2))
        self.bn2 = torch.nn.BatchNorm2d(16)
        self.conv3 = torch.nn.Conv2d(16, 32, kernel_size=(3, 3), stride=(2, 2))
        self.bn3 = torch.nn.BatchNorm2d(32)
        self.pool = torch.nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = torch.nn.Flatten()
        self.out = torch.nn.Linear(32, config.n_class)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.pool(x)
        x = self.flatten(x)
        return self.out(x)

@dataclass
class MnistModel(Model):
    n_class: int = 10

    def __post_init__(self):
        self._model = MnistModule(self)
    
    def __call__(self, x):
        return self._model(x)
    
    def predict(self, x):
        self._model.eval()
        out = self._model(x)
        self._model.train()
        return out

@dataclass
class Adam:
    lr: float = 0.05

    def optim(self, params):
        return torch.optim.Adam(params, self.lr)

@dataclass
class Loss:
    loss_name: str = "CrossEntropyLoss"

    def func(self):
        if self.loss_name == "CrossEntropyLoss":
            return torch.nn.CrossEntropyLoss()
        else:
            raise ValueError(f"Unknown loss name {self.loss_name}")

@dataclass
class MnistTrainer(Trainer):
    model: MnistModel = MnistModel()

    epochs: int = 2
    optimizer: Adam = Adam()
    loss: Loss = Loss()

    print_every_n: int = 100

    def fit(self, data):
        train_loader, eval_loader = data

        optimizer = self.optimizer.optim(self.model._model.parameters())
        criterion = self.loss.func()

        for epoch in range(self.epochs):  # loop over the dataset multiple times

            running_loss = 0.0
            for i, data in enumerate(train_loader, 0):
                if i > 1000:break
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                # print statistics
                running_loss += loss.item()
                if i % self.print_every_n == self.print_every_n - 1:    # print every 2000 mini-batches
                    print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / self.print_every_n:.3f}')
                    running_loss = 0.0
            
            eval_loss = 0.
            i = 0
            for i, data in enumerate(eval_loader):
                inputs, labels = data
                outputs = self.model.predict(inputs)
                loss = criterion(outputs, labels)
                eval_loss += loss.item()
            print(f'Eval loss: {eval_loss / i}')
            
        print('Finished Training')

@dataclass
class MnistDatasetBuilder(DatasetBuilder):
    root: Path = Path("data/mnist")
    batch_size: int = 32

    def prepare_data(self):
        transform = transforms.Compose(
            [transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))])
        train_dataset = MNIST(root=str(self.root), train=True, download=True, transform=transform)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=0)
        eval_dataset = MNIST(root=str(self.root), train=False, transform=transform)
        eval_loader = torch.utils.data.DataLoader(eval_dataset, batch_size=self.batch_size, shuffle=False, num_workers=0)
        return train_loader, eval_loader

@dataclass
class MnistEvaluator(Evaluator):
    root: Path = Path("data/mnist")

    def evaluate(self, model: Model):
        transform = transforms.Compose(
            [transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))])
        eval_dataset = MNIST(root=str(self.root), train=False, transform=transform)
        eval_loader = torch.utils.data.DataLoader(eval_dataset, batch_size=64, shuffle=False, num_workers=0)

        avg_acc = 0.0
        i = 0
        for i, (inputs, outputs) in enumerate(eval_loader):
            preds = model.predict(inputs)
            avg_acc += (outputs == preds.argmax(axis=1)).float().mean().item()

        print(f"Final avg acc: {avg_acc / i * 100:.3f}")

if __name__ == "__main__":
    mnist_config = Config(
        MnistTrainer(),
        MnistDatasetBuilder(),
        MnistEvaluator()
    )

    mnist_config.launch_training()
