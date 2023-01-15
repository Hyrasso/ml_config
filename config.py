from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict
from typing import TypeVar, Generic
from pprint import pprint

Data = TypeVar("Data")

@dataclass
class Model(ABC):
    ...

    @abstractmethod
    def __call__(self, x):
        """ Defines forward pass"""
        ...
    
    @abstractmethod
    def predict(self, x):
        """ Forward pass in evaluation mode """
        ...

@dataclass
class Trainer(ABC, Generic[Data]):
    model: Model

    @abstractmethod
    def fit(self, data: Data):
        ...

@dataclass
class DatasetBuilder(ABC, Generic[Data]):
    ...

    @abstractmethod
    def prepare_data(self) -> Data:
        ...

@dataclass
class Evaluator(ABC):
    ...

    @abstractmethod
    def evaluate(self, model: Model):
        ...

@dataclass
class Config:
    trainer: Trainer
    dataset_builder: DatasetBuilder
    evaluator: Evaluator

    def launch_training(self):
        pprint(self)
        pprint(asdict(self))
        
        data = self.dataset_builder.prepare_data()

        self.trainer.fit(data)

        self.evaluator.evaluate(self.trainer.model)