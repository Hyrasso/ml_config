from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict, fields, is_dataclass
from inspect import getfile, getsourcelines
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
        
        self.trainer.fit(self.dataset_builder)

        self.evaluator.evaluate(self.trainer.model)

def parametrize(param_cls):
    """ Used to define config for a class
    
    ```
    >>> @parametrize(optim.Adam)
    ... @dataclass
    ... class Adam(Optim):
    ...     lr: float = 0.3
    ... 
    >>> adam = Adam()
    >>> optim = adam.get(net.parameters())
    ```
    """

    def wrapper(config_cls):
        if not is_dataclass(config_cls):
            raise ValueError("The parametrize decorator should be used on a dataclass")

        # this function signature is the same as config_cls
        def get(self, *args, **kwargs):
            call_kwargs = {f.name: getattr(self, f.name) for f in fields(self)}
            for k, v in kwargs.items():
                if k in call_kwargs:
                    raise ValueError(f"Bad argument '{k}', it is already set in the config class '{config_cls.__name__}' at {getfile(config_cls)}:{getsourcelines(config_cls)[1]}")
                else:
                    call_kwargs[k] = v
            return param_cls(*args, **call_kwargs)
        
        config_cls.get = get
        return config_cls
    
    return wrapper
