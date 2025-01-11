from abc import ABC, abstractmethod
from typing import Callable

from dspy import Program


class BaseTask(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def load_dataset(self):
        pass

    @abstractmethod
    def get_program(self) -> Program:
        pass

    @abstractmethod
    def get_metric(self) -> Callable:
        pass

    def get_trainset(self, TRAIN_NUM=None):
        return self.trainset[:TRAIN_NUM]

    def get_devset(self, DEV_NUM=None):
        if hasattr(self, "devset"):
            return self.devset[:DEV_NUM]

        if DEV_NUM is None:
            return self.trainset

        return self.trainset[-DEV_NUM:]

    def get_testset(self, TEST_NUM=None):
        return self.testset[:TEST_NUM]

    def set_splits(self, TRAIN_NUM=None, DEV_NUM=None, TEST_NUM=None):
        if TRAIN_NUM:
            self.TRAIN_NUM = TRAIN_NUM
        if DEV_NUM:
            self.DEV_NUM = DEV_NUM
        if TEST_NUM:
            self.TEST_NUM = TEST_NUM

    def get_max_tokens(self):
        return 150
