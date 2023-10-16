from abc import ABC


class BaseEvaluator(ABC):

    def __init__(self):
        pass

    def evaluate(self, *args, **kwargs):
        raise NotImplementedError 