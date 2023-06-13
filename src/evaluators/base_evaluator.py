from abc import ABC


class BaseEvaluator(ABC):

    def __init__(self):
        pass

    def on_epoch_end(self, *args, **kwargs):
        raise NotImplementedError 