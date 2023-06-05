import torch

from torchmetrics.image.fid import FrechetInceptionDistance
from avalanche.evaluation import Metric
from avalanche.evaluation.metrics.mean import Mean


# a standalone metric implementation
class FIDMetric(Metric[float]):
    """
    This metric computes the Frechet Inception Distance (FID) between two
    distributions of images. It uses the FID implementation from
    `torchmetrics <https://torchmetrics.readthedocs.io/en/latest/>`_.
    """
    def __init__(self):
        self._mean_accuracy = Mean()

    @torch.no_grad()
    def update(
        self,
        predicted_y: torch.Tensor,
        true_y: torch.Tensor,
    ) -> None:
        true_y = torch.as_tensor(true_y)
        predicted_y = torch.as_tensor(predicted_y)

        if len(true_y) != len(predicted_y):
            raise ValueError("Size mismatch for true_y and predicted_y tensors")
        
        fid = FrechetInceptionDistance()
        self._mean_accuracy.update(fid(predicted_y, true_y))


    def result(self) -> float:
        return self._mean_accuracy.result()

    def reset(self):
        self._mean_accuracy.reset()