import os
import yaml
from typing import Any
from munch import DefaultMunch


def get_configuration(yaml_path: str) -> Any:
    with open(yaml_path, "r") as f:
        yaml_dict = yaml.safe_load(f)
        return DefaultMunch.fromDict(yaml_dict)
    

def extract_into_tensor(arr, timesteps, broadcast_shape):
    """
    Extract values from a 1-D numpy array for a batch of indices.

    :param arr: the 1-D numpy array.
    :param timesteps: a tensor of indices into the array to extract.
    :param broadcast_shape: a larger shape of K dimensions with the batch
                            dimension equal to the length of timesteps.
    :return: a tensor of shape [batch_size, 1, ...] where the shape has K dims.
    """
    res = arr.to(timesteps.device)[timesteps].float()
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res.expand(broadcast_shape)