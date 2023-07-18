import os
import types
import torch
import torch.optim.lr_scheduler

from typing import List, Union, Optional
from PIL import Image


def wrap_in_pipeline(model, scheduler, pipeline_class, num_inference_steps: int, default_eta: float = 0.0, def_output_type: str = "torch"):
    """
    Wrap a model in a pipeline for sampling.

    Args:
        model: The model to wrap.
        scheduler: The scheduler to use for sampling.
        pipeline_class: The pipeline class to use.
        num_inference_steps: The number of inference steps to use.
        eta: The eta value to use.
        def_output_type: The output type to use. Options are "torch", "torch_raw", and "pil". Defaults to "torch".
    """
    assert def_output_type in ["torch", "torch_raw", "pil"], f"Invalid output type {def_output_type}"
    
    def generate(self, batch_size: int, target_steps: Union[List[int], int] = 0, generation_steps: int = num_inference_steps, eta: float = default_eta, output_type: str = def_output_type, seed: Optional[int] = None) -> torch.Tensor:   
        generator = torch.manual_seed(seed) if seed is not None else None         
        pipeline = pipeline_class(unet=self, scheduler=scheduler)
        pipeline.set_progress_bar_config(disable=True)
        samples = pipeline(
            batch_size, 
            generator=generator,
            num_inference_steps=generation_steps,
            eta=eta,
            output_type=output_type, 
            target_steps=target_steps,
        )
        return samples
    
    model.generate = types.MethodType(generate, model)


def make_grid(images, rows, cols):
    w, h = images[0].size
    grid = Image.new("RGB", size=(cols * w, rows * h))
    for i, image in enumerate(images):
        grid.paste(image, box=(i % cols * w, i // cols * h))
    return grid


def evaluate_diffusion(output_dir, eval_batch_size, epoch, generator, generation_steps: int = 20, eta: float = 0.0, seed: Optional[int] = None):
    os.makedirs(output_dir, exist_ok=True)
    images = generator.generate(eval_batch_size, output_type="pil", seed=seed, generation_steps=generation_steps, eta=eta)[0]
    image_grid = make_grid(images, rows=10, cols=10)

    # Save the images
    test_dir = os.path.join(output_dir, "samples")
    os.makedirs(test_dir, exist_ok=True)
    image_grid.save(f"{test_dir}/{epoch:04d}.png")