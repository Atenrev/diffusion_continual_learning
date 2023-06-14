import os
import torch
import torch.optim.lr_scheduler

from PIL import Image


def wrap_in_pipeline(model, scheduler, pipeline_class, num_inference_steps: int, eta: float = 1.0, output_type: str = "torch"):
    def generate(batch_size: int) -> torch.Tensor:
        device = next(model.parameters()).device
        pipeline = pipeline_class(unet=model, scheduler=scheduler)
        pipeline.set_progress_bar_config(disable=True)
        samples = pipeline(
            batch_size, 
            num_inference_steps=num_inference_steps,
            eta=eta,
            output_type=output_type, 
        )
        return samples
    
    model.generate = generate


def make_grid(images, rows, cols):
    w, h = images[0].size
    grid = Image.new("RGB", size=(cols * w, rows * h))
    for i, image in enumerate(images):
        grid.paste(image, box=(i % cols * w, i // cols * h))
    return grid


def evaluate_diffusion(output_dir, eval_batch_size, epoch, pipeline, steps: int = 10, seed: int = 42, eta: float = 1.0):
    # Sample some images from random noise (this is the backward diffusion process).
    # The default pipeline output type is `List[PIL.Image]`
    images = pipeline(
        batch_size=eval_batch_size,
        generator=torch.manual_seed(seed),
        num_inference_steps=steps, 
        eta=eta,
        output_type="pil",
    ).images

    # Make a grid out of the images
    image_grid = make_grid(images, rows=4, cols=5)

    # Save the images
    test_dir = os.path.join(output_dir, "samples")
    os.makedirs(test_dir, exist_ok=True)
    image_grid.save(f"{test_dir}/{epoch:04d}.png")