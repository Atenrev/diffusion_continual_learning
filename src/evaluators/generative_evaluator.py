import os
import torch

from PIL import Image
from tqdm import tqdm
from torchmetrics.image.fid import FrechetInceptionDistance

from src.common.diffusion_utils import make_grid
from src.evaluators.base_evaluator import BaseEvaluator


class GenerativeModelEvaluator(BaseEvaluator):
    def __init__(self, device: str = "cuda"):
        self.device = device
        self.reset_fid()

    def evaluate(self, model, dataloader, epoch: int = 0, save_images=20, save_path="results"):
        print("Evaluating FID...")
        self.fid.reset()

        for batch in tqdm(dataloader):
            batch = batch["pixel_values"].to(self.device)
            pred = model.generate(batch.shape[0])

            if batch.shape[1] == 1:
                batch = torch.cat([batch] * 3, dim=1)
                pred = torch.cat([pred] * 3, dim=1)

            self.fid.update(pred, real=False)
            self.fid.update(batch, real=True)

        fid = self.fid.compute().cpu().detach().item()
        print(f"Evaluation complete. FID: {fid}")

        if save_images > 0:
            generated_images = model.generate(save_images)
            # To PIL image
            generated_images = generated_images.mul(255).to(torch.uint8)
            generated_images = generated_images.permute(0, 2, 3, 1).cpu().numpy()
            generated_images = [Image.fromarray(img.squeeze()) for img in generated_images]
            nrows = int(save_images**0.5)
            ncols = save_images // nrows + save_images % nrows
            generated_images = make_grid(generated_images, rows=nrows, cols=ncols)
            generated_images.save(os.path.join(save_path, f"generated_images_{epoch}.png"))

        return fid

    def reset_fid(self):
        self.fid = FrechetInceptionDistance(normalize=True, feature=64)

        if self.device == "cuda":
            self.fid.cuda()