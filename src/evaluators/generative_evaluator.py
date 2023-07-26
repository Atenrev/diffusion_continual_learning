import os
import torch

from PIL import Image
from tqdm import tqdm
from torchmetrics.image.fid import FrechetInceptionDistance

from src.common.diffusion_utils import make_grid
from src.evaluators.base_evaluator import BaseEvaluator


class GenerativeModelEvaluator(BaseEvaluator):
    def __init__(self, device: str = "cuda", save_path="results", save_images: int = 0, fid_feature_size: int = 2048):
        self.device = device
        self.save_path = save_path
        self.fid_feature_size = fid_feature_size
        self.save_images = save_images
        self.reset_fid()

    def evaluate_fid(self, model, dataloader, epoch: int = 0, fid_images: int = 10000, gensteps: int = 20) -> float:
        print("Evaluating FID...")
        self.fid.reset()

        print("Processing real images...")
        batch_size = 1

        for batch in tqdm(dataloader):
            batch = batch["pixel_values"].to(self.device)
            batch_size = max(batch_size, batch.shape[0])

            if batch.min() < 0:
                batch = (batch + 1) / 2
            
            if batch.shape[1] == 1:
                batch = torch.cat([batch] * 3, dim=1)

            self.fid.update(batch, real=True)

        print("Processing generated images...")
            
        if fid_images == 0:
            images_to_generate = len(dataloader.dataset)
        else:
            images_to_generate = fid_images

        bar = tqdm(total=images_to_generate // batch_size+1)
        while images_to_generate > batch_size:
            pred = model.generate(batch_size, generation_steps=gensteps, output_type="torch")

            if pred.shape[1] == 1:
                pred = torch.cat([pred] * 3, dim=1)

            self.fid.update(pred, real=False)
            images_to_generate -= batch_size
            bar.update(1)

        pred = model.generate(images_to_generate, generation_steps=gensteps, output_type="torch")

        if pred.shape[1] == 1:
                pred = torch.cat([pred] * 3, dim=1)

        self.fid.update(pred, real=False)
        bar.update(1)
        bar.close()

        fid = self.fid.compute().cpu().detach().item()
        print(f"Evaluation complete. FID: {fid}")

        if self.save_images > 0:
            generated_images = model.generate(self.save_images, generation_steps=gensteps, output_type="torch")
            # To PIL image
            generated_images = generated_images.mul(255).to(torch.uint8)
            generated_images = generated_images.permute(0, 2, 3, 1).cpu().numpy()
            generated_images = [Image.fromarray(img.squeeze()) for img in generated_images]
            nrows = int(self.save_images**0.5)
            ncols = self.save_images // nrows + self.save_images % nrows
            generated_images = make_grid(generated_images, rows=nrows, cols=ncols)
            out_dir = os.path.join(self.save_path, "samples")
            os.makedirs(out_dir, exist_ok=True)
            generated_images.save(os.path.join(out_dir, f"samples_epoch_{epoch}_gensteps_{gensteps}_fid_{fid:.4f}.png"))

        return fid
    
    def evaluate(self, model, dataloader, epoch: int = 0, fid_images: int = 10000, gensteps: int = 20, compute_auc: bool = True) -> dict:
        """
        Computes the FID score for the given model and dataloader.

        If compute_auc is True, then the FID score is computed for gensteps 2, 5, 10, 20 
        and the AUC is computed.

        Args:
            model (torch.nn.Module): The model to evaluate.
            dataloader (torch.utils.data.DataLoader): The dataloader to use for evaluation.
            epoch (int, optional): The epoch number. Defaults to 0.
            fid_images (int, optional): The number of images to use for FID computation. Defaults to 10000.
            gensteps (int, optional): The number of steps to use for generation. Defaults to 20.
            compute_auc (bool, optional): Whether to compute the AUC or not. Defaults to True.

        Returns:    
            dict: A dictionary containing the FID score and the AUC (if compute_auc is True).
        """
        if not compute_auc:
            fid = self.evaluate_fid(model, dataloader, epoch, fid_images, gensteps)
            return {"fid": fid}
        
        gensteps_list = [2, 5, 10, 20]
        fid_list = []
        for genstep in gensteps_list:
            fid = self.evaluate_fid(model, dataloader, epoch, fid_images, genstep)
            fid_list.append(fid)

        auc = torch.trapz(torch.asarray(fid_list), x=torch.asarray([2, 5, 10, 20])).item()
        results = {
            "auc": auc,
        }
        for i, genstep in enumerate(gensteps_list):
            results[f"fid_{genstep}"] = fid_list[i]

        return results
        

    def reset_fid(self):
        self.fid = FrechetInceptionDistance(normalize=True, feature=self.fid_feature_size)

        if self.device == "cuda":
            self.fid.cuda()