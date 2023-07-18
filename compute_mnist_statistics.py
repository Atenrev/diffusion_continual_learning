import os
import torch
import argparse
import numpy as np

from tqdm import tqdm
from typing import Any
from src.models.simple_cnn import SimpleCNN
from torchvision import transforms

from src.common.utils import get_configuration
from src.datasets.fashion_mnist import create_dataloader
from src.pipelines.pipeline_ddim import DDIMPipeline
from src.common.visual import plot_bar


preprocess = transforms.Compose(
        [
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )


def __parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_config_path", type=str,
                        default="configs/model/cnn.json")
    parser.add_argument("--weights_path", type=str,
                        default="results/cnn_fmnist/")
    parser.add_argument("--generator_path", type=str,
                        default="results/fashion_mnist/diffusion/generation/ddim_medium_mse_42")

    parser.add_argument("--classifier_batch_size", type=int, default=256)
    parser.add_argument("--generator_batch_size", type=int, default=128)

    parser.add_argument("--n_samples", type=int, default=10000)
    parser.add_argument("--n_steps", type=int, default=20)
    parser.add_argument("--eta", type=float, default=0.0)
    parser.add_argument("--device", type=str, default="cuda")

    return parser.parse_args()


def load_or_train_mnist_classifier(model_config: Any, device: str, weights_path: str, batch_size: int = 256):
    model = SimpleCNN(
        n_channels=model_config.model.channels,
        num_classes=model_config.model.n_classes
    )

    if os.path.exists(os.path.join(weights_path, "model.pt")):
        print("Loading model from disk")
        model.load_state_dict(torch.load(
            os.path.join(weights_path, "model.pt")))
        model.to(device)
        return model

    print("Model not found, training from scratch")
    os.makedirs(weights_path, exist_ok=True)
    model = model.to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=model_config.optimizer.lr,
    )
    criterion = torch.nn.CrossEntropyLoss()

    train_loader, test_loader = create_dataloader(batch_size, preprocess)

    for epoch in range(20):
        print(f"Epoch {epoch}")

        for batch in tqdm(train_loader):
            optimizer.zero_grad()

            batch_data = batch["pixel_values"].to(device)
            batch_labels = batch["label"].to(device)
            pred = model(batch_data)
            loss = criterion(pred, batch_labels)

            loss.backward()
            optimizer.step()

        print("Evaluating model")

        accuracy_list = []
        for batch in tqdm(test_loader):
            with torch.no_grad():
                batch_data = batch["pixel_values"].to(device)
                batch_labels = batch["label"].to(device)
                pred = model(batch_data)
                accuracy = (pred.argmax(dim=1) == batch_labels).float().mean()
                accuracy_list.append(accuracy.item())

        print(
            f"Epoch {epoch} finished. Accuracy: {sum(accuracy_list) / len(accuracy_list)}")

    torch.save(model.state_dict(), os.path.join(weights_path, "model.pt"))
    return model


def main(args):
    device = args.device
    model_config = get_configuration(args.model_config_path)
    classifier = load_or_train_mnist_classifier(
        model_config, args.device, args.weights_path, args.classifier_batch_size)
    classifier.eval()

    evaluator_classifier = SimpleCNN(
        n_channels=model_config.model.channels,
        num_classes=model_config.model.n_classes
    ).to(device)
    evaluator_optimizer = torch.optim.Adam(
        evaluator_classifier.parameters(),
        lr=model_config.optimizer.lr,
    )
    criterion = torch.nn.CrossEntropyLoss()
    _, test_loader = create_dataloader(args.classifier_batch_size, preprocess)

    generator_pipeline = DDIMPipeline.from_pretrained(args.generator_path)
    generator_pipeline.set_progress_bar_config(disable=True)
    generator_pipeline = generator_pipeline.to(args.device)

    # initializes dict with the 10 classes to 0
    samples_per_class = {i: 0 for i in range(10)}
    n_iterations = args.n_samples // args.generator_batch_size
    pbar = tqdm(range(n_iterations))
    for it in pbar:
        generated_samples = generator_pipeline(
            args.generator_batch_size,
            num_inference_steps=args.n_steps,
            eta=args.eta,
            output_type="torch_raw",
        )

        # Resize to 28x28
        # generated_samples = torch.nn.functional.interpolate(
        #     generated_samples, size=(28, 28), mode="bilinear", align_corners=False
        # )

        with torch.no_grad():
            classes = classifier(generated_samples)
        classes = torch.argmax(classes, dim=1)
        classes_np = classes.cpu().numpy()

        for c in classes_np:
            samples_per_class[c] += 1

        evaluator_optimizer.zero_grad()
        preds = evaluator_classifier(generated_samples)
        loss_evaluator = criterion(preds, classes)
        loss_evaluator.backward()
        evaluator_optimizer.step()
        pbar.set_description(
            f"Loss: {loss_evaluator.item():.4f}, Accuracy: {(preds.argmax(dim=1) == classes).float().mean().item():.4f}")

        if it % 50 == 0 and it > 0:
            print("Evaluating model")
            evaluator_classifier.eval()
            accuracy_list = []
            for batch in test_loader:
                with torch.no_grad():
                    batch_data = batch["pixel_values"].to(device)
                    batch_labels = batch["label"].to(device)
                    pred = evaluator_classifier(batch_data)
                    accuracy = (pred.argmax(dim=1) == batch_labels).float().mean()
                    accuracy_list.append(accuracy.item())
            print(f"Accuracy: {sum(accuracy_list) / len(accuracy_list)}")
            evaluator_classifier.train()

    # Extract class names and sample counts
    class_names = list(samples_per_class.keys())
    sample_counts = list(samples_per_class.values())

    # Plot bar chart
    save_path = os.path.join(args.generator_path, "mnist_samples_per_class.png")
    plot_bar(
        class_names,
        sample_counts,
        x_label="Classes",
        y_label="Number of samples",
        title="Number of samples for each class",
        save_path=save_path
    )


if __name__ == "__main__":
    args = __parse_args()
    main(args)
