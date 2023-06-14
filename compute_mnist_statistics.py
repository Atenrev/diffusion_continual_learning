import os
import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from typing import Any, Optional
from avalanche.models import SimpleMLP
from torchvision import transforms

from src.common.utils import get_configuration
from src.datasets.mnist import create_dataloader
from src.pipelines.pipeline_ddim import DDIMPipeline


def __parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_config_path", type=str, default="configs/model/mlp.json")
    parser.add_argument("--weights_path", type=str, default="results/mlp_mnist/")
    parser.add_argument("--generator_path", type=str, default="results/diffusion_None_mse_42/")

    parser.add_argument("--classifier_batch_size", type=int, default=256)
    parser.add_argument("--generator_batch_size", type=int, default=256)

    parser.add_argument("--n_samples", type=int, default=1000)
    parser.add_argument("--n_steps", type=int, default=20)
    parser.add_argument("--eta", type=float, default=0.0)
    parser.add_argument("--device", type=str, default="cuda")

    return parser.parse_args()


def load_or_train_mnist_classifier(model_config: Any, device: str, weights_path: str, batch_size: int = 256):
    model = SimpleMLP(
        input_size=model_config.model.input_size *
        model_config.model.input_size * model_config.model.channels,
        num_classes=model_config.model.n_classes
    )

    if os.path.exists(os.path.join(weights_path, "model.pt")):
        print("Loading model from disk")
        model.load_state_dict(torch.load(os.path.join(weights_path, "model.pt")))
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

    preprocess = transforms.Compose(
        [
            transforms.Resize((28, 28)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )

    train_loader, test_loader = create_dataloader(batch_size, preprocess)

    for epoch in range(10):
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

        print(f"Epoch {epoch} finished. Accuracy: {sum(accuracy_list) / len(accuracy_list)}")

    
    torch.save(model.state_dict(), os.path.join(weights_path, "model.pt"))
    return model


def main(args):
    model_config = get_configuration(args.model_config_path)
    classifier = load_or_train_mnist_classifier(model_config, args.device, args.weights_path, args.classifier_batch_size)
    generator_pipeline = DDIMPipeline.from_pretrained(args.generator_path)
    generator_pipeline.set_progress_bar_config(disable=True)
    generator_pipeline = generator_pipeline.to(args.device)
    
    # initializes dict with the 10 classes to 0
    samples_per_class = {i: 0 for i in range(10)}
    n_iterations = args.n_samples // args.generator_batch_size
    for _ in tqdm(range(n_iterations)):
        generated_samples = generator_pipeline(
            args.generator_batch_size, 
            num_inference_steps=args.n_steps,
            eta=args.eta,
            output_type="torch", 
        )

        # Resize to 28x28
        generated_samples = torch.nn.functional.interpolate(
            generated_samples, size=(28, 28), mode="bilinear", align_corners=False
        )

        classes = classifier(generated_samples)
        classes = torch.argmax(classes, dim=1)
        classes = classes.cpu().numpy()

        for c in classes:
            samples_per_class[c] += 1

    # Extract class names and sample counts
    class_names = list(samples_per_class.keys())
    sample_counts = list(samples_per_class.values())

    # Set up the figure and axis
    fig, ax = plt.subplots()

    # Plotting the bar graph
    ax.bar(class_names, sample_counts, color='skyblue')

    # Adding labels and title
    ax.set_xlabel('Classes')
    ax.set_ylabel('Number of Samples')
    ax.set_title('Number of Samples for Each Class')

    # Adjusting the appearance
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Rotating the x-axis labels if necessary
    plt.xticks(rotation=45)

    # Setting the x-axis tick positions and labels
    ax.set_xticks(np.arange(0, len(class_names)))
    ax.set_xticklabels(class_names)

    # Save the graph to disk
    plt.tight_layout()
    plt.savefig('mnist_statistics.png')


if __name__ == "__main__":
    args = __parse_args()
    main(args)
