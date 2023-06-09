from datasets import load_dataset
from torchvision import transforms
from torch.utils.data import DataLoader


def create_dataloader(batch_size: int = 128, transform: transforms.Compose = None, classes: list = None):
    # load dataset from the hub
    dataset = load_dataset("mnist")

    # filter dataset
    if classes is not None:
        dataset = dataset.filter(lambda example: example["label"] in classes)

    if transform is None:
        transform = transforms.Compose([
                # transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                # transforms.Lambda(lambda t: (t * 2) - 1)
        ])

    def apply_transforms(examples):
        examples["pixel_values"] = [transform(image.convert("L")) for image in examples["image"]]
        del examples["image"]
        return examples

    transformed_dataset = dataset.with_transform(apply_transforms).remove_columns("label")

    # create dataloader
    dataloader = DataLoader(transformed_dataset["train"], batch_size=batch_size, shuffle=True)
    
    return dataloader