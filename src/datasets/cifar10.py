from datasets import load_dataset
from torchvision import transforms
from torch.utils.data import DataLoader


def create_dataloader(batch_size: int = 128, 
                      train_transform: transforms.Compose = None,
                      test_transform: transforms.Compose = None,
                      classes: list = None
    ):
    # load dataset from the hub
    train_dataset = load_dataset("cifar10", split="train")
    test_dataset = load_dataset("cifar10", split="test")

    # filter dataset
    if classes is not None:
        train_dataset = train_dataset.filter(lambda example: example["label"] in classes)
        test_dataset = test_dataset.filter(lambda example: example["label"] in classes)

    if train_transform is None:
        train_transform = transforms.Compose([
                transforms.ToTensor(),
        ])
    if test_transform is None:
        test_transform = transforms.Compose([
                transforms.ToTensor(),
        ])

    def apply_train_transforms(examples):
        examples["pixel_values"] = [train_transform(image) for image in examples["img"]]
        del examples["img"]
        return examples
    
    def apply_test_transforms(examples):
        examples["pixel_values"] = [test_transform(image) for image in examples["img"]]
        del examples["img"]
        return examples

    transformed_train_dataset = train_dataset.with_transform(apply_train_transforms)
    transformed_test_dataset = test_dataset.with_transform(apply_test_transforms)

    # create dataloader
    train_dataloader = DataLoader(transformed_train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(transformed_test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_dataloader, test_dataloader