import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from copy import deepcopy
from tqdm import tqdm
from torchvision.datasets import FashionMNIST, CIFAR10  
from torch.utils.data import DataLoader
from torchvision.models import resnet18

class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean
        
    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean
    
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


def get_data_loaders(batch_size=64, dataset='FashionMNIST'):
    transform_train = transforms.Compose([
        transforms.Resize(32),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
        transforms.Normalize((0.5,), (0.5,)),
        AddGaussianNoise(0., 0.1),
    ])

    transform_test = transforms.Compose([
        transforms.Resize(32),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
        transforms.Normalize((0.5,), (0.5,))
    ])

    if dataset == 'FashionMNIST':
        transform_train = transforms.Compose([
            transforms.Resize(32),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
            transforms.Normalize((0.5,), (0.5,)),
            AddGaussianNoise(0., 0.1),
        ])

        transform_test = transforms.Compose([
            transforms.Resize(32),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
            transforms.Normalize((0.5,), (0.5,))
        ])
        train_dataset = FashionMNIST(root='./data', train=True, transform=transform_train, download=True)
        train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [50000, 10000])
        test_dataset = FashionMNIST(root='./data', train=False, transform=transform_test, download=True)
    elif dataset == 'CIFAR10':
        transform_train = transforms.Compose([
            transforms.Resize(32),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            AddGaussianNoise(0., 0.1),
        ])

        transform_test = transforms.Compose([
            transforms.Resize(32),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        train_dataset = CIFAR10(root='./data', train=True, transform=transform_train, download=True)
        train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [45000, 5000])
        test_dataset = CIFAR10(root='./data', train=False, transform=transform_test, download=True)
    else:
        raise ValueError(f"Dataset {dataset} not supported.")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    return train_loader, val_loader, test_loader


def get_model(num_classes=10):
    model = resnet18(pretrained='imagenet') 
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def train_model(model, train_loader, val_loader, num_epochs=10, learning_rate=0.0001, device='cuda'):
    best_model = None
    best_accuracy = 0.0
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    model.to(device)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        accuracy = 0.0
        bar = tqdm(train_loader)
        for inputs, labels in bar:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            accuracy += (outputs.argmax(1) == labels).float().mean()
            bar.set_description(f"Loss: {loss.item():.4f}, Accuracy: {accuracy.item() / (bar.n + 1):.4f}")

        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {running_loss / len(train_loader)}, Accuracy: {accuracy / len(train_loader)}")

        # Evaluation
        model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, labels in tqdm(val_loader):
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_accuracy = 100 * correct / total

        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            best_model = deepcopy(model)

        print(f"Validation Accuracy: {val_accuracy:.2f}%")

    return best_model


def evaluate_model(model, test_loader, device='cuda'):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in tqdm(test_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f"Test Accuracy: {100 * correct / total:.2f}%")


def save_model(model, save_folder='./results/cnn_fmnist'):
    os.makedirs(save_folder, exist_ok=True)
    save_path = os.path.join(save_folder, 'resnet.pth')
    torch.save(model.state_dict(), save_path)
    print(f"Model saved at {save_path}")


def main():
    parser = argparse.ArgumentParser(description='Train and save a ResNet model on the given dataset.')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for data loaders')
    parser.add_argument('--num_epochs', type=int, default=100, help='Number of epochs for training')
    parser.add_argument('--learning_rate', type=float, default=0.0005, help='Learning rate for the optimizer')
    parser.add_argument('--dataset', type=str, default='FashionMNIST', help='Dataset to use for training')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use for training (cuda or cpu)')
    parser.add_argument('--output_path', type=str, default='./results/cnn_fmnist', help='Path to save the model')
    args = parser.parse_args()

    train_loader, val_loader, test_loader = get_data_loaders(batch_size=args.batch_size, dataset=args.dataset)
    model = get_model()
    model = train_model(model, train_loader, val_loader, num_epochs=args.num_epochs,
                        learning_rate=args.learning_rate, device=args.device)
    evaluate_model(model, test_loader, device=args.device)
    save_model(model, save_folder=args.output_path)

if __name__ == "__main__":
    main()
