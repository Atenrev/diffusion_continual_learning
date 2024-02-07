import os
import torch
import torchvision
import torchvision.transforms as transforms
import random


if __name__ == "__main__":
    # Define the path where the examples will be saved
    output_folder = "examples_cifar10"
    os.makedirs(output_folder, exist_ok=True)

    # Define the Fashion MNIST dataset and dataloader
    transform = transforms.Compose([transforms.ToTensor()])
    dataset = torchvision.datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)

    # Define class names for Fashion MNIST 
    class_names = [
        "airplane",
        "automobile",
        "bird",
        "cat",
        "deer",
        "dog",
        "frog",
        "horse",
        "ship",
        "truck",
    ]
    # Task orders: {3, 7}, {8, 0}, {1, 4}, {2, 6}, {9, 5}
    task_orders = [1, 2, 3, 0, 2, 4, 3, 0, 1, 4]

    # Initialize counters for each class
    class_counters = {class_name: 0 for class_name in class_names}
    num_samples_per_class = 1  # Number of samples to save per class

    # Loop through the dataset and save random samples for each class
    for image, label in dataloader:
        class_name = class_names[label]
        task_order = task_orders[label]

        if class_counters[class_name] < num_samples_per_class:
            # Generate a random file name
            random_suffix = random.randint(0, 99999)
            file_name = f"{class_name.replace('/', '_')}_{class_counters[class_name]}_task_{task_order}_{random_suffix}.png"
            file_path = os.path.join(output_folder, file_name)

            # Save the image
            torchvision.utils.save_image(image, file_path)

            # Increment the counter for the class
            class_counters[class_name] += 1

        # Check if we have saved enough samples for all classes
        if all(count >= num_samples_per_class for count in class_counters.values()):
            break

    print("Random samples saved to the 'examples_cifar10' folder.")
