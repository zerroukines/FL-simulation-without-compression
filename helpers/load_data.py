import logging
import numpy as np
import torch
from torchvision import transforms
from torch.utils.data import DataLoader, TensorDataset
from flwr_datasets import FederatedDataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_data(data_sampling_percentage=0.5, client_id=1, total_clients=2):
    """Load federated dataset partition based on client ID for PyTorch.

    Args:
        data_sampling_percentage (float): Percentage of the dataset to use for training.
        client_id (int): Unique ID for the client (1-based).
        total_clients (int): Total number of clients.

    Returns:
        Tuple of DataLoader objects: `(train_loader, test_loader)`.
    """
    # Download and partition dataset
    fds = FederatedDataset(dataset="cifar10", partitioners={"train": total_clients})
    partition = fds.load_partition(client_id - 1, "train")  # 0-based indexing
    partition.set_format("numpy")

    # Split into train and test (80% train, 20% test)
    partition = partition.train_test_split(test_size=0.2, seed=42)
    x_train, y_train = partition["train"]["img"], partition["train"]["label"]
    x_test, y_test = partition["test"]["img"], partition["test"]["label"]

    # Apply data sampling
    num_samples = int(data_sampling_percentage * len(x_train))
    indices = np.random.choice(len(x_train), num_samples, replace=False)
    x_train, y_train = x_train[indices], y_train[indices]

    # Define transforms for vit_h_14
    transform = transforms.Compose([
        transforms.ToPILImage(),              # Convert NumPy array to PIL Image
        transforms.Resize((224, 224)),        # Resize to 224x224 for vit_h_14
        transforms.ToTensor(),                # Convert to tensor (0-1 range, CHW format)
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet norms
    ])

    # Convert NumPy arrays to PyTorch tensors and apply transforms
    def preprocess_data(images, labels):
        tensor_images = []
        for img in images:
            img_tensor = transform(img)  # Apply transform to each image
            tensor_images.append(img_tensor)
        return torch.stack(tensor_images), torch.tensor(labels, dtype=torch.long)

    x_train_tensor, y_train_tensor = preprocess_data(x_train, y_train)
    x_test_tensor, y_test_tensor = preprocess_data(x_test, y_test)

    # Create TensorDatasets
    train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(x_test_tensor, y_test_tensor)

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    return train_loader, test_loader

# Test the function
if __name__ == "__main__":
    train_loader, test_loader = load_data(client_id=1, total_clients=2)
    for images, labels in train_loader:
        logger.info(f"Batch shape: {images.shape}, Labels shape: {labels.shape}")
        break