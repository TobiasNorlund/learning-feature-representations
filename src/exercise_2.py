import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy.stats import multivariate_normal
import torch
from itertools import cycle
from torch.utils.data.dataloader import DataLoader
import torchvision


IMAGE_SIZE = 28, 28
DATA_DIM = 28 * 28
K = 64
device = "cuda:0" if torch.cuda.is_available() else "cpu:0"


def get_cifar10_data(path, subtract_mean: bool=True, whiten=False):
    train_data = torchvision.datasets.CIFAR10(path,  train=True, download=True)
    test_data = torchvision.datasets.CIFAR10(path, train=False, download=True)

    def transform(images: np.array) -> np.array:
        """Resizes to IMAGE_SIZE, converts to greyscale, and optionally
        whitens the images."""
        images = [Image.fromarray(x) for x in images]  # to PIL
        images = [x.resize(IMAGE_SIZE) for x in images]  # resize to IMAGE_SIZE
        images = [x.convert('L') for x in images]  # to greyscale
        images = np.array([np.array(x) for x in images])  # back to numpy 2D array
        if whiten:
            pass  # TODO
        return images

    X_train = transform(train_data.data)
    X_test = transform(test_data.data)
    X_train = X_train.reshape(X_train.shape[0], -1) / 255.
    X_test = X_test.reshape(X_test.shape[0], -1) / 255.

    X_mean = X_train.mean(axis=0)
    if subtract_mean:
        X_train -= X_mean
        X_test -= X_mean

    return X_train, X_test, X_mean


def get_mnist_data(path, standardize=True, add_noise: bool=False):
    """
    Downloads, and pre-processes the MNIST data
    """
    # Download
    X_train = torchvision.datasets.MNIST(path, train=True, transform=None, target_transform=None, download=True)
    X_test = torchvision.datasets.MNIST(path, train=False, transform=None, target_transform=None, download=True)

    # Flatten and normalize
    X_train = X_train.data.numpy().reshape(X_train.data.shape[0], -1) / 255.
    X_test = X_test.data.numpy().reshape(X_test.data.shape[0], -1) / 255.

    assert X_train.shape[1] == DATA_DIM, "DATA_DIM didn't match the actual data"
    assert X_test.shape[1] == DATA_DIM, "DATA_DIM didn't match the actual data"

    X_mean = X_train.mean(axis=0)
    if standardize:
        # Subtract the mean intensity of each pixel
        X_train -= X_mean
        X_test -= X_mean

    if add_noise:
        # Add some random gaussian noise (for stability)
        X_train += np.random.normal(0, 1/100, size=X_train.shape)
        X_test += np.random.normal(0, 1/100, size=X_test.shape)

    return X_train, X_test, X_mean


class DEM(torch.torch.nn.Module):
    def __init__(self, sigma):
        super().__init__()
        self.sigma = sigma
        self.V = torch.nn.Parameter(torch.randn(K, DATA_DIM).double())
        self.W = torch.nn.Parameter(torch.randn(K, K).double())
        self.b = torch.nn.Parameter(torch.zeros(DATA_DIM).double())
        self.c = torch.nn.Parameter(torch.zeros(K).double())
        self.softplus = torch.nn.Softplus()
    
    def forward(self, x):
        g_theta = torch.sigmoid(x @ self.V.T)
        h_theta = self.softplus(g_theta @ self.W + self.c).sum(-1)
        # p_theta = -1/(2 * self.sigma**2) * torch.norm(x, dim=1)**2 + x @ self.b + h_theta
        p_theta = x @ self.b + h_theta
        return p_theta


def train_cnce(batch_size, cifar10_data_path, sigma=1.0, noise_multiplier=0.1, max_steps=np.inf):
    # Get the CIFAR10 data
    cifar_train, cifar_test, _ = get_cifar10_data(cifar10_data_path)
    print(f"Loaded {len(cifar_train)} training examples")

    # Create noisy examples (add gaussian noise)
    noise_train = cifar_train + np.random.normal(size=cifar_train.shape) * noise_multiplier
    noise_train /= (np.linalg.norm(noise_train, axis=1) / np.linalg.norm(cifar_train, axis=1))[:, np.newaxis]

    noise_test = cifar_test + np.random.normal(size=cifar_test.shape) * noise_multiplier
    noise_test /= (np.linalg.norm(noise_test, axis=1) / np.linalg.norm(cifar_test, axis=1))[:, np.newaxis]

    cifar_train_dataloader = DataLoader(cifar_train, batch_size=batch_size)
    noise_train_dataloader = DataLoader(noise_train, batch_size=batch_size)
    cifar_test_dataloader = DataLoader(cifar_test, batch_size=batch_size)
    noise_test_dataloader = DataLoader(noise_test, batch_size=batch_size)

    # Construct the model
    model = DEM(sigma=sigma)
    model.to(device)

    def cnce_loss(cifar_batch, noise_batch):
        a = model(cifar_batch)  # a = log p_theta(xi)
        b = model(noise_batch)  # b = log p_theta(xi_prime)

        stacked = torch.stack((a, b), dim=1)

        loss = a - torch.logsumexp(stacked, dim=1)
        loss = loss.mean()

        # Normal cNCE loss should be maximized, torch by default minimizes
        return -loss

    def evaluate():
        losses = []
        for step, (cifar_batch, noise_batch) in enumerate(zip(cifar_test_dataloader, noise_test_dataloader)):
            cifar_batch = cifar_batch.to(device)
            noise_batch = noise_batch.to(device)
            losses.append(cnce_loss(cifar_batch, noise_batch).cpu().detach().numpy())
        return np.array(losses)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Training loop
    losses = []
    for step, (cifar_batch, noise_batch) in enumerate(cycle(zip(cifar_train_dataloader, noise_train_dataloader))):
        if step > max_steps:
            break
        cifar_batch = cifar_batch.to(device)
        noise_batch = noise_batch.to(device)

        optimizer.zero_grad()
        
        loss = cnce_loss(cifar_batch, noise_batch)
        loss.backward()
        optimizer.step()

        loss_numpy = loss.cpu().detach().numpy()
        losses.append(loss_numpy)
        if step % 100 == 0:
            print(f"{step}: Loss: {loss_numpy}")
        
        if step % 1000 == 0:
            print("Evaluating...")
            eval_losses = evaluate()
            print(f"\tloss avg: {eval_losses.mean()}\tloss std: {eval_losses.std()}")
        
    
    return model, losses


if __name__ == '__main__':
    train_cnce(batch_size=1000, sigma=1.0, cifar10_data_path="../data/")
