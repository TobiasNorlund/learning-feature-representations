import torch
from torch.utils.data.dataloader import DataLoader
from scipy.stats import multivariate_normal
import torchvision
import numpy as np

DATA_DIM = 28*28


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

    if standardize:
        # Subtract the mean intensity of each pixel
        X_mean = X_train.mean(axis=0)
        X_train -= X_mean
        X_test -= X_mean

    if add_noise:
        # Add some random gaussian noise (for stability?)
        X_train += np.random.normal(0, 1/100, size=X_train.shape)
        X_test += np.random.normal(0, 1/100, size=X_test.shape)

    return X_train, X_test


class NoiseDataset(torch.utils.data.IterableDataset):
    def __init__(self, mu=np.zeros(DATA_DIM), cov=np.diag(np.ones(DATA_DIM))):
        super(NoiseDataset, self).__init__()
        self.mu = mu
        self.cov = cov

    def get_log_probs(self, batch):
        """
        Returns the log probabilities (from PDF) for observing each sample in batch, according to this noise distribution
        """
        return torch.tensor(multivariate_normal.logpdf(batch, mean=self.mu, cov=self.cov), dtype=torch.float64)

    def __iter__(self):
        while True:
            yield np.random.multivariate_normal(self.mu, self.cov)


class GaussianEBM(torch.nn.Module):
    def __init__(self, precision_matrix_init=torch.diag(torch.ones(28*28, dtype=torch.float64))):
        super(GaussianEBM, self).__init__()
        self.precision_matrix = torch.nn.Parameter(precision_matrix_init)  # TODO: Should apply mask
        self.log_Z = torch.nn.Parameter(torch.tensor(-1000.0, dtype=torch.float64))

    def forward(self, x):
        """
        x: Tensor of shape [batch_size, 784]

        returns: (
            float32 tensor [batch_size]: log probabilities
        )
        """
        x = x.unsqueeze(2)
        log_probs = -1/2 * x.permute(0, 2, 1) @ self.precision_matrix @ x - self.log_Z
        return log_probs.squeeze(-1).squeeze(-1)


def train_nce(batch_size, eta):
    # Get the MNIST data
    mnist_train, _ = get_mnist_data("../data/", add_noise=True)
    print(f"Loaded {len(mnist_train)} training examples")

    # Create noise data source
    noise_dataset = NoiseDataset(cov=np.diag(mnist_train.var(axis=0) * 2 )) #cov=np.diag(np.ones(DATA_DIM) * 1/100))

    mnist_dataloader = DataLoader(mnist_train, batch_size=int(batch_size*eta), shuffle=True)
    noise_dataloader = DataLoader(noise_dataset, batch_size=int(batch_size*(1-eta)))

    # Construct the model
    model = GaussianEBM(precision_matrix_init=torch.tensor(np.diag(1 / (mnist_train.var(axis=0) * 2) )))

    nu = (1-eta)/eta

    def nce_loss(mnist_batch, noise_batch):
        """
        Computes the NCE loss for a given batch of real data (mnist) and noise
        """
        a = model(mnist_batch)                             # a = log p_theta(xi)
        b = noise_dataset.get_log_probs(mnist_batch)       # b = log p_noise(xi)
        c = model(noise_batch)                             # c = log p_theta(xi_prime)
        d = noise_dataset.get_log_probs(noise_batch)       # d = log p_noise(xi_prime)

        loss = (a - b - torch.log(torch.exp(a-b) + nu)).mean() + \
            (d - c - torch.log(1 + nu*torch.exp(d-c))).mean()

        # Normal NCE loss should be maximized, torch by default minimizes
        loss = -loss

        return loss

    optimizer = torch.optim.SGD(model.parameters(), lr=1.0)

    # Training loop
    for step, (mnist_batch, noise_batch) in enumerate(zip(mnist_dataloader, noise_dataloader)):
        optimizer.zero_grad()
        
        loss = nce_loss(mnist_batch, noise_batch)
        
        loss.backward()
        print(f"Loss: {loss.detach().numpy()}")
        print(f"log_Z: {model.log_Z}")
        print(f"Grad log_Z: {model.log_Z.grad}")

        optimizer.step()
        print()


if __name__ == "__main__":
    train_nce(batch_size=100, eta=0.5)