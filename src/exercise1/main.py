import torch
from torch.utils.data.dataloader import DataLoader
from scipy.stats import multivariate_normal
from itertools import cycle
import torchvision
import numpy as np

DATA_DIM = 28*28
device = "cuda:0" if torch.cuda.is_available() else "cpu:0"

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


class NoiseDataset(torch.utils.data.Dataset):
    def __init__(self, mu=np.zeros(DATA_DIM), cov=np.diag(np.ones(DATA_DIM))):
        super().__init__()
        self.mu = mu
        self.cov = cov
        self.noise = torch.tensor(np.random.multivariate_normal(self.mu, self.cov, size=100000))

    def get_log_probs(self, batch):
        """
        Returns the log probabilities (from PDF) for observing each sample in batch, according to this noise distribution
        """
        return torch.tensor(multivariate_normal.logpdf(batch.cpu(), mean=self.mu, cov=self.cov), dtype=torch.float64, device=device)

    def __getitem__(self, index):
        return self.noise[index]

    def __len__(self):
        return len(self.noise)




class GaussianEBM(torch.nn.Module):
    def __init__(self, precision_matrix_init=torch.diag(torch.ones(28*28, dtype=torch.float64)), mask_precision_matrix=False):
        super().__init__()
        self.precision_matrix = torch.nn.Parameter(precision_matrix_init)
        sign, neg_logdet = np.linalg.slogdet(self.precision_matrix.cpu().detach().numpy())
        init_log_Z = 28*28/2*np.log(2*np.pi) - 1/2*neg_logdet
        self.log_Z = torch.nn.Parameter(torch.tensor(init_log_Z, dtype=torch.float64))
        self.mask = self.get_precision_matrix_mask()
        self.mask_precision_matrix = mask_precision_matrix

    def get_precision_matrix_mask(self):
        """
        Return mask for 4-neighboring pixels
        """
        m = np.zeros((28*28, 28*28), np.float64)
        def idx(y, x):
            return 28*y + x
        for y in range(28):
            for x in range(28):
                # Pixel (y, x) should correlate with its 4 neighbors only
                m[idx(y, x), idx(y, x)] = 1.0
                if x+1 < 28:
                    m[idx(y, x), idx(y, x+1)] = 1.0
                if x-1 >= 0:
                   m[idx(y, x), idx(y, x-1)] = 1.0
                if y+1 < 28:
                    m[idx(y, x), idx(y+1, x)] = 1.0
                if y-1 >= 0:
                    m[idx(y, x), idx(y-1, x)] = 1.0
        return torch.tensor(m, device=device)

    def forward(self, x):
        """
        x: Tensor of shape [batch_size, 784]

        returns: (
            float32 tensor [batch_size]: log probabilities
        )
        """
        masked_precision_matrix = self.precision_matrix * self.mask if self.mask_precision_matrix else self.precision_matrix
        x = x.unsqueeze(2)
        log_probs = -1/2 * x.permute(0, 2, 1) @ masked_precision_matrix @ x - self.log_Z
        return log_probs.squeeze(-1).squeeze(-1)


def train_nce(batch_size, eta, mnist_data_path, max_steps=np.inf, mask_precision_matrix=False):
    # Get the MNIST data
    mnist_train, _, _ = get_mnist_data(mnist_data_path, add_noise=True)
    print(f"Loaded {len(mnist_train)} training examples")

    # Create noise data source
    noise_dataset = NoiseDataset(cov=np.diag(mnist_train.var(axis=0)))

    mnist_dataloader = DataLoader(mnist_train, batch_size=int(batch_size*eta), shuffle=True)
    noise_dataloader = DataLoader(noise_dataset, batch_size=int(batch_size*(1-eta)), shuffle=True)

    # Construct the model
    precision_matrix_init = np.diag(1 / mnist_train.var(axis=0))
    model = GaussianEBM(precision_matrix_init=torch.tensor(precision_matrix_init), mask_precision_matrix=mask_precision_matrix)
    model.to(device)

    nu = (1-eta)/eta

    def nce_loss(mnist_batch, noise_batch):
        """
        Computes the NCE loss for a given batch of real data (mnist) and noise
        """
        a = model(mnist_batch)                             # a = log p_theta(xi)
        b = noise_dataset.get_log_probs(mnist_batch)       # b = log p_noise(xi)
        c = model(noise_batch)                             # c = log p_theta(xi_prime)
        d = noise_dataset.get_log_probs(noise_batch)       # d = log p_noise(xi_prime)

        print(f"a: {a.mean()}")
        print(f"b: {b.mean()}")
        print(f"c: {c.mean()}")
        print(f"d: {d.mean()}")

        # Only correct if nu = 1 !!
        loss_term_1 = (a - torch.logsumexp(torch.stack((a, b), dim=1), dim=1)).mean()
        loss_term_2 = (d - torch.logsumexp(torch.stack((c, d), dim=1), dim=1)).mean()
        print(f"Loss term 1: {loss_term_1}")
        print(f"Loss term 2: {loss_term_2}")

        loss = loss_term_1 + loss_term_2

        # Normal NCE loss should be maximized, torch by default minimizes
        loss = -loss

        return loss

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

    # Training loop
    losses = []
    for step, (mnist_batch, noise_batch) in enumerate(cycle(zip(mnist_dataloader, noise_dataloader))):
        if step > max_steps:
            break
        
        mnist_batch = mnist_batch.to(device)
        noise_batch = noise_batch.to(device)
        
        optimizer.zero_grad()
        
        loss = nce_loss(mnist_batch, noise_batch)
        loss.backward()
        
        loss_numpy = loss.cpu().detach().numpy()
        print(f"{step}: Loss: {loss_numpy}")
        losses.append(loss_numpy)
        print(f"\tlog_Z: {model.log_Z.cpu()}")
        sign, neg_logdet = np.linalg.slogdet(model.precision_matrix.cpu().detach().numpy())
        print(f"\tTrue log_Z: {28*28/2*np.log(2*np.pi) - 1/2*neg_logdet}")
        #print(f"\tGrad log_Z: {model.log_Z.grad.cpu()}")
        #print(f"\tPrecision matrix grad norm: {torch.norm(model.precision_matrix.grad).cpu()}")

        optimizer.step()
        # print()
    
    return model.precision_matrix, losses


def train_cnce(batch_size, mnist_data_path, max_steps=np.inf, mask_precision_matrix=False):
    # Get the MNIST data
    mnist_train, _, _ = get_mnist_data(mnist_data_path, add_noise=True)
    print(f"Loaded {len(mnist_train)} training examples")

    # Create noisy examples (add gaussian noise)
    noise_data = mnist_train + np.random.multivariate_normal(
        mean=np.zeros(DATA_DIM),
        cov=np.diag(mnist_train.var(axis=0) * 0.01),
        size=mnist_train.shape[0])

    mnist_dataloader = DataLoader(mnist_train, batch_size=batch_size)
    noise_dataloader = DataLoader(noise_data, batch_size=batch_size)

    # Construct the model
    model = GaussianEBM(precision_matrix_init=torch.tensor(np.diag(1 / mnist_train.var(axis=0) )), mask_precision_matrix=mask_precision_matrix)
    model.to(device)

    def cnce_loss(mnist_batch, noise_batch):
        a = model(mnist_batch)  # a = log p_theta(xi)
        b = model(noise_batch)  # b = log p_theta(xi_prime)

        stacked = torch.stack((a, b), dim=1)

        loss = a - torch.logsumexp(stacked, dim=1)
        loss = loss.mean()

        # Normal cNCE loss should be maximized, torch by default minimizes
        return -loss

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

    # Training loop
    losses = []
    for step, (mnist_batch, noise_batch) in enumerate(cycle(zip(mnist_dataloader, noise_dataloader))):
        if step > max_steps:
            break
        mnist_batch = mnist_batch.to(device)
        noise_batch = noise_batch.to(device)

        optimizer.zero_grad()
        
        loss = cnce_loss(mnist_batch, noise_batch)
        loss.backward()

        loss_numpy = loss.cpu().detach().numpy()
        print(f"{step}: Loss: {loss_numpy}")
        losses.append(loss_numpy)

        optimizer.step()
        # print()
    
    return model.precision_matrix, losses

if __name__ == "__main__":
    # train_nce(batch_size=100, eta=0.5, mnist_data_path="../data/")
    train_cnce(batch_size=100, mnist_data_path="../data/")