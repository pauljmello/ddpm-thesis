import os
import argparse

import torch, torchvision
import numpy as np

seed = 3407  # https://arxiv.org/abs/2109.08203
np.random.seed(seed)
torch.manual_seed(seed)

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.cuda.set_device(torch.cuda.current_device())


class BiVariateGaussianDatasetForMI(torch.utils.data.Dataset):

    def __init__(self, d, rho, N):
        super(BiVariateGaussianDatasetForMI, self).__init__()

        cov = torch.eye(2 * d)
        cov[d:2 * d, 0:d] = rho * torch.eye(d)
        cov[0:d, d:2 * d] = rho * torch.eye(d)
        f = torch.distributions.MultivariateNormal(torch.zeros(2 * d), cov)
        Z = f.sample((N,))
        self.X, self.Y = Z[:, :d], Z[:, d:2 * d]

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

    def getX(self):
        return self.X

    def getY(self):
        return self.Y

    def sample_batch(self, batch_size):
        index_joint = np.random.choice(range(self.__len__()), size=batch_size, replace=False)
        index_marginal = np.random.choice(range(self.__len__()), size=batch_size, replace=False)
        return self.X[index_joint], self.Y[index_joint], self.Y[index_marginal]

class MNISTForMI(torch.utils.data.Dataset):
    def __init__(self, download_data=True):
        super(MNISTForMI, self).__init__()

        if download_data:
            datasets_path = os.path.join(os.path.dirname(os.path.dirname(os.getcwd())), "datasets")
            os.makedirs(datasets_path, exist_ok=True)
            mnist_root = datasets_path
            dataset = torchvision.datasets.MNIST(root=mnist_root, train=True, download=True)
            X = dataset.data.float()
            self.X = ((X - X.mean()) / X.std()).unsqueeze(1)
            self.Y = dataset.targets.float()
        else:
            self.X = None
            self.Y = None

    def add_noise(self, image, noise_level=0.1):
        noise = torch.randn_like(image) * noise_level
        noisy_image = image + noise
        return torch.clamp(noisy_image, 0, 1)  # Clamp the values to be between 0 and 1

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

    def getX(self):
        return self.X

    def getY(self):
        return self.Y

    def sample_batch(self, batch_size):
        data_length = self.__len__()
        if batch_size > data_length:
            batch_size = data_length

        index_joint = np.random.choice(range(data_length), size=batch_size, replace=False)
        index_marginal = np.random.choice(range(data_length), size=batch_size, replace=False)

        X_joint = self.X[index_joint]
        scaling_factors = np.random.uniform(0.01, 0.99, size=batch_size).reshape(-1, 1, 1, 1)
        noise = torch.randn_like(X_joint) * torch.tensor(scaling_factors, dtype=torch.float32)
        X_noisy = X_joint + noise

        scaling_factors_2 = np.random.uniform(0.01, 0.99, size=batch_size).reshape(-1, 1, 1, 1)
        noise_2 = torch.randn_like(X_joint) * torch.tensor(scaling_factors_2, dtype=torch.float32)
        X_joint = X_joint + noise_2

        X_concatenated = torch.cat((X_joint, X_noisy), dim=1)  # clean on left, noisy on right
        return X_concatenated, self.Y[index_joint], self.Y[index_marginal]

    # # Concatenated and NonConcatenated Images
    # def sample_batch(self, batch_size):
    #     data_length = self.__len__()
    #     if batch_size > data_length:
    #         batch_size = data_length
    #     index_joint = np.random.choice(range(data_length), size=batch_size, replace=False)
    #     index_marginal = np.random.choice(range(data_length), size=batch_size, replace=False)
    #
    #     X_joint = self.X[index_joint]
    #     #scaling_factor = np.random.uniform(0.10, 0.90)
    #     noise = torch.randn_like(X_joint) * 0.10
    #     X_noisy = X_joint + noise
    #     X_concatenated = torch.cat((X_joint, X_noisy), dim=1)  # clean on left, noisy on right
    #     if random.choice([True, False]):
    #         return X_concatenated, self.Y[index_joint], self.Y[index_marginal]
    #     else:
    #         return X_noisy, self.Y[index_joint], self.Y[index_marginal]


class FashionMNISTForMI(torch.utils.data.Dataset):
    def __init__(self, download_data=True):
        super(FashionMNISTForMI, self).__init__()

        if download_data:
            datasets_path = os.path.join(os.path.dirname(os.path.dirname(os.getcwd())), "datasets")
            os.makedirs(datasets_path, exist_ok=True)
            fashion_mnist_root = datasets_path
            dataset = torchvision.datasets.FashionMNIST(root=fashion_mnist_root, train=True, download=True)
            X = dataset.data.float()
            self.X = ((X - X.mean()) / X.std()).unsqueeze(1)
            self.Y = dataset.targets.float()
        else:
            self.X = None
            self.Y = None

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

    def getX(self):
        return self.X

    def getY(self):
        return self.Y

    def add_noise(self, image, noise_level=0.1):
        noise = torch.randn_like(image) * noise_level
        noisy_image = image + noise
        return torch.clamp(noisy_image, 0, 1)

    def sample_batch(self, batch_size):
        data_length = self.__len__()
        if batch_size > data_length:
            batch_size = data_length

        index_joint = np.random.choice(range(data_length), size=batch_size, replace=False)
        index_marginal = np.random.choice(range(data_length), size=batch_size, replace=False)

        X_joint = self.X[index_joint]
        scaling_factors = np.random.uniform(0.01, 0.99, size=batch_size).reshape(-1, 1, 1, 1)
        noise = torch.randn_like(X_joint) * torch.tensor(scaling_factors, dtype=torch.float32)
        X_noisy = X_joint + noise

        scaling_factors_2 = np.random.uniform(0.01, 0.99, size=batch_size).reshape(-1, 1, 1, 1)
        noise_2 = torch.randn_like(X_joint) * torch.tensor(scaling_factors_2, dtype=torch.float32)
        X_joint = X_joint + noise_2
        X_concatenated = torch.cat((X_joint, X_noisy), dim=1)  # clean on left, noisy on right
        return X_concatenated, self.Y[index_joint], self.Y[index_marginal]

class CIFAR10ForMI(torch.utils.data.Dataset):
    def __init__(self, download_data=True):
        super(CIFAR10ForMI, self).__init__()

        if download_data:
            datasets_path = os.path.join(os.path.dirname(os.path.dirname(os.getcwd())), "datasets")
            os.makedirs(datasets_path, exist_ok=True)
            cifar10_root = datasets_path
            dataset = torchvision.datasets.CIFAR10(root=cifar10_root, train=True, download=True)
            X = torch.tensor(dataset.data).float()

            self.X = X.permute(0, 3, 1, 2)  # Change shape to [N, C, H, W]
            self.X = (self.X - self.X.mean()) / self.X.std()
            self.Y = torch.tensor(dataset.targets).float()
        else:
            self.X = None
            self.Y = None

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

    def getX(self):
        return self.X

    def getY(self):
        return self.Y

    def add_noise(self, image, noise_level=0.1):
        noise = torch.randn_like(image) * noise_level
        noisy_image = image + noise
        return torch.clamp(noisy_image, 0, 1)

    def sample_batch(self, batch_size):
        data_length = self.__len__()
        if batch_size > data_length:
            batch_size = data_length

        index_joint = np.random.choice(range(data_length), size=batch_size, replace=False)
        index_marginal = np.random.choice(range(data_length), size=batch_size, replace=False)

        X_joint = self.X[index_joint]
        scaling_factors = np.random.uniform(0.01, 0.99, size=batch_size).reshape(-1, 1, 1, 1)
        noise = torch.randn_like(X_joint) * torch.tensor(scaling_factors, dtype=torch.float32)
        X_noisy = X_joint + noise

        scaling_factors_2 = np.random.uniform(0.01, 0.99, size=batch_size).reshape(-1, 1, 1, 1)
        noise_2 = torch.randn_like(X_joint) * torch.tensor(scaling_factors_2, dtype=torch.float32)
        X_noisy = X_noisy + noise_2
        X_concatenated = torch.cat((X_joint, X_noisy), dim=1)  # clean on left, noisy on right
        return X_concatenated, self.Y[index_joint], self.Y[index_marginal]



class CelebAForMI(torch.utils.data.Dataset):
    def __init__(self, download_data=True):
        super(CelebAForMI, self).__init__()

        if download_data:
            datasets_path = os.path.join(os.path.dirname(os.path.dirname(os.getcwd())), "datasets")
            os.makedirs(datasets_path, exist_ok=True)
            celebA_root = datasets_path
            dataset = torchvision.datasets.CelebA(root=celebA_root, split="train", download=True)

            X = dataset.data.float()

            self.X = X.permute(0, 3, 1, 2)  # Change shape to [N, C, H, W]
            self.X = (self.X - self.X.mean()) / self.X.std()
            self.Y = torch.tensor(dataset.targets).float()
        else:
            self.X = None
            self.Y = None

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

    def getX(self):
        return self.X

    def getY(self):
        return self.Y

    def add_noise(self, image, noise_level=0.1):
        noise = torch.randn_like(image) * noise_level
        noisy_image = image + noise
        return torch.clamp(noisy_image, 0, 1)

    def sample_batch(self, batch_size):
        data_length = self.__len__()
        if batch_size > data_length:
            batch_size = data_length

        index_joint = np.random.choice(range(data_length), size=batch_size, replace=False)
        index_marginal = np.random.choice(range(data_length), size=batch_size, replace=False)

        X_joint = self.X[index_joint]
        scaling_factors = np.random.uniform(0.01, 0.90, size=batch_size).reshape(-1, 1, 1, 1)
        noise = torch.randn_like(X_joint) * torch.tensor(scaling_factors, dtype=torch.float32)
        X_noisy = X_joint + noise
        X_concatenated = torch.cat((X_joint, X_noisy), dim=1)  # Clean on left, noisy on right
        return X_concatenated, self.Y[index_joint], self.Y[index_marginal]


def weight_init(m):
    if isinstance(m, torch.nn.Conv2d):
        torch.nn.init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            torch.nn.init.normal_(m.bias.data)
    elif isinstance(m, torch.nn.ConvTranspose2d):
        torch.nn.init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            torch.nn.init.normal_(m.bias.data)
    elif isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_normal_(m.weight.data)
        torch.nn.init.zeros_(m.bias.data)


def get_args():
    loadData = "gaussian"  # Working: "mnist", "gaussian", "cifar10", "fashion-mnist"    | Not Working: "celeba"
    GaussianDimensionality = 2
    batchSize = 512
    learingRate = 0.0001  # MNIST and Fashion-MNIST: 0.0001 | CIFAR-10: 0.0001
    epochs = 10000  # 1000000 for MNIST and Fashion-MNIST and CIFAR-10 | 250000 for gaussian
    saveMI = 500
    epochAvgRateMI = 500
    sampleMIRate = epochs / (epochs / 500)
    n_rhos = 10

    parser = argparse.ArgumentParser(description="Run the experiments of MINE", formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--figs_dir", default="../images/", help="folder to output the resulting images")

    parser.add_argument("--n_iterations", type=int, default=epochs, help="number of training epochs")

    parser.add_argument("--batch_size", type=int, default=batchSize, help="mini-batch size for the SGD")

    parser.add_argument("--learning_rate", type=float, default=learingRate, help="initial learning rate")

    parser.add_argument("--n_verbose", type=int, default=saveMI, help="number of iterations for showing the current MI, if -1, then never")

    parser.add_argument("--n_window", type=int, default=epochAvgRateMI, help="number of iterations taken into consideration for the averaging the MI (moving average)")

    parser.add_argument("--save_progress", type=int, default=sampleMIRate, help="sampling rate of the MI, if -1, nothing is saved")

    parser.add_argument("--d", type=int, default=GaussianDimensionality, help="dimensionality of the Gaussians in the example")

    parser.add_argument("--n_rhos", type=int, default=n_rhos, help="number of rhos for the Gaussian experiment")

    parser.add_argument("--example", choices=["gaussian", "mnist", "cifar10", "fashion-mnist", "celeba"], default=loadData, help="example to run")

    return parser.parse_args()
