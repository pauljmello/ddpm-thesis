import os
import math

import torch
from torch import nn
import utils

seed = 3407  # https://arxiv.org/abs/2109.08203
torch.manual_seed(seed)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.cuda.set_device(torch.cuda.current_device())


class MLP(nn.Module):

    def __init__(self, dimX, dimY, hidden_size):
        super(MLP, self).__init__()

        self.f_theta = nn.Sequential(
            nn.Linear(dimX + dimY, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, X, Y):
        Z = torch.cat((X, Y), 1)
        return self.f_theta(Z)



# Works for Concatenated Images Only
class CNN(torch.nn.Module):

    def __init__(self, dimX, dimY):
        super(CNN, self).__init__()

        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(2, 5, 5, 1, padding=1),  # in_channels is now 2
            torch.nn.MaxPool2d(5, 2, 2),
            torch.nn.ReLU6(inplace=True),
        )
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv2d(5, 50, 5),
            torch.nn.MaxPool2d(5, 2, 2),
            torch.nn.ReLU6(inplace=True),
        )
        self.fc1 = torch.nn.Sequential(
            torch.nn.Linear(50 * 5 * 5 + 1, 100),
            torch.nn.ReLU6(inplace=True),
        )
        self.fc2 = torch.nn.Sequential(
            torch.nn.Linear(100 + 1, 50),
            torch.nn.ReLU6(inplace=True)
        )
        self.fc3 = torch.nn.Sequential(
            torch.nn.Linear(50 + 1, 1)
        )

    def forward(self, X, Y):
        Y = Y.view(-1, 1)
        Z = self.conv1(X)
        Z = self.conv2(Z)
        Z = Z.view(Z.size(0), -1)  # Flatten Z before passing to fully connected layer
        Z = self.fc1(torch.cat([Z, Y], 1))
        Z = self.fc2(torch.cat([Z, Y], 1))
        return self.fc3(torch.cat([Z, Y], 1))


# Dual Channel CNN Concatenated and Not Concatenated Image dimensions
# class CNN(nn.Module):
#
#     def __init__(self, dimX, dimY):
#         super(CNN, self).__init__()
#
#         # Make conv1 a list so we can dynamically assign it later
#         self.conv1 = []
#
#         self.conv2 = nn.Sequential(
#             nn.Conv2d(5, 50, 5),
#             nn.MaxPool2d(5, 2, 2),
#             nn.ReLU6(inplace=True),
#         )
#         self.fc1 = nn.Sequential(
#             nn.Linear(50 * 5 * 5 + 1, 100),
#             nn.ReLU6(inplace=True),
#         )
#         self.fc2 = nn.Sequential(
#             nn.Linear(100 + 1, 50),
#             nn.ReLU6(inplace=True)
#         )
#         self.fc3 = nn.Sequential(
#             nn.Linear(50 + 1, 1)
#         )
#
#     def forward(self, X, Y):
#         Y = Y.view(-1, 1)
#         # Check the number of channels
#         num_channels = X.size(1)
#
#         if len(self.conv1) == 0 or self.conv1[0].in_channels != num_channels:
#             self.conv1 = nn.Sequential(
#                 nn.Conv2d(num_channels, 5, 5, 1, padding=1),
#                 nn.MaxPool2d(5, 2, 2),
#                 nn.ReLU6(inplace=True),
#             )
#             self.conv1 = self.conv1.to(X.device)  # Make sure it's on the same device as X
#
#         # if statement to determine which hardcoded conv to use as first layer
#
#         Z = self.conv1(X)
#         Z = self.conv2(Z)
#         Z = Z.view(Z.size(0), -1)  # Flatten Z before passing to fully connected layer
#         Z = self.fc1(torch.cat([Z, Y], 1))
#         Z = self.fc2(torch.cat([Z, Y], 1))
#         return self.fc3(torch.cat([Z, Y], 1))

# Prior Working Class
# class CNNRGB(nn.Module):
#     def __init__(self, dimX, dimY):
#         super(CNNRGB, self).__init__()
#
#         self.input_height = dimX[0]
#         self.input_width = dimX[1]
#
#         self.conv_layers = nn.Sequential(
#             self._make_block(3, 16),
#             self._make_block(16, 32),
#             self._make_block(32, 64)
#         )
#
#         self.fc_layers_cache = {}  # cache to store dynamically created FC layers
#
#     def _make_block(self, in_channels, out_channels):
#         return nn.Sequential(
#             nn.Conv2d(in_channels, out_channels, kernel_size=5, padding=2, stride=2),
#             nn.SiLU(inplace=True),
#         )
#
#     def _get_fc_layers(self, input_dim):
#         # Check if the fc layers for this input dim already exist
#         if input_dim in self.fc_layers_cache:
#             return self.fc_layers_cache[input_dim]
#
#         # Otherwise, create new fc layers
#         fc_layers = nn.Sequential(
#             nn.Linear(input_dim + 1, 256+1),  # Added +1 for Y
#             nn.SiLU(inplace=True),
#             nn.Linear(256 + 1, 128+1),
#             nn.SiLU(inplace=True),
#             nn.Linear(128 + 1, 1),
#         )
#         self.fc_layers_cache[input_dim] = fc_layers
#         return fc_layers
#
#     def forward(self, X, Y):
#         Y = Y.view(-1, 1)
#
#         # Reshaping and permuting X as needed
#         if len(X.shape) == 6:
#             X = X.squeeze(1)
#         if X.shape[-1] == 3:
#             X = X.permute(0, 3, 1, 2)
#
#         Y_broadcasted = Y.view(Y.shape[0], 1, 1, 1).expand(-1, X.shape[1], X.shape[2], X.shape[3]).to(device)
#         Z = self.conv_layers(X + Y_broadcasted).to(device)
#         Z_flat = Z.reshape(Z.size(0), -1).to(device)
#
#         # Get the appropriate fc layers based on the input dim
#         fc_layers = self._get_fc_layers(Z_flat.size(1)).to(device)
#
#         return fc_layers(torch.cat([Z_flat, Y], 1))

class CNNRGB(torch.nn.Module):
    def __init__(self, dimX, dimY):
        super(CNNRGB, self).__init__()

        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(6, 5, 5, 1, padding=1),
            torch.nn.MaxPool2d(5, 2, 2),
            torch.nn.ReLU6(inplace=True),
        )
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv2d(5, 50, 5),
            torch.nn.MaxPool2d(5, 2, 2),
            torch.nn.ReLU6(inplace=True),
        )
        self.fc1 = torch.nn.Sequential(
            torch.nn.Linear(50 * 6 * 6 + 1, 100),
            torch.nn.ReLU6(inplace=True),
        )
        self.fc2 = torch.nn.Sequential(
            torch.nn.Linear(100 + 1, 50),
            torch.nn.ReLU6(inplace=True)
        )
        self.fc3 = torch.nn.Sequential(
            torch.nn.Linear(50 + 1, 1)
        )

    def forward(self, X, Y):
        Y = Y.view(-1, 1)
        Z = self.conv1(X)
        Z = self.conv2(Z)
        Z = Z.view(Z.size(0), -1)  # Flatten Z before passing to fully connected layer
        Z = self.fc1(torch.cat([Z, Y], 1))
        Z = self.fc2(torch.cat([Z, Y], 1))
        return self.fc3(torch.cat([Z, Y], 1))

class MINE(nn.Module):

    def __init__(self, dimX, dimY, network_type, moving_average_rate=0.001, hidden_size=512):
        super(MINE, self).__init__()

        if network_type == "mlp":
            self.network = MLP(dimX, dimY, hidden_size).to(device)
        elif network_type == "cnn":
            self.network = CNN(dimX, dimY).to(device)
        elif network_type == "cnnRGB":
            self.network = CNNRGB(dimX, dimY).to(device)
        self.network.apply(utils.weight_init)
        self.moving_average_rate = moving_average_rate

    def forward(self, dimX, dimY):
        return self.network.forward(dimX, dimY)

    def get_mi(self, X, Y, Y_tilde):
        T = self.network(X, Y).mean()
        expT = torch.exp(self.network(X, Y_tilde)).mean()
        mi = (T - torch.log(expT)).item() / math.log(2)
        return mi, T, expT

    def getMIMine(self, XT, label, label_tilde):
        T = self.mine_model(XT, label)
        expT = torch.exp(self.mine_model(XT, label_tilde))
        mi_tensor = T - torch.log(expT)
        mi_tensor /= math.log(2)  # Convert to bits
        average_mi = torch.mean(mi_tensor).item()
        return average_mi

    def train(self, dataset, learning_rate, batch_size, n_iterations, n_verbose, n_window, save_progress, dims, args):
        # Because of the moving average, we can't continue training from a prior model checkpoint
        optimizer = torch.optim.Adam(self.network.parameters(), lr=learning_rate)

        mi_progress = []
        moving_average_expT = 1
        mi = torch.empty(n_window)

        # Define the checkpoints at which the model should be saved
        save_intervals = [1000, 5000, 10000, 25000, 50000, 100000, 150000, 250000, 500000, 750000, 1000000]

        mine_model_dir = os.path.abspath(os.path.join("../../models/mine model", args.example))
        os.makedirs(mine_model_dir, exist_ok=True)

        another_model_dir = os.path.abspath(os.path.join("../models", args.example))
        os.makedirs(another_model_dir, exist_ok=True)

        if save_progress > 0:
            mi_progress = torch.zeros(int(n_iterations / save_progress))

        for iteration in range(n_iterations):
            X, Y, Y_tilde = dataset.sample_batch(batch_size)

            X = torch.autograd.Variable(X).to(device)
            Y = torch.autograd.Variable(Y).to(device)
            Y_tilde = torch.autograd.Variable(Y_tilde).to(device)

            mi_lb, T, expT = self.get_mi(X, Y, Y_tilde)
            moving_average_expT = ((1 - self.moving_average_rate) * moving_average_expT + self.moving_average_rate * expT).item()
            loss = -1.0 * (T - expT / moving_average_expT)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                mi[iteration % n_window] = mi_lb
                if iteration >= n_window and iteration % n_verbose == n_verbose - 1:
                    print(f"Iteration {iteration + 1}: {mi.mean().item()}")

                if save_progress > 0 and iteration % save_progress == save_progress - 1:
                    mi_progress[int(iteration / save_progress)] = mi.mean().item()

            if iteration + 1 in save_intervals:
                if args.example == "gaussian":
                    checkpoint_path_gaussian = os.path.join(f"../../models/mine model/{args.example}/{args.d}D", f"MI E{iteration + 1}.pt")
                    torch.save(self.network, checkpoint_path_gaussian)
                    print(f"Gaussian Checkpoint saved at iteration {iteration + 1}: {checkpoint_path_gaussian}")

                    checkpoint_path_other_alternate = os.path.join(f"../models/{args.example}/{args.d}D", f"MI E{iteration + 1}.pt")
                    torch.save(self.network, checkpoint_path_other_alternate)
                    print(f"Alternate checkpoint saved at iteration {iteration + 1}: {checkpoint_path_other_alternate}")
                else:
                    # Save checkpoints for other types of datasets
                    checkpoint_path_other = os.path.join(f"../../models/mine model/{args.example}/", f"MI E{iteration + 1}.pt")
                    torch.save(self.network, checkpoint_path_other)
                    print(f"Checkpoint saved at iteration {iteration + 1}: {checkpoint_path_other}")

                    checkpoint_path_other_alternate = os.path.join(f"../models/{args.example}/", f"MI E{iteration + 1}.pt")
                    torch.save(self.network, checkpoint_path_other_alternate)
                    print(f"Alternate checkpoint saved at iteration {iteration + 1}: {checkpoint_path_other_alternate}")

        if save_progress > 0:
            return mi_progress

        return mi.mean().item()
