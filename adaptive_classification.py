#  Misc. Libraries
import os
import re
import shutil
import uuid
from PIL import Image
from tqdm import tqdm
from collections import defaultdict

#  General Libraries
import numpy as np
import matplotlib
import matplotlib.style as mplstyle
from matplotlib import pyplot as plt

#  Torch Libraries
import torch
import torch.nn.functional as func
import torch.utils.data
import torchvision
import torchvision.transforms.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms import ToPILImage

#  Image Libraries
from image_transform import RGBTransform, GrayscaleTransform

#  DDPM Libraries
from ddpm import ddpm  # These are used even if PyCharm says they aren't

#  Model Libraries
from dataset_classifier import Net, NetRGB  # These are used even if PyCharm says they aren't

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.cuda.set_device(torch.cuda.current_device())

#  Set Seed for Reproducibility

#  seed = 3407  # https://arxiv.org/abs/2109.08203
seed = np.random.randint(0, 1_000_000)  # random seed
np.random.seed(seed)
torch.manual_seed(seed)

matplotlib.use("Agg")
mplstyle.use(["dark_background", "fast"])


def main():
    dataset = "mnist"  # "mnist", "fashion-mnist", "cifar10"
    schedule = "sigmoid"  # "auto", "linear", "cosine" "sigmoid", "geometric"

    ddpm_epochs_options = [1, 5, 10, 25, 50, 100, 250, 500, 750, 1000]  # Full

    arbitrary_steps = 1000
    arbitrary_ddpm_epochs = 1000

    generate_data_quantity = 8
    generate_data = True
    destroy_data = True
    destroy_failsafe = True

    plot_grid = False

    classify_single_epoch = True

    plot_full_time_series = True

    # if plot_grid:
    #     classify_single_epoch = False
    #
    #     noisy_classification_model = True
    #     classify = adaptive_classification(dataset, schedule, arbitrary_steps, arbitrary_ddpm_epochs, generate_data, destroy_data, destroy_failsafe, generate_data_quantity, noisy_classification_model, plot_grid, classify_single_epoch)
    #     classify.run()
    #
    #     noisy_classification_model = False
    #     classify = adaptive_classification(dataset, schedule, arbitrary_steps, arbitrary_ddpm_epochs, generate_data, destroy_data, destroy_failsafe, generate_data_quantity, noisy_classification_model, plot_grid, classify_single_epoch)
    #     classify.run()

    if plot_full_time_series:
        noisy_classification_model = True
        for ddpm_epochs in ddpm_epochs_options:
            classify = adaptive_classification(dataset, schedule, arbitrary_steps, ddpm_epochs, generate_data, destroy_data, destroy_failsafe, generate_data_quantity, noisy_classification_model, plot_grid, classify_single_epoch)
            classify.run()

        noisy_classification_model = False
        generate_data = False
        destroy_data = False
        destroy_failsafe = False
        for ddpm_epochs in ddpm_epochs_options:
            classify = adaptive_classification(dataset, schedule, arbitrary_steps, ddpm_epochs, generate_data, destroy_data, destroy_failsafe, generate_data_quantity, noisy_classification_model, plot_grid, classify_single_epoch)
            classify.run()

    # if classify_single_epoch:
    #     classify = adaptive_classification(dataset, schedule, arbitrary_steps, arbitrary_ddpm_epochs, generate_data, destroy_data, destroy_failsafe, generate_data_quantity, noisy_classification_model, plot_grid, classify_single_epoch)
    #     classify.run()


# TODO: Take MNIST classifier and work it on image pairs to do the test
# TODO: Create charts / graphs for diffusion error classification


class adaptive_classification:
    def __init__(self, dataset, schedule, steps, ddpm_epochs, generate_data, destroy_data, destroy_failsafe, generate_data_quantity, noisy_classification_model, plot_grid, classify_single_epoch):
        super(adaptive_classification).__init__()

        self.ddpm_model = None
        self.noisy_classification_model = noisy_classification_model

        self.loadData = dataset
        self.scheduler = schedule

        self.batchSize = 512
        self.steps = steps
        self.ddpm_model_epochs = ddpm_epochs

        # We only need to generate this dataset once using the ddpm_models
        self.generate_novel_data = generate_data  # Generates novel data
        self.destroy_data_directories = destroy_data  # Delete previous reconstruction data
        self.destroy_failsafe = destroy_failsafe  # Failsafe for generate_novel_data and destroy_data_directories
        self.generate_data_quantity = generate_data_quantity  # Generates self.batchSize * self.generate_data_quantity images (will save train and recon data so ultimately 4 * 512 * 2)

        self.classify_single_epoch = classify_single_epoch
        self.plotGrid = plot_grid

        if dataset == "mnist" or dataset == "fashion-mnist":
            self.numChannels = 1
            self.imageSize = 28
            if schedule == "auto":
                self.scheduler = "linear"
        elif dataset == "cifar10":
            self.numChannels = 3
            self.imageSize = 32
            if schedule == "auto":
                self.scheduler = "linear"
        elif dataset == "celeba":
            self.numChannels = 3
            self.imageSize = 64
            if schedule == "auto":
                self.scheduler = "sigmoid"

        self.initializeDiffusionParameters()

    def initializeDiffusionParameters(self):
        # Assuming 'getSchedule' is a method of the class that this function belongs to
        self.Beta = self.getSchedule(self.scheduler).to(device)  # Adjusted Noise Schedule based on step size
        Sqrt_Sigma = self.Beta  # Sigma Squared
        Alpha = 1.0 - self.Beta  # Alpha Schedule

        self.Alpha_Bar = torch.cumprod(Alpha, dim=0)  # Product Value of Alpha
        self.Sqrt_Alpha_Cumprod = torch.sqrt(self.Alpha_Bar)  # Square Root of Product Value of Alpha
        Alpha_Cumprod_Previous = func.pad(self.Alpha_Bar[:-1], (1, 0), value=1.0)  # Previous Product Value of Alpha
        self.Sqrt_1_Minus_Alpha_Cumprod = torch.sqrt(1.0 - self.Alpha_Bar)  # Square Root of 1 - Product Value of Alpha
        Log_one_minus_Alpha_Cumprod = torch.log(1.0 - self.Alpha_Bar)  # Log of 1 - Product Value of Alpha

        self.Sqrt_Recipricol_Alpha_Cumprod = torch.sqrt(1.0 / self.Alpha_Bar)  # Square Root of Reciprocal of Product Value of Alpha
        self.Sqrt_Recipricol_Alpha_Cumprod_Minus_1 = torch.sqrt(1.0 / self.Alpha_Bar - 1)  # Square Root of Reciprocal of Product Value of Alpha - 1

        self.Posterior_Variance = self.Beta * (1.0 - Alpha_Cumprod_Previous) / (1.0 - self.Alpha_Bar)  # Var(x_{t-1} | x_t, x_0)
        self.Posterior_Log_Variance_Clamp = torch.log(self.Posterior_Variance.clamp(min=1e-20))  # Log of Var(x_{t-1} | x_t, x_0)
        self.Posterior1 = (self.Beta * torch.sqrt(Alpha_Cumprod_Previous) / (1.0 - self.Alpha_Bar))  # 1 / (Var(x_{t-1} | x_t, x_0))
        self.Posterior2 = (1.0 - Alpha_Cumprod_Previous) * torch.sqrt(Alpha) / (1.0 - self.Alpha_Bar)  # (1 - Alpha_{t-1}) / (Var(x_{t-1} | x_t, x_0))

    def printparameters(self):
        print(f"Dataset: {self.loadData}")
        print(f"Scheduler: {self.scheduler}")
        print(f"Batch Size: {self.batchSize}")
        print(f"DDPM Epochs: {self.ddpm_model_epochs}")
        print(f"Generate Novel Data: {self.generate_novel_data}")
        print(f"Generate Data Quantity: {self.generate_data_quantity * self.batchSize} images")
        print(f"Training Data Quantity: {self.generate_data_quantity * self.batchSize} images")
        print(f"Noisy Classification Model: {self.noisy_classification_model}\n\n")

    def prepareDirectories(self):
        model_path = f"./models/ddpm model/{self.loadData}/{self.scheduler}"
        reconstruction_path = os.path.join("./datasets", "reconstruction data")
        os.makedirs(reconstruction_path, exist_ok=True)
        schedule_path = os.path.join(reconstruction_path, self.loadData, self.scheduler)
        os.makedirs(schedule_path, exist_ok=True)
        epoch_path = os.path.join(schedule_path, f"E {self.ddpm_model_epochs}")
        os.makedirs(epoch_path, exist_ok=True)

        model_directories = []
        for d in os.listdir(model_path):
            full_dir_path = os.path.join(model_path, d)
            if os.path.isdir(full_dir_path):
                model_directories.append(d)

        # Create and clean directories
        for dir_name in model_directories:
            t_folder_path = os.path.join(epoch_path, f"T {dir_name}")
            os.makedirs(t_folder_path, exist_ok=True)

        # Check if files exist
        files_exist = False
        for dir_name in model_directories:
            full_dir_path = os.path.join(epoch_path, f"T {dir_name}")
            if os.listdir(full_dir_path):
                files_exist = True
                break

        if files_exist and self.generate_novel_data:
            print(f"Previous files found in the target directories under {epoch_path}.")
            self.delete_files(epoch_path)
        elif files_exist:
            print("Data files found.\n")

    def delete_files(self, path):
        response = input(f"Are you sure you want to delete all files and directories in {path}? (yes/no): ")
        if response.lower() != 'yes':
            print("Deletion cancelled.")
            return
        try:
            for root, directories, files in os.walk(path, topdown=False):
                for file in files:
                    os.unlink(os.path.join(root, file))
                for dirs in directories:
                    shutil.rmtree(os.path.join(root, dirs))
            print("Files deleted successfully.\n\n")
        except Exception as e:
            print(f"An error occurred while deleting files: {e}")

    def setDDPMModels(self):
        base_path = f"./models/ddpm model/{self.loadData}/{self.scheduler}"
        if not os.path.exists(base_path):
            print(f"Base path not found: {base_path}")
            return

        counter = 0
        for step in self.get_sorted_timesteps(base_path):
            model_path = os.path.join(base_path, step, f"ddpm E{self.ddpm_model_epochs}.pt")
            if os.path.isfile(model_path):
                counter += 1
                setattr(self, f"ddpm_model_{counter}", torch.load(model_path))
                print(f"DDPM Model {counter}: Loaded with {self.ddpm_model_epochs} epochs at {step} time steps.")
            else:
                print(f"Model file not found for {step} time steps. Please check the path: {model_path}")
        print("\n")
        if counter == 0:
            print("No models were loaded. Please check the paths and epoch settings.")

    def getDataset(self):
        trainData = None
        grey = GrayscaleTransform(self.imageSize, self.numChannels)
        RGB = RGBTransform(self.imageSize)

        if self.loadData == "mnist":
            trainData = torchvision.datasets.MNIST(root="./datasets", download=True, train=True, transform=grey)
        elif self.loadData == "fashion-mnist":
            trainData = torchvision.datasets.FashionMNIST(root="./datasets", download=True, train=True, transform=grey)
        elif self.loadData == "cifar10":
            trainData = torchvision.datasets.CIFAR10(root="./datasets", download=True, train=True, transform=RGB)
        elif self.loadData == "celeba":
            trainData = torchvision.datasets.CelebA(root="./datasets", download=True, transform=RGB)

        DataSet = DataLoader(trainData, batch_size=self.batchSize, shuffle=True,  pin_memory=True, num_workers=4, pin_memory_device="cuda", persistent_workers=True)

        if self.loadData == "celeba":
            Label = trainData.identity
        else:
            Label = trainData.targets
        return trainData, DataSet, Label

    def dataGenerationLoop(self, dataset):
        for idx, (X0, labels) in enumerate(dataset):
            if idx >= self.generate_data_quantity:
                break
            X0, batched_labels = X0.to(device), labels.to(device)
            print(f"Processing Trajectories T {self.steps} | Batch: {idx + 1}")
            self.saveReconImages(batched_labels, X0)
            print("\033[K", end='\r')  # Clear tqdm line

    def getSchedule(self, schedule):
        if schedule == "linear":
            return torch.linspace(1e-4, 2e-2, self.steps)
        elif schedule == "cosine":
            return self.cosine_schedule()
        elif schedule == "sigmoid":
            return self.sigmoid_schedule(-3, 3)
        elif schedule == "snr":
            return self.snr_schedule(1e-4, 2e-2)
        elif schedule == "geometric":
            return self.geometric_snr_schedule(1e-4, 2e-2)

    def cosine_schedule(self):
        x = torch.linspace(0, self.steps, self.steps + 1, device=device)
        y = torch.cos(((x / self.steps) + 0.01) / (1 + 0.01) * torch.pi * 0.5) ** 2
        z = y / y[0]
        return torch.clip((1 - (z[1:] / z[:-1])), 0.0001, 0.9999).to(device)

    def sigmoid_schedule(self, start, end):
        sequence = torch.linspace(0, self.steps, self.steps + 1, dtype=torch.float32, device=device) / self.steps
        v_start = torch.tensor(start, dtype=torch.float32, device=device).sigmoid()
        v_end = torch.tensor(end, dtype=torch.float32, device=device).sigmoid()
        alpha = (-((sequence * (end - start) + start) / 1).sigmoid() + v_end) / (v_end - v_start)
        alpha = alpha / alpha[0]
        betas = 1 - (alpha[1:] / alpha[:-1])
        return torch.clamp(betas, 0.0001, 0.9999)

    def snr_schedule(self, SNR_start, SNR_end):
        betas = torch.linspace(SNR_start, SNR_end, steps=self.steps, device=device)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = func.pad(alphas_cumprod[:-1], (1, 0), value=1.0).to(device)
        sqrt_alphas_cumprod_prev = torch.sqrt(alphas_cumprod_prev)
        sqrt_one_minus_alphas_cumprod_prev = torch.sqrt(1. - alphas_cumprod_prev)
        sqrt_one_minus_alphas_cumprod_prev[0] = 1e-10
        timesteps_to_noise_ratios = sqrt_alphas_cumprod_prev / sqrt_one_minus_alphas_cumprod_prev
        return timesteps_to_noise_ratios.to(device)

    def geometric_snr_schedule(self, SNR_start, SNR_end):
        ratio = (SNR_end / SNR_start) ** (1 / (self.steps - 1))
        snr_values = SNR_start * (ratio ** torch.arange(self.steps, device=device))
        return torch.clamp(snr_values, 0.0001, 0.9999).to(device)

    def p_posterior_mean_variance(self, X0, Xt, t):  # p(x_{t-1} | x_t, x_0) = N(posterior_mean, posterior_model_variance)
        posterior_mean = self.getExtract(self.Posterior1, t, Xt.shape) * X0 + self.getExtract(self.Posterior2, t, Xt.shape) * Xt  # p(x_{t-1} | x_t, x_0) = N(posterior1_mean, posterior_model)
        posterior_model_variance = self.getExtract(self.Posterior_Variance, t, Xt.shape)  # p(x_{t-1} | x_t, x_0) = N(posterior2_mean, posterior_model_variance)
        return posterior_mean, posterior_model_variance

    def q_posterior_mean_variance(self, X0, Xt, t):  # q(x_{t-1} | x_t, x_0) = N(posterior_mean, posterior_model_variance)
        posterior_mean = self.getExtract(self.Posterior1, t, Xt.shape) * X0 + self.getExtract(self.Posterior2, t, Xt.shape) * Xt
        posterior_model_variance = self.getExtract(self.Posterior_Variance, t, Xt.shape)
        posterior_model_log_variance_clamp = self.getExtract(self.Posterior_Log_Variance_Clamp, t, Xt.shape)
        return posterior_mean, posterior_model_variance, posterior_model_log_variance_clamp

    def q_sample(self, X0, t):  # Sample from q(Xt | X0) = N(x_t; sqrt(alpha_bar_t) * x_0, sqrt(1 - alpha_bar_t) * noise, t)
        noise = torch.randn_like(X0, device=device)  # Sample from N(0, I)
        QSample = self.getExtract(self.Sqrt_Alpha_Cumprod, t, X0.shape) * X0 + self.getExtract(self.Sqrt_1_Minus_Alpha_Cumprod, t, X0.shape) * noise  # Sample from q(Xt | X0)
        return QSample, noise

    def pred_X0_from_XT(self, Xt, noise, t):  # p(x_{t-1} | x_t)
        return self.getExtract(self.Sqrt_Recipricol_Alpha_Cumprod, t, Xt.shape) * Xt - self.getExtract(self.Sqrt_Recipricol_Alpha_Cumprod_Minus_1, t, Xt.shape) * noise  # Sample from p(x_{t-1} | x_t)

    def pred_V(self, X0, t, noise):
        return self.getExtract(self.Sqrt_Alpha_Cumprod, t, X0.shape) * noise - self.getExtract(self.Sqrt_1_Minus_Alpha_Cumprod, t, X0.shape) * X0  # Sample from q(V | X0)

    # TODO: Update p_prior_mean_variance to account for predicting_X0 and predicting_V from the target objective.
    def p_prior_mean_variance(self, Xt, t):  # Sample from p_{theta}(x_{t-1} | x_t) & q(x_{t-1} | x_t, x_0) = N(posterior_mean, posterior_model_variance)
        prediction = self.pred_X0_from_XT(Xt.float(), self.ddpm_model(Xt.float(), t), t)  # p(x_{t-1} | x_t)
        if self.loadData != "gaussian":
            prediction = prediction.clamp(-1.0, 1.0)  # Clamp to [-1, 1]
        model_mean, posterior_variance, posterior_log_variance = self.q_posterior_mean_variance(prediction, Xt, t)  # Sample from q(x_{t-1} | x_t, x_0)
        return model_mean, posterior_variance, posterior_log_variance

    @torch.no_grad()
    def p_sample(self, sample, t):  # Sample from p_{theta}(x_{t-1} | x_t) = N(x_{t-1}; UNet(x_{t}, t), sigma_bar_t * I)
        mean, posterior_variance, posterior_log_variance = self.p_prior_mean_variance(sample, t)  # Sample from p_{theta}(x_{t-1} | x_t)
        noise = torch.randn_like(sample, device=device)  # Sample from N(0, I)
        nonzero_mask = ((t != 0).float().view(-1, *([1] * (len(sample.shape) - 1))))  # Mask for t != 0
        return mean + nonzero_mask * torch.exp(0.5 * posterior_log_variance) * noise  # Sample from p_{theta}(x_{t-1} | x_t)

    @staticmethod
    def getExtract(tensor: torch.Tensor, t: torch.Tensor, X):
        return tensor.gather(-1, t).view(t.shape[0], *((1,) * (len(X) - 1)))

    @torch.no_grad()
    def ForwardTrajectory(self, init_image, description):
        XT = init_image.clone()
        for step in tqdm(range(0, self.steps), desc=description):
            XT = self.p_sample(XT, torch.full((self.batchSize,), step, device=device, dtype=torch.long))
        print("\033[K", end='\r')  # Clear tqdm line
        return XT

    @torch.no_grad()
    def ReverseTrajectory(self, init_image, description):
        XT = init_image.clone()
        for step in tqdm(reversed(range(0, self.steps)), desc=description):
            XT = self.p_sample(XT, torch.full((self.batchSize,), step, device=device, dtype=torch.long))
        print("\033[K", end='\r')  # Clear tqdm line
        return XT

    def calculate_accuracy(self, predictions, epoch, label, timesteps):
        accuracies = []
        for t in timesteps:
            prediction = predictions[epoch][t].get(label, {"total": 0, "correct": 0})
            total = prediction["total"]
            correct = prediction["correct"]
            if total == 0:
                total = 1
            accuracies.append(correct / total)
        return accuracies

    def calculate_accuracy_by_class(self, predictions, class_labels, timesteps):
        accuracy_by_class = {}
        for label in class_labels:
            accuracies = []
            for t in timesteps:
                prediction = predictions[t].get(label, {"total": 0, "correct": 0})
                total = prediction["total"]
                correct = prediction["correct"]
                if total == 0:
                    total = 1
                accuracies.append(correct / total)
            accuracy_by_class[label] = accuracies
        return accuracy_by_class

    def plotGridGraph(self, input_predictions, recon_predictions):
        class_labels = set()
        for epoch_data in input_predictions.values():
            for label_data in epoch_data.values():
                class_labels.update(label_data.keys())

        epochs = sorted(input_predictions.keys())
        class_labels = sorted(class_labels)

        num_rows = len(epochs)
        num_cols = len(class_labels)

        fig, axs = plt.subplots(num_rows, num_cols, figsize=(num_cols * 5, num_rows * 4))

        for row, epoch in enumerate(epochs):
            for col, label in enumerate(class_labels):
                ax = axs[row, col]
                timesteps = sorted(input_predictions[epoch].keys())
                input_accuracy = self.calculate_accuracy(input_predictions, epoch, label, timesteps)
                recon_accuracy = self.calculate_accuracy(recon_predictions, epoch, label, timesteps)

                # Plotting
                ax.plot(timesteps, input_accuracy, color="blue", label="Input Accuracy", marker="o", linestyle="-")
                ax.plot(timesteps, recon_accuracy, color="red", label="Recon Accuracy", marker="o", linestyle="-")

                ax.set_title(f"Epoch: {epoch}, Label: {label}")
                ax.set_xlabel("Timestep (T)")
                ax.set_ylabel("Classification Accuracy")
                ax.legend(loc="upper right")

        # Adjust layout and display the plot
        plt.tight_layout()
        if self.noisy_classification_model:
            plt.savefig(f"./datasets/reconstruction data/{self.loadData}/{self.scheduler}/Noised Model Classification Accuracy of DDPM models.png")
        else:
            plt.savefig(f"./datasets/reconstruction data/{self.loadData}/{self.scheduler}/Classification Accuracy of DDPM model.png")
        plt.close()

    # Create plots for the classification accuracy
    def plotPointGraph(self, input_predictions, input_labels, recon_predictions, recon_labels):
        class_labels_input = set()
        for timestep_data in input_predictions.values():
            for label in timestep_data.keys():
                class_labels_input.add(label)

        class_labels_recon = set()
        for timestep_data in recon_predictions.values():
            for label in timestep_data.keys():
                class_labels_recon.add(label)

        class_labels = class_labels_input.union(class_labels_recon)  # Combine input and recon class labels
        timesteps = sorted(list(input_predictions.keys()), key=int)  # Sort the timesteps numerically

        if not class_labels:
            print("No class labels found. Skipping plot generation.")
            return

        input_accuracy_by_class = self.calculate_accuracy_by_class(input_predictions, class_labels, timesteps)
        recon_accuracy_by_class = self.calculate_accuracy_by_class(recon_predictions, class_labels, timesteps)

        num_rows = (len(class_labels) + 1) // 2
        num_cols = 2
        fig, axs = plt.subplots(num_rows, num_cols, figsize=(20, 15), squeeze=False)  # Ensure axs is always 2D

        for i, label in enumerate(class_labels):
            row, col = divmod(i, num_cols)
            ax = axs[row, col]

            # Plotting with 'o-' to connect the points with lines
            ax.plot(timesteps, input_accuracy_by_class[label], 'o-', color="blue", label="Input Acc.")
            ax.plot(timesteps, recon_accuracy_by_class[label], 'o-', color="red", label="Recon Acc.")

            ax.set_title(f"Class Label: {label}")
            ax.set_xlabel("Time (t)")
            ax.set_ylabel("Accuracy")
            ax.set_ylim([-0.05, 1.05])
            ax.legend(loc="upper right")

        plt.tight_layout()
        if self.noisy_classification_model:
            plt.savefig(f"./datasets/reconstruction data/{self.loadData}/{self.scheduler}/E {self.ddpm_model_epochs}/Noised Model Classification Accuracy of DDPM model epochs {self.ddpm_model_epochs}.png")
        else:
            plt.savefig(f"./datasets/reconstruction data/{self.loadData}/{self.scheduler}/E {self.ddpm_model_epochs}/Classification Accuracy of DDPM model epochs {self.ddpm_model_epochs}.png")
        plt.close()

    def plotBarChart(self, input_predictions, input_labels, recon_predictions, recon_labels):
        class_labels_input = set()
        for timestep_data in input_predictions.values():
            for label in timestep_data.keys():
                class_labels_input.add(label)

        class_labels_recon = set()
        for timestep_data in recon_predictions.values():
            for label in timestep_data.keys():
                class_labels_recon.add(label)

        class_labels = class_labels_input.union(class_labels_recon)  # Combine input and recon class labels
        timesteps = sorted(input_predictions.keys())

        input_accuracy_by_class = {}
        recon_accuracy_by_class = {}

        for label in class_labels:
            input_accuracy_by_class[label] = []
            recon_accuracy_by_class[label] = []

        # Collecting accuracy data
        for label in class_labels:
            for t in timesteps:
                input_data = input_predictions[t].get(label, {"total": 0, "correct": 0})
                input_accuracy = input_data["correct"] / input_data["total"] if input_data["total"] > 0 else 0.0
                input_accuracy_by_class[label].append(input_accuracy)

                recon_data = recon_predictions[t].get(label, {"total": 0, "correct": 0})
                recon_accuracy = recon_data["correct"] / recon_data["total"] if recon_data["total"] > 0 else 0.0
                recon_accuracy_by_class[label].append(recon_accuracy)

        fig, ax = plt.subplots(figsize=(20, 15))
        width = 0.35

        timesteps_indices = np.arange(len(timesteps)) * (len(class_labels) + 1) * width

        for i, label in enumerate(sorted(class_labels)):
            recon_acc = recon_accuracy_by_class[label]
            bar_positions = timesteps_indices + i * width  # X location of bars for this class label
            ax.bar(bar_positions, recon_acc, width, label=f"Recon: {label}", align="center")

        ax.set_xlabel("Timesteps")
        ax.set_ylabel("Accuracy")
        ax.set_title("Reconstruction Accuracy by Class and Timestep")

        mid_group_offset = (len(class_labels) - 1) * width / 2
        ax.set_xticks(timesteps_indices + mid_group_offset)
        ax.set_xticklabels(timesteps)

        ax.legend()

        plt.tight_layout()
        if self.noisy_classification_model:
            plt.savefig(f"./datasets/reconstruction data/{self.loadData}/{self.scheduler}/E {self.ddpm_model_epochs}/Noised Model Reconstruction Accuracy Bar Chart of DDPM model epochs {self.ddpm_model_epochs}.png")
        else:
            plt.savefig(f"./datasets/reconstruction data/{self.loadData}/{self.scheduler}/E {self.ddpm_model_epochs}/Reconstruction Accuracy Bar Chart of DDPM model epochs {self.ddpm_model_epochs}.png")
        plt.close()

    def get_sorted_timesteps(self, base_path):
        timesteps_dirs = []
        for directories in os.listdir(base_path):
            full_path = os.path.join(base_path, directories)
            if os.path.isdir(full_path):
                timesteps_dirs.append(directories)
        timestep_pattern = re.compile(r"\d+")
        return sorted(timesteps_dirs, key=lambda x: int(timestep_pattern.findall(x)[0]) if timestep_pattern.findall(x) else 0)

    def generateData(self, dataset):
        base_path = f"./models/ddpm model/{self.loadData}/{self.scheduler}"
        model_files = os.listdir(base_path)
        model_pattern = re.compile(r"(\d+)")

        step_to_model = {}
        counter = 0

        for filename in model_files:
            match = model_pattern.match(filename)
            if match:
                counter += 1
                step = int(match.group(1))
                model_suffix = f"ddpm_model_{counter}"
                step_to_model[step] = model_suffix

        sorted_steps = sorted(step_to_model.keys())

        for step in sorted_steps:
            print(f"Processing step: {step}")
            self.process_step((step, step_to_model[step], dataset))

    def process_step(self, args):
        step, model_suffix, dataset = args
        self.steps = step
        self.initializeDiffusionParameters()
        self.ddpm_model = getattr(self, model_suffix)
        self.dataGenerationLoop(dataset)

    def saveReconImages(self, batched_labels, X0):
        savePath = f"./datasets/reconstruction data/{self.loadData}/{self.scheduler}/E {self.ddpm_model_epochs}/T {self.steps}/"
        if not os.path.exists(savePath):
            os.makedirs(savePath, exist_ok=True)
        current_labels = batched_labels.clone().detach()
        X0_batch, current_labels_batch, current_labels_marginal_batch = self.sample_batch_from_X0_and_labels(X0, current_labels, self.batchSize)
        self.process_trajectory(X0_batch, current_labels_batch, savePath)

    def sample_batch_from_X0_and_labels(self, X0, current_labels, batch_size):
        data_len = len(X0)
        index_joint = np.random.choice(range(data_len), size=batch_size, replace=False)
        index_marginal = np.random.choice(range(data_len), size=batch_size, replace=False)
        X0_batch = X0[index_joint]
        current_labels_batch = current_labels[index_joint]
        current_labels_marginal_batch = current_labels[index_marginal]
        return X0_batch, current_labels_batch, current_labels_marginal_batch

    def process_trajectory(self, training_input, current_labels_batch, savePath):
        final_noise = self.ForwardTrajectory(training_input, description=f"Forward Process")
        noise_recon = self.ReverseTrajectory(final_noise, description=f"Reverse Process")

        self.saveImagePair(training_input, noise_recon, current_labels_batch, savePath)

    def process_images_batch(self, batch_args):
        filenames, net, device = batch_args
        batch_imgs = []
        for filename in filenames:
            transformed_image = self.load_and_transform_image(filename, device)
            batch_imgs.append(transformed_image)
        batch_tensor = torch.stack(batch_imgs).to(device)
        outputs = net(batch_tensor)
        predicted_labels = torch.argmax(outputs, dim=1).tolist()
        results = []
        for i, filename in enumerate(filenames):
            label = self.extract_label_from_filename(filename)
            result_type = 'input' if "Input" in filename else 'recon'
            results.append((result_type, predicted_labels[i], label))
        return results

    def load_and_transform_image(self, filename, device):
        # Ensure this method returns a tensor of shape [channels, height, width]
        with Image.open(filename) as img:
            if self.loadData in ["mnist", "fashion-mnist"]:
                img = transforms.functional.to_grayscale(img)
                transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Resize((28, 28), antialias=False)
                ])
            elif self.loadData == "cifar10":
                transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Resize((32, 32), antialias=False)
                ])
            return transform(img).to(device)

        # Old Method for MNIST
        # resize_transform = transforms.Compose([
        #     transforms.ToTensor(),
        #     transforms.Resize((28, 28), antialias=False)
        # ])
        # with Image.open(filename) as img:
        #     img = transforms.functional.to_grayscale(img)
        #     return resize_transform(img).to(device)

    def saveImagePair(self, input_img_batch, recon_img_batch, current_labels_batch, savePath):
        convertToImage = ToPILImage()
        os.makedirs(savePath, exist_ok=True)

        for i, (input_img, recon_img, label) in enumerate(zip(input_img_batch, recon_img_batch, current_labels_batch)):
            label = label.item()
            unique_id = uuid.uuid4()

            input_img = convertToImage(input_img.squeeze(0))
            recon_img = convertToImage(recon_img.squeeze(0))

            input_filename = f"{savePath} Label {label} ID {unique_id} A Input.jpg"
            recon_filename = f"{savePath} Label {label} ID {unique_id} B Reconstructed Noise.jpg"

            input_img.save(input_filename)
            recon_img.save(recon_filename)

    def classify_and_compare(self, net, input_image_filenames, recon_image_filenames, batch_size=512):
        net = net.to(device)
        all_input_results, all_recon_results = [], []

        # Process input images
        print("Processing Input Images")
        for i in range(0, len(input_image_filenames), batch_size):
            batch_filenames = input_image_filenames[i:i + batch_size]
            batch_args = (batch_filenames, net, device)
            batch_results = self.process_images_batch(batch_args)
            all_input_results.extend(batch_results)

        # Process reconstructed images
        print("Processing Reconstructed Images")
        for i in range(0, len(recon_image_filenames), batch_size):
            batch_filenames = recon_image_filenames[i:i + batch_size]
            batch_args = (batch_filenames, net, device)
            batch_results = self.process_images_batch(batch_args)
            all_recon_results.extend(batch_results)

        # Extract the results
        input_predictions, input_labels = self.extract_results(all_input_results)
        recon_predictions, recon_labels = self.extract_results(all_recon_results)

        return input_predictions, recon_predictions, input_labels, recon_labels

    def extract_results(self, results):
        predictions, labels = [], []
        for result_type, predicted_label, label in results:
            predictions.append(predicted_label)
            labels.append(label)
        return predictions, labels

    @staticmethod
    def extract_label_from_filename(filepath):
        label = label_part = None
        parts = filepath.split(os.path.sep)  # Split the filepath

        for part in parts:
            if "Label" in part:
                label_part = part
                break

        if "Label" in label_part:
            label_match = re.search(r'Label (\d)', label_part)  # Get just label
            if label_match:
                label = int(label_match.group(1))

        return label

    def process_directory(self, model, dir_path):
        image_filenames = []
        for filename in os.listdir(dir_path):
            full_path = os.path.join(dir_path, filename)
            image_filenames.append(full_path)

        if not image_filenames:
            return {}, {}, {}, {}

        input_image_filenames = []
        recon_image_filenames = []

        for f in image_filenames:
            if "A Input" in f:
                input_image_filenames.append(f)
            if "B Reconstructed" in f:
                recon_image_filenames.append(f)

        input_predictions, recon_predictions, input_labels, recon_labels = self.classify_and_compare(model, input_image_filenames, recon_image_filenames)

        # Initializing dictionaries for counting predictions
        input_predictions_count = defaultdict(lambda: {"total": 0, "correct": 0})
        recon_predictions_count = defaultdict(lambda: {"total": 0, "correct": 0})

        # Counting correct predictions
        for input_pred, input_label in zip(input_predictions, input_labels):
            input_predictions_count[input_label]["total"] += 1
            if input_pred == input_label:
                input_predictions_count[input_label]["correct"] += 1

        for recon_pred, recon_label in zip(recon_predictions, recon_labels):
            recon_predictions_count[recon_label]["total"] += 1
            if recon_pred == recon_label:
                recon_predictions_count[recon_label]["correct"] += 1

        return input_predictions_count, recon_predictions_count, input_labels, recon_labels

    def classification(self, model, epoch):
        base_directory = f"./datasets/reconstruction data/{self.loadData}/{self.scheduler}/E {epoch}"
        if not os.path.exists(base_directory):
            print(f"The base directory {base_directory} does not exist.")
            return

        t_directories = []
        for dir_name in os.listdir(base_directory):
            if dir_name.startswith("T "):
                dir_number = int(dir_name[2:])
                full_path = os.path.join(base_directory, dir_name)
                t_directories.append((dir_number, full_path))

        t_directories.sort()

        total_input_predictions = {}
        total_recon_predictions = {}
        total_input_labels = {}
        total_recon_labels = {}

        for t, dir_path in t_directories:
            print(f"\nClassification of T = {t}")
            input_predictions, recon_predictions, input_labels, recon_labels = self.process_directory(model, dir_path)
            total_input_predictions[t] = input_predictions
            total_recon_predictions[t] = recon_predictions
            total_input_labels[t] = input_labels
            total_recon_labels[t] = recon_labels

        return total_input_predictions, total_recon_predictions, total_input_labels, total_recon_labels

    def process_all_epochs(self, model):
        ddpm_epochs_options = [1, 5, 10, 25, 50, 100, 250, 500, 750, 1000]
        aggregated_input_predictions, aggregated_recon_predictions = {}, {}

        print("Generating Grid")
        for epoch in ddpm_epochs_options:
            print(f"Processing Epoch: {epoch}")
            input_predictions, recon_predictions, _, _ = self.classification(model, epoch)
            aggregated_input_predictions[epoch], aggregated_recon_predictions[epoch] = input_predictions, recon_predictions

        self.plotGridGraph(aggregated_input_predictions, aggregated_recon_predictions)

    def run(self):

        self.printparameters()  # Print the parameters

        self.prepareDirectories()  # Prepare the directories

        train_data, dataset, label = self.getDataset()  # Get the dataset
        self.setDDPMModels()  # Set the models
        print("DDPM Models Loaded")

        model_classifier = None
        if self.noisy_classification_model:
            model_classifier = torch.load(f"./models/classifier model/noisy {self.loadData} classifier E1000.pt").to(device)
            print(f"Noisy {self.loadData} Classifier Loaded\n\n")
        else:
            model_classifier = torch.load(f"./models/classifier model/{self.loadData} classifier E1000.pt").to(device)
            print(f"{self.loadData} Classifier Loaded\n\n")

        if self.generate_novel_data:  # Create the directories for the data
            self.generateData(dataset)  # Generate the data

        if self.classify_single_epoch:
            input_predictions, recon_predictions, input_labels, recon_labels = self.classification(model_classifier, self.ddpm_model_epochs)
            self.plotPointGraph(input_predictions, input_labels, recon_predictions, recon_labels)
            self.plotBarChart(input_predictions, input_labels, recon_predictions, recon_labels)

        if self.plotGrid:
            self.process_all_epochs(model_classifier)


if __name__ == "__main__":
    main()
