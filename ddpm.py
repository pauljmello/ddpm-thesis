#  Author: Paul-Jason Mello
#  Date: February 16th, 2024
#  Version 2.4

#  Misc. Libraries
import io
import os
import re
import sys
import time
import datetime
import math

#  General Libraries
from tqdm import tqdm
import numpy as np
import matplotlib
import matplotlib.style as mplstyle
from matplotlib import pyplot as plt
from matplotlib.animation import ArtistAnimation

#  Torch Libraries
import torch
import torch.nn.functional as func
import torch.utils.data
import torchvision
from torch import optim
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader, SubsetRandomSampler

#  Image Libraries
from image_transform import RGBTransform, GrayscaleTransform, ConvertToImage
#  Model Libraries
from unet import unet

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.cuda.set_device(torch.cuda.current_device())

#  Set Seed for Reproducibility
#  seed = 3407  # https://arxiv.org/abs/2109.08203
seed = np.random.randint(0, 1_000_000)  # random seed
np.random.seed(seed)
torch.manual_seed(seed)

matplotlib.use("Agg")
mplstyle.use(["dark_background", "fast"])

# Append mine class to source
sys.path.append("mine/src")

# Diffusion Probabilistic Models Algorithms
#
# Algorithm 1 Training
# 1:  repeat
# 2:      x_0 ∼ q(x_0)
# 3:      t ∼ Uniform({1, . . . , T })
# 4:       ∼ N (0, I)
# 5:      Take gradient descent step on
#             ∇θ ||  − _θ * (√ (̄α_t) * x_0 + √(1−α_t) * , t) || ^ 2
# 6: until converged
#
#
# Algorithm 2 Sampling
# 1: xT ∼ N (0, I)
# 2: for t = T, . . . , 1 do
# 3:      z ∼ N (0, I) if t > 1, else z = 0
# 4:      x_t−1 = 1/(√(α_t)) * (x_t − (1−α_t)/√(1−α_t) * _θ(x_t, t)) + σtz
# 5: end for
# 6: return x_0

# TODO Optimization Measures: Best speed optimization would be reducing sampling of data to only sample single image instead of batch size quantity of images.
# TODO Optimization Measures: Above measure must probably adjust UNET num_groups.

def main():
    # Base diffusion training data is collected according to the average over 10 Epochs @ 1000 steps on RTX 4090
    # Model Parameters: model_channels=32, num_res_blocks=2, attention_resolutions=(8, 16), dropout=0,
    #                   conv_resample=True, num_heads=2, FreeU=self.FreeU, b1=1.1, b2=1.2, s1=1.0, s2=1.0
    # Dataset       | Batch Size |  Loss  | Time per Epoch  || Sampling Speed
    # MNIST         |    512     | 0.0273 | 0:09:10 Seconds || ~1:30:00 Minutes
    # Fashion MNIST |    512     | 0.0273 | 0:09:10 Seconds || ~1:30:00 Minutes
    # CIFAR10       |    512     | 0.0353 | 1:02:00 Seconds || ~5:30:00 Minutes
    # CelebA        |    512     | 0.0353 | 1:02:00 Seconds || ~1:30:00 Minutes  (Not Working (Mine Forward))

    lr = 2e-4
    epochs = 1000  # Epochs

    batch_size = 512  # Batch Size, Test higher Batch Sizes

    steps = 1000
    t_steps = steps

    # TODO fix "Not Working" datasets
    dataset = "mnist"  # "mnist", "fashion-mnist", "cifar10"  | Not Working: "celeba" (add MINE integration)

    # Noise Scheduling should be determined by image size. See https://arxiv.org/abs/2301.10972 for more details.
    # https://arxiv.org/abs/2102.09672, https://arxiv.org/abs/2212.11972
    # "auto" is the best because it adjusts based on the datasets image size.
    # TODO: fix "Not Working" noise schedulers
    schedule = "linear"  # "auto", "linear", "cosine", "sigmoid", "geometric"   | Not Working: "snr"  | "auto" > "linear" > "cosine" > "sigmoid" > "geometric"

    # TODO: fix "Not Working" loss metrics
    loss_metric = "MSE"  # Working: "MSE", "L1", "PSNR" | Not Working: "SCORE", "KL", "ELBO", "DSM", "ADSM", "NLL"        | "MSE" > "L1" > "PSNR"

    # Score based loss functions ("SCORE", "ELBO", "DSM", "ADSM") might work with X0 / V predictions. Score has been shown to create something w/ V.
    # predict_noise: standard diffusion model prediction objective for training
    # predict_X0: predict the initial image x_0
    # predict_V: predict the initial image x_0 and the noise vector v_0
    prediction_objective = "predict_noise"  # Options: "predict_noise", "predict_X0", "predict_V"   | "predict_noise" > "predict_X0" > "predict_V"

    # Empowerment Factor
    empower = 0.0  # 0.0 = No Empowerment, 1.0 = Full Empowerment

    # Efficient diffusion Training via Min-SNR Weighting Strategy: https://arxiv.org/abs/2303.09556
    # Works better on predict_noise and predict_X0
    weighted_snr = False

    # Enable Fast Fourier Transform (FFT) for image synthesis
    # FreeU: https://arxiv.org/abs/2309.11497
    FreeU = False

    # Pretrained Model
    pretrained_model = True

    # Number of Samples to take from the diffusion process for sequence generation
    images_per_sample = 5

    # full ddpm_epoch = [1, 5, 10, 25, 50, 100, 250, 500, 750, 1000]
    # full ratios = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1500, 2000, 3000, 4000, 5000]

    # for epoch in ddpm_epochs:
    #     for num in ratios:
    #         steps = num
    #         t_steps = steps
    #         ddpm_instance = ddpm(loss_metric, dataset, steps, t_steps, epoch, batch_size, lr, images_per_sample, schedule, prediction_objective, weighted_snr, empower, FreeU, pretrained_model)
    #         ddpm_instance.run()

    DDPM = ddpm(loss_metric, dataset, steps, t_steps, epochs, batch_size, lr, images_per_sample, schedule, prediction_objective, weighted_snr, empower, FreeU, pretrained_model)
    DDPM.run()

class ddpm:
    def __init__(self, loss_metric, dataset, steps, t_steps, epochs, batch_size, lr, images_per_sample, schedule, prediction_objective, weighted_snr, empower, FreeU, pretrained_model):
        super(ddpm).__init__()

        self.cum_diff = 0

        self.single_image = True

        self.ddpm_model = None
        self.mine_model = None

        self.save_model = False
        self.model_checkpoints = [1, 5, 10, 25, 50, 100, 250, 500, 750, 1000]

        self.lr = lr
        self.steps = steps
        self.FreeU = FreeU
        self.epochs = epochs + 1  # + 1 for readability
        self.empower = empower
        self.t_steps = t_steps
        self.load_data = dataset
        self.scheduler = schedule
        self.batch_size = batch_size
        self.snr_weight = weighted_snr
        self.loss_metric = loss_metric

        self.pretrained_model = pretrained_model  # Use Pretrained Model
        self.prediction_objective = prediction_objective

        self.plot_every = 10
        self.images_per_sample = images_per_sample
        self.series_frequency = int(self.t_steps / self.images_per_sample)

        self.collect_mi_variance_charts = False
        self.collect_mi_trajectory_charts = True
        self.collect_conditional_mi_charts = True
        self.collect_incremental_information = True

        self.generate_gif = True  # Generate Gif
        self.collect_sequence_plots = True  # Collect Sequence Plots (Distribution / Image Series)
        self.generate_example_denoising_images = True

        self.minimal_data = False  # Use Minimal Data Subset, for testing purposes
        self.minimal_data_size = 10_000  # Size of minimal_data Subset

        self.current_epoch = 0

        self.loss = 0.0
        self.loss_list = []

        if dataset == "mnist" or dataset == "fashion-mnist":
            self.channel_count = 1
            self.image_size = 28
            self.channel_mult = (1, 2)
            self.attention_resolution = (8, 16)
            if schedule == "auto":
                self.scheduler = "linear"
        elif dataset == "cifar10":
            self.channel_count = 3
            self.image_size = 32
            self.channel_mult = (1, 2, 4)
            self.attention_resolution = (8, 16)
            if schedule == "auto":
                self.scheduler = "linear"
        elif dataset == "celeba":
            self.channel_count = 3
            self.image_size = 64
            self.channel_mult = (1, 2, 3, 4)
            self.attention_resolution = (8, 16, 32)
            if schedule == "auto":
                self.scheduler = "sigmoid"

        self.Beta = self.getSchedule(self.scheduler).to(device)  # Noise Schedule
        Sqrt_Sigma = self.Beta  # Sigma Squared
        Alpha = 1.0 - self.Beta  # Alpha Schedule

        self.Alpha_Bar = torch.cumprod(Alpha, dim=0)  # Product Value of Alpha
        self.Sqrt_Alpha_Cumprod = torch.sqrt(self.Alpha_Bar)  # Square Root of Product Value of Alpha
        Alpha_Cumprod_Previous = func.pad(self.Alpha_Bar[:-1], (1, 0), value=1.0)  # Previous Product Value of Alpha   # Never forget the two months I lost to this bug
        self.Sqrt_1_Minus_Alpha_Cumprod = torch.sqrt(1.0 - self.Alpha_Bar)  # Square Root of 1 - Product Value of Alpha
        Log_one_minus_Alpha_Cumprod = torch.log(1.0 - self.Alpha_Bar)  # Log of 1 - Product Value of Alpha

        self.Sqrt_Recipricol_Alpha_Cumprod = torch.sqrt(1.0 / self.Alpha_Bar)  # Square Root of Reciprocal of Product Value of Alpha
        self.Sqrt_Recipricol_Alpha_Cumprod_Minus_1 = torch.sqrt(1.0 / self.Alpha_Bar - 1)  # Square Root of Reciprocal of Product Value of Alpha - 1

        self.Posterior_Variance = self.Beta * (1.0 - Alpha_Cumprod_Previous) / (1.0 - self.Alpha_Bar)  # Var(x_{t-1} | x_t, x_0)
        self.Posterior_Log_Variance_Clamp = torch.log(self.Posterior_Variance.clamp(min=1e-20))  # Log of Var(x_{t-1} | x_t, x_0)
        self.Posterior1 = (self.Beta * torch.sqrt(Alpha_Cumprod_Previous) / (1.0 - self.Alpha_Bar))  # 1 / (Var(x_{t-1} | x_t, x_0))
        self.Posterior2 = (1.0 - Alpha_Cumprod_Previous) * torch.sqrt(Alpha) / (1.0 - self.Alpha_Bar)  # (1 - Alpha_{t-1}) / (Var(x_{t-1} | x_t, x_0))

        # Efficient Diffusion Training via Min-SNR Weighting Strategy: https://arxiv.org/abs/2303.09556
        gamma = 5
        if self.snr_weight:
            snr = self.Alpha_Bar / (1 - self.Alpha_Bar)
            self.snrClip = snr.clone()
            self.snrClip.clamp_(max=gamma)

####################################### INITIALIZATION #######################################
    def printSystemDynamics(self):
        print("Cuda: ", torch.cuda.is_available())
        print("Device Count: ", torch.cuda.device_count())
        print("Device: ", device)
        print("Device Name: ", torch.cuda.get_device_name(device))
        print("\nImages Series Count: ", self.images_per_sample)
        print("Steps Between Images: ", self.series_frequency)

    @staticmethod
    def printModelInfo(model):
        print("\nModel Info:")
        print("\tModel: ", model)
        print("\tModel Parameters: ", sum(p.numel() for p in model.parameters() if p.requires_grad))

    def printTrainingInfo(self):
        print("\nHyperparameters:")
        print("\tData: ", self.load_data)
        print("\tEpochs: ", self.epochs - 1)
        print("\tSteps: ", self.steps)
        print("\tBatch Size: ", self.batch_size)
        print("\tLearning Rate: ", self.lr)
        print("\tScheduler: ", self.scheduler)
        print("\tLoss Metric: ", self.loss_metric)
        print("\tPrediction Objective: ", self.prediction_objective)

    def folderCheck(self, folder_path):
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

    def generateFolders(self):
        main_folders = ['models', 'datasets', 'images']
        for folder in main_folders:
            self.folderCheck(folder)

        models_subfolders = ['ddpm model', 'mine model', 'classifier model']
        for subfolder in models_subfolders:
            path = os.path.join('models', subfolder)
            self.folderCheck(path)

            if subfolder == 'ddpm model':
                ddpm_path = os.path.join(path, self.load_data)
                scheduler_path = os.path.join(ddpm_path, self.scheduler)
                steps_path = os.path.join(scheduler_path, str(self.steps))
                self.folderCheck(steps_path)

            elif subfolder == 'mine model':
                mine_model_path = os.path.join(path, self.load_data)
                self.folderCheck(mine_model_path)

        images_path = os.path.join('images', self.load_data, self.scheduler)
        self.folderCheck(images_path)

        images_subfolders = ['charts', 'noise', 'sequence plots']
        for subfolder in images_subfolders:
            subfolder_path = os.path.join(images_path, subfolder)
            self.folderCheck(subfolder_path)

            if subfolder == 'charts':
                for chart_folder in ['mi variance test', 'mi trajectory test', 'conditional mi test', 'mi increment test', 'noise']:
                    chart_path = os.path.join(subfolder_path, chart_folder)
                    self.folderCheck(chart_path)

            elif subfolder == 'sequence plots':
                for sequence_folder in ['distribution series', 'image series']:
                    sequence_path = os.path.join(subfolder_path, sequence_folder)
                    self.folderCheck(sequence_path)

    def getDataset(self):
        trainData = None
        grey = GrayscaleTransform(self.image_size, self.channel_count)
        RGB = RGBTransform(self.image_size)

        # testData = torchvision.datasets.MNIST(root=".", download=True, train=False, transform=dataTransformation)
        if self.load_data == "mnist":
            trainData = torchvision.datasets.MNIST(root="./datasets", download=True, train=True, transform=grey)
        elif self.load_data == "fashion-mnist":
            trainData = torchvision.datasets.FashionMNIST(root="./datasets", download=True, train=True, transform=grey)
        elif self.load_data == "cifar10":
            trainData = torchvision.datasets.CIFAR10(root="./datasets", download=True, train=True, transform=RGB)
        elif self.load_data == "celeba":
            trainData = torchvision.datasets.CelebA(root="./datasets", download=True, transform=RGB)

        if self.minimal_data:
            subset = list(np.random.choice(np.arange(0, len(trainData)), self.minimal_data_size, replace=False))
            DataSet = DataLoader(trainData, batch_size=self.batch_size, pin_memory=True, num_workers=4, pin_memory_device="cuda", persistent_workers=True, sampler=SubsetRandomSampler(subset))
        else:
            DataSet = DataLoader(trainData, batch_size=self.batch_size, shuffle=True, pin_memory=True, num_workers=4, pin_memory_device="cuda",  persistent_workers=True)

        if self.load_data == "celeba":
            Label = trainData.identity
        else:
            Label = trainData.targets
        return trainData, DataSet, Label

    def getDDPMModel(self):
        model = None
        pattern = re.compile(r'ddpm E(\d+)\.pt$')

        if self.pretrained_model:
            exact_model_path = f"./models/ddpm model/{self.load_data}/{self.scheduler}/{self.steps}/ddpm E{self.epochs}.pt"
            if os.path.isfile(exact_model_path):
                print(f"Loading DDPM Model from: {exact_model_path}")
                self.current_epoch = self.epochs
                model = torch.load(exact_model_path, map_location=torch.device('cuda'))  # Assuming CUDA is used
                return model

            # If exact match not found, look for the closest model
            for epoch in range(min(10000, self.epochs), 0, -1):
                model_path = f"./models/ddpm model/{self.load_data}/{self.scheduler}/{self.steps}/ddpm E{epoch}.pt"
                if os.path.isfile(model_path):
                    print(f"Loading DDPM Model from: {model_path}")
                    model = torch.load(model_path, map_location=torch.device('cuda'))  # Assuming CUDA is used
                    self.current_epoch = self.epochs
                    match = pattern.search(model_path)
                    if match:
                        self.current_epoch = int(match.group(1))
                    break
        else:
            if model is None:
                model = unet(
                    in_channels=self.channel_count,  # Channel Count                      |   Number of Channels                                  |  (int)    {1: Grayscale, 3: RGB}
                    model_channels=32,  # Number of Channels in the Model               |   Tradeoff: Large Complexity vs. Lower Performance    |  (int)    {96, 128, 192, 256, ...}
                    out_channels=self.channel_count,  # Channel Count                     |   Number of Channels                                  |  (int)    {1: Grayscale, 3: RGB}
                    num_res_blocks=1,  # Number of Residual Blocks                      |   Tradeoff: Large Complexity vs. Lower Performance    |  (int)    {2, 3, 4, ...}
                    attention_resolutions=self.attention_resolution,  # Attention Resolutions   |   Capture spatial information                         |  (tuple)  {8, 16, 32, 64, ...}
                    dropout=0,  # Dropout Rate                                          |   Dropout Rate                                        |  (float)  (0, 1)
                    channel_mult=self.channel_mult,  # Channel Multiplier               |   Number of Channels Multiplied at each Layer         |  (tuple)  {1, 2, 3, 4, ...}
                    conv_resample=True,  # Convolutional Resampling                     |   Inc/Dec Spatial Resolution using Convolution        |  (bool)   {True, False}
                    num_heads=2,  # Number of Attention Heads                           |   Number of Attention Heads in Multi-Head Attention   |  (int)    {4, 8, 16, 32, ...}

                    # TODO: Mess with these Scaling Factors (Try OTHER Datasets)
                    FreeU=self.FreeU,

                    # If 1 < b features are amplified. If 0 < b < 1 features are dampened. If b = 1 features are unaffected.
                    b1=1.0,  # Scaling Factor of Channel Importance                     |   Backbone Suppression of High-Frequency Components   |  (float)  {1.0 ≤ b1 ≤ 1.2}
                    b2=1.0,  # Scaling Factor of Channel Importance                     |   Backbone Suppression of High-Frequency Components   |  (float)  {1.2 ≤ b2 ≤ 1.6}

                    # If 1 < s features are amplified and smoothed. If 0 < s < 1 features are more tailed or noisy. If s = 1 features are unaffected.
                    s1=1.0,  # Scaling Factor creates mask to retain central features   |   Attenuate the skip feature map                      |  (float)  {s1 ≤ 1.0}
                    s2=1.0,  # Scaling Factor creates mask to retain central features   |   Attenuate the skip feature map                      |  (float)  {s2 ≤ 1.0}
                )

        model = model.to(device)
        # model = nn.DataParallel(model)  # Parallelize Model Multi-GPU setup
        optimizer = optim.Adam(model.parameters(), lr=self.lr, eps=1e-8)
        return model, optimizer

    def getMINEModel(self):
        mine_model = None
        mine_model_dir = f"./models/mine model/{self.load_data}"
        mnist_model_path = "./models/mine model/mnist/mnist MI.pt"
        cifar_model_path = "./models/mine model/cifar10/cifar10 MI.pt"
        highest_epoch = 0
        if os.path.isdir(mine_model_dir):
            for filename in os.listdir(mine_model_dir):
                match = re.match(r"MI E(\d+).pt", filename)
                if match:
                    epoch = int(match.group(1))
                    if epoch > highest_epoch:
                        highest_epoch = epoch
                        mine_model_path = os.path.join(mine_model_dir, f"MI E{epoch}.pt")
                        mine_model = torch.load(mine_model_path).to(device)
        if mine_model is not None:
            print(f"Loading Mine Model from: {mine_model_path}")
            return mine_model
        print(f"No pretrained mine model found for {self.load_data}. {self.channel_count} channel model will be used.")
        if self.channel_count == 1:
            print("Loading MNIST MI model...")
            return torch.load(mnist_model_path).to(device)
        elif self.channel_count == 3:
            print("Loading CIFAR MI model...")
            return torch.load(cifar_model_path).to(device)
        else:
            raise ValueError("Unsupported number of channels for the mine model.")

    def saveModel(self):
        if self.pretrained_model:
            print("\nPretrained Model Used. Model will Not be Saved.")
        else:
            print("\n\t\t...Saving Model...")
            if not self.minimal_data:
                torch.save(self.ddpm_model, f"models/ddpm model/{self.load_data}/{self.scheduler}/{self.steps}/ddpm E{self.current_epoch}.pt")
            else:
                torch.save(self.ddpm_model, f"models/ddpm model/{self.load_data}/{self.scheduler}/{self.steps}/minimal ddpm E{self.current_epoch}.pt")

####################################### INITIALIZATION #######################################

####################################### SCHEDULER #######################################
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
        x = torch.linspace(0, self.t_steps, self.steps + 1, device=device)
        y = torch.cos(((x / self.t_steps) + 0.01) / (1 + 0.01) * torch.pi * 0.5) ** 2
        z = y / y[0]
        return torch.clip((1 - (z[1:] / z[:-1])), 0.0001, 0.9999).to(device)

    def sigmoid_schedule(self, start, end):
        sequence = torch.linspace(0, self.t_steps, self.steps + 1, dtype=torch.float32, device=device) / self.t_steps
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

####################################### SCHEDULER #######################################

####################################### LOSS #######################################
    def getLoss(self, XT, prediction_noise, target, t, snr):
        loss = None
        target = target.to(torch.float16)
        prediction_noise = prediction_noise.to(torch.float16)
        if self.loss_metric == "MSE":
            loss = func.mse_loss(target, prediction_noise)
        elif self.loss_metric == "L1":
            loss = func.l1_loss(target, prediction_noise)
        elif self.loss_metric == "NLL":
            loss = func.nll_loss(target, prediction_noise)
        elif self.loss_metric == "PSNR":
            loss = self.psnr_loss(target, prediction_noise)
        elif self.loss_metric == "KL":
            loss = self.kl_divergence_loss(target, prediction_noise, t)
        elif self.loss_metric == "ELBO":
            loss = self.elbo_loss(target, prediction_noise, t)
        elif self.loss_metric == "DSM":
            loss = self.denoising_score_matching_loss(XT, prediction_noise, t)
        elif self.loss_metric == "ADSM":
            loss = self.anneal_dsm_score_loss(XT, t)
        elif self.loss_metric == "SCORE":
            loss = self.score_loss(target, prediction_noise)

        if self.snr_weight:
            loss = torch.mean(loss * self.getExtract(snr, t, loss.shape))

        return loss

    def denoising_score_matching_loss(self, XT, prediction_noise, t):
        XT.requires_grad_(True)
        perturbed_x = XT + prediction_noise.detach()
        with autocast():
            score_x = self.ddpm_model(XT.float(), t)
            perturbed_score_x = self.ddpm_model(perturbed_x.float(), t)
        score_diff = perturbed_score_x - score_x
        loss = 0.5 * func.mse_loss(score_diff, score_x)
        return loss

    # Num_time_steps shou;d be number of classes, but I've changed it to number of timesteps because I changed the original code
    # where labels were used instead of t timesteps.
    def anneal_dsm_score_loss(self, XT, t, sigma_begin=1, sigma_end=0.01, anneal_power=2.0):
        sigmas = torch.tensor(np.exp(np.linspace(np.log(sigma_begin), np.log(sigma_end), self.steps))).float().to(device)
        used_sigmas = sigmas[t].view(XT.shape[0], *([1] * len(XT.shape[1:])))
        perturbed_samples = XT + torch.randn_like(XT) * used_sigmas
        target = - 1 / (used_sigmas ** 2) * (perturbed_samples - XT)
        scores = self.ddpm_model(perturbed_samples, t)
        target = target.view(target.shape[0], -1)
        scores = scores.view(scores.shape[0], -1)
        loss = 1 / 2. * ((scores - target) ** 2).sum(dim=-1) * used_sigmas.squeeze() ** anneal_power
        return loss.mean(dim=0)

    def kl_divergence_loss(self, epsilon_theta, target, t):
        posterior_mean, posterior_model_variance, _ = self.q_posterior_mean_variance(epsilon_theta, target, t)
        prior_mean, prior_model_variance, _ = self.p_prior_mean_variance(target, t)
        epsilon = 1e-7
        posterior_model_variance = torch.clamp(posterior_model_variance, min=epsilon)
        prior_model_variance = torch.clamp(prior_model_variance, min=epsilon)
        kl_divergence = self.kl_divergence(prior_mean, prior_model_variance, posterior_mean, posterior_model_variance)
        return kl_divergence.mean()

    def elbo_loss(self, X0, Xt, t):
        posterior_mean, posterior_variance = self.p_posterior_mean_variance(X0, Xt, t)
        prior_mean, prior_variance = torch.zeros_like(X0), torch.ones_like(X0)
        log_likelihood = torch.sum(self.log_likelihood(X0, Xt, t))
        kl_div = torch.sum(self.kl_divergence(posterior_mean, posterior_variance, prior_mean, prior_variance))
        elbo = log_likelihood - kl_div  # ELBO = E[log p(x|z, t)] - D_KL(q(z|x, t) || p(z))
        return -elbo

    def psnr_loss(self, epsilon_theta, target):
        pixelMax = 255.0 if self.channel_count == 3 else 1.0
        mse = torch.mean((epsilon_theta - target) ** 2)
        epsilon = 1e-10
        psnr = 20 * torch.log10(pixelMax / torch.sqrt(mse + epsilon))
        return -psnr

    def score_loss(self, X0, prediction_noise):
        true_score = X0 - prediction_noise
        mse_loss = func.mse_loss(prediction_noise, true_score.half())
        return mse_loss

    ####################################### LOSS #######################################

    ####################################### OBJECTIVE FUNCTION #######################################
    def getTargetObjective(self, X0, noise, t):
        target = snr = None
        gamma = 5
        if self.prediction_objective == "predict_noise":
            if self.snr_weight:
                snr = self.Alpha_Bar / (1 - self.Alpha_Bar)
                snr = torch.min(snr, torch.tensor(gamma))
                snr = torch.min(gamma / snr, torch.tensor(1.0))
            target = noise
        elif self.prediction_objective == "predict_X0":
            if self.snr_weight:
                snr = self.snrClip
            target = X0
        elif self.prediction_objective == "predict_V":
            if self.snr_weight:
                snr = self.Alpha_Bar / (1 - self.Alpha_Bar)
                snr = snr / (snr + 1)
            target = self.pred_V(X0, t, noise)
        return target, snr

    def pred_X0_from_XT(self, Xt, noise, t):  # p(x_{t-1} | x_t)
        return self.getExtract(self.Sqrt_Recipricol_Alpha_Cumprod, t, Xt.shape) * Xt - self.getExtract(self.Sqrt_Recipricol_Alpha_Cumprod_Minus_1, t, Xt.shape) * noise  # Sample from p(x_{t-1} | x_t)

    def pred_V(self, X0, t, noise):
        return self.getExtract(self.Sqrt_Alpha_Cumprod, t, X0.shape) * noise - self.getExtract(self.Sqrt_1_Minus_Alpha_Cumprod, t, X0.shape) * X0  # Sample from q(V | X0)

    ####################################### OBJECTIVE FUNCTION #######################################

    ####################################### SAMPLE #######################################
    # TODO IDEA: Implement a graph which shows the MI of the MINE model stretched over time, and under the associated image series.
    # TODO Above Example:
    # A series plot of two rows where the top row is the MI through time from forward and backward process and the bottom is the respective images at an interval of timesteps.
    def getSampleImages(self, batched_labels, X0, noise, prediction_noise, t):
        savePathMI = "images/" + str(self.load_data) + "/" + str(self.scheduler) + "/charts/mi trajectory test/Epoch = " + str(self.current_epoch) + ""
        savePathVar = "images/" + str(self.load_data) + "/" + str(self.scheduler) + "/charts/mi variance test/Epoch = " + str(self.current_epoch) + ""
        savePathIncMI = "images/" + str(self.load_data) + "/" + str(self.scheduler) + "/charts/mi increment test/Epoch = " + str(self.current_epoch) + ""
        savePathCondMI = "images/" + str(self.load_data) + "/" + str(self.scheduler) + "/charts/conditional mi test/Epoch = " + str(self.current_epoch) + ""
        savePathExpDenoise = "images/" + str(self.load_data) + "/" + str(self.scheduler) + "/manifold/E" + str(self.current_epoch) + "/T"+str(self.steps) + ""
        savePathTest = "images/" + str(self.load_data) + "/" + str(self.scheduler) + "/charts/test/"
        # savePathCat = "images/" + str(self.load_data) + "/" + str(self.scheduler) + "/charts/stacked processing/Epoch = " + str(self.current_epoch) + "" For Stacked Processing Only

        current_labels = batched_labels.clone().detach()
        X0_batch, current_labels_batch, current_labels_marginal_batch = self.sample_batch_from_X0_and_labels(X0, current_labels, self.batch_size)

        # Collect Process Trajectories Charts
        if self.collect_mi_trajectory_charts:
            print("\n\t\t...Generating Trajectories...")
            self.process_mi_trajectory(X0_batch, current_labels_batch, current_labels_marginal_batch, savePathMI, mi_method="mutual information")

        # Collect Conditional MI Charts
        if self.collect_conditional_mi_charts:
            print("\n\t\t...Generating Conditional MI Experiment...")
            self.process_conditional_mi(X0_batch, current_labels_batch, current_labels_marginal_batch, savePathCondMI, mi_method="conditional mutual information")

        # Collect Process Variance Charts
        if self.collect_mi_variance_charts:
            print("\n\t\t...Generating Variance...")
            self.process_mi_variance(X0_batch, current_labels_batch, current_labels_marginal_batch, savePathVar, mi_method="mutual information")
        
        # Collect Incremental Information Charts
        if self.collect_incremental_information:
            print("\n\t\t...Generating Incremental Charts...")
            self.process_incremental_information(X0_batch, current_labels_batch, current_labels_marginal_batch, savePathIncMI, mi_method="incremental information")

        # model.plot_empowerment_dynamics(savePath + " Model Empowerment Dynamics.jpg")

        if self.generate_example_denoising_images:
            # XT and current_labels_batch are properly synchronized | To doublecheck just generate timestep 100 images.
            # torchvision.utils.save_image(X0_batch, "images/" + str(self.load_data) + "/" + str(self.scheduler) + "/manifold XT.png")
            if not any(os.path.isfile(os.path.join(savePathExpDenoise, f'class_{label}', 'step_0.png')) for label in range(10)):
                XT = self.simpleForward(X0_batch)
                self.saveDenoisingImageExamples(XT, current_labels_batch, savePathExpDenoise)

        if self.collect_sequence_plots:
            savePathRP = "images/" + str(self.load_data) + "/" + str(self.scheduler) + "/sequence plots/image series/Epoch = " + str(self.current_epoch) + ""
            print("\n\t\t...Generating Image Sequences...")
            final_noise = self.simpleForward(X0_batch)
            self.getNoiseHistogram(prediction_noise, final_noise, bins=128)
            self.plotDistributionHistogram(final_noise, t)
            _ = self.plotReverseProcessImageSynthesis(final_noise, savePathRP)

    def process_mi_trajectory(self, X0_batch, current_labels_batch, current_labels_marginal_batch, savePath, mi_method):
        plt.figure(figsize=(10, 10))
        self.saveImage(X0_batch, f'{savePath} A Input.jpg')

        mine_fp_mi, final_noise = self.trajectory(X0_batch, current_labels_batch, current_labels_marginal_batch, mi_method, reverse=False, description=f'Forward Process')
        self.plotGraph(arr=mine_fp_mi, label="MINE Predicted MI", x_label="Timestep", y_label="Mutual Information", title=f'MINE Forward Process', savePath=f'{savePath} B FP MINE.jpg', reverse=False)
        plt.figure(figsize=(10, 10))
        self.saveImage(final_noise, f'{savePath} C Corrupted.jpg')

        mine_rp_mi, noise_recon = self.trajectory(final_noise, current_labels_batch, current_labels_marginal_batch, mi_method, reverse=True, description=f'Reverse Process')
        self.plotGraph(arr=mine_rp_mi, label="MINE Predicted MI", x_label="Timestep", y_label="Mutual Information", title=f'MINE Reverse Process, start from FP output corruption', savePath=f'{savePath} E RP MINE Noise.jpg', reverse=True)
        plt.figure(figsize=(10, 10))
        self.saveImage(noise_recon, f'{savePath} F Reconstructed Noise.jpg')

    def process_conditional_mi(self, X0_batch, current_labels_batch, current_labels_marginal_batch, savePath, mi_method):
        plt.figure(figsize=(10, 10))
        self.saveImage(X0_batch, f'{savePath} A Input.jpg')

        mine_fp_mi, final_noise = self.trajectory(X0_batch, current_labels_batch, current_labels_marginal_batch, mi_method, reverse=False, description=f'Forward Process Expiraments')
        self.plotGraph(arr=mine_fp_mi, label="MINE Predicted MI", x_label="Timestep", y_label="Conditional Mutual Information", title=f'Experiment: MINE Forward Process, Conditional Mutual Information', savePath=f'{savePath} B FP MINE.jpg', reverse=False)
        plt.figure(figsize=(10, 10))
        self.saveImage(final_noise, f'{savePath} C Corrupted.jpg')

        mine_rp_mi, noise_recon = self.trajectory(final_noise, current_labels_batch, current_labels_marginal_batch, mi_method, reverse=True, description=f'Reverse Process Expiraments')
        self.plotGraph(arr=mine_rp_mi, label="MINE Predicted MI", x_label="Timestep", y_label="Conditional Mutual Information", title=f'Experiment: MINE Reverse Process, Conditional Mutual Information', savePath=f'{savePath} E RP MINE Noise.jpg', reverse=True)
        plt.figure(figsize=(10, 10))
        self.saveImage(noise_recon, f'{savePath} F Reconstructed Noise.jpg')

    def process_incremental_information(self, X0_batch, current_labels_batch, current_labels_marginal_batch, savePath, mi_method):
        plt.figure(figsize=(10, 10))
        self.saveImage(X0_batch, f'{savePath} A Input.jpg')

        self.cum_diff = 0
        mine_fp_mi, final_noise = self.trajectory(X0_batch, current_labels_batch, current_labels_marginal_batch, mi_method, reverse=False, description=f'Forward Process')
        self.plotGraph(arr=mine_fp_mi, label="MINE Prediction", x_label="Timestep", y_label="Incremental Information", title=f'MINE FP Incremental Information Calculation Chart', savePath=f'{savePath} B FP MINE.jpg', reverse=False)
        plt.figure(figsize=(10, 10))
        self.saveImage(final_noise, f'{savePath} C Corrupted.jpg')

        self.cum_diff = 0
        mine_rp_mi, noise_recon = self.trajectory(final_noise, current_labels_batch, current_labels_marginal_batch, mi_method, reverse=True, description=f'Reverse Process')
        self.plotGraph(arr=mine_rp_mi, label="MINE Prediction", x_label="Timestep", y_label="Incremental Information", title=f'MINE RP Incremental Information Calculation Chart', savePath=f'{savePath} E RP MINE Noise.jpg', reverse=True)
        plt.figure(figsize=(10, 10))
        self.saveImage(noise_recon, f'{savePath} F Reconstructed Noise.jpg')

    def process_mi_variance(self, X0_batch, current_labels_batch, current_labels_marginal_batch, savePath, mi_method):
        plt.figure(figsize=(10, 10))
        self.saveImage(X0_batch, f'{savePath} A Input.jpg')

        mine_fp_mi = [[] for _ in range(5)]
        mine_rp_mi = [[] for _ in range(5)]
        final_noise = None
        recon_noise = None

        for i in range(5):
            mine_fp_mi[i], final_noise = self.trajectory(X0_batch, current_labels_batch, current_labels_marginal_batch, mi_method, reverse=False, description=f'Forward Process Variance')
        fp_avg_variance = self.getAverageVariance(*mine_fp_mi)
        self.plotGraphVariance(arr1=mine_fp_mi[0], arr2=mine_fp_mi[1], arr3=mine_fp_mi[2], arr4=mine_fp_mi[3], arr5=mine_fp_mi[4], avg_var=fp_avg_variance, avg_label="Avg. Var.",
                               x_label="Timestep", y_label="Mutual Information", title=f'MINE Forward Process Variance', savePath=f'{savePath} B FP MINE.jpg', reverse=False)
        plt.figure(figsize=(10, 10))
        self.saveImage(final_noise, f'{savePath} C Corrupted.jpg')

        for i in range(5):
            mine_rp_mi[i], recon_noise = self.trajectory(final_noise, current_labels_batch, current_labels_marginal_batch, mi_method, reverse=True, description=f'Reverse Process Variance')
        rp_avg_variance = self.getAverageVariance(*mine_rp_mi)
        self.plotGraphVariance(arr1=mine_rp_mi[0], arr2=mine_rp_mi[1], arr3=mine_rp_mi[2], arr4=mine_rp_mi[3], arr5=mine_rp_mi[4], avg_var=rp_avg_variance, avg_label="Avg. Var.",
                               x_label="Timestep", y_label="Mutual Information", title=f'MINE Reverse Process Variance', savePath=f'{savePath} E RP MINE Noise.jpg', reverse=True)
        plt.figure(figsize=(10, 10))
        self.saveImage(recon_noise, f'{savePath} F Reconstructed Noise.jpg')

    def getAverageVariance(self, arr1, arr2, arr3, arr4, arr5):
        avg_var = []
        for i in range(len(arr1)):
            avg_var.append((arr1[i] + arr2[i] + arr3[i] + arr4[i] + arr5[i]) / 5)
        return avg_var

####################################### SAMPLE #######################################

####################################### MINE #######################################
    def sample_batch_from_X0_and_labels(self, X0, current_labels, batch_size):
        data_len = len(X0)
        index_joint = np.random.choice(range(data_len), size=batch_size, replace=False)
        index_marginal = np.random.choice(range(data_len), size=batch_size, replace=False)
        X0_batch = X0[index_joint]
        current_labels_batch = current_labels[index_joint]
        current_labels_marginal_batch = current_labels[index_marginal]
        return X0_batch, current_labels_batch, current_labels_marginal_batch

    def prepare_mine_data(self, xt, prev, label, label_tilde):
        label_tensor = label.view(-1, 1).float()
        label_tilde_tensor = label_tilde.view(-1, 1).float()
        if self.single_image:
            xt = xt[:1]
            prev = prev[:1]
            label_tensor = label_tensor[:1]
            label_tilde_tensor = label_tilde_tensor[:1]
        return xt, prev, label_tensor, label_tilde_tensor

    def get_mi_type(self, XT, prev, label, label_tilde, mi_method, reverse):
        if mi_method == 'mutual information':
            return self.get_mi_mine(XT, prev, label, label_tilde)
        elif mi_method == 'incremental information':
            return self.get_inc_mi_mine(XT, prev, label, label_tilde, reverse)
        elif mi_method == 'conditional mutual information':
            return self.get_conditional_mi_mine(XT, prev, label, label_tilde)

    def get_mi_mine(self, xt, prev, labels, shuffled_labels):
        xt, prev, label, label_tilde = self.prepare_mine_data(xt, prev, labels, shuffled_labels)

        xt_cat = torch.cat((xt, xt), dim=1)

        T_joint = self.mine_model(xt_cat, label)
        T_marginal = torch.exp(self.mine_model(xt_cat, label_tilde))

        mi_tensor = T_joint - torch.log(T_marginal + 1e-10)
        mi_tensor /= math.log(2)
        return torch.mean(mi_tensor).item()

    def get_inc_mi_mine(self, xt, prev, labels, shuffled_labels, reverse):
        xt, prev, label, label_tilde = self.prepare_mine_data(xt, prev, labels, shuffled_labels)

        channel_tensor = torch.cat((xt, prev), dim=1)

        T_joint = self.mine_model(channel_tensor, label)
        T_marginal = torch.exp(self.mine_model(channel_tensor, label_tilde))

        mi_tensor = T_joint - torch.log(T_marginal + 1e-10)
        mi_tensor /= math.log(2)
        cur_val = torch.mean(mi_tensor).item()

        prev_cat = torch.cat((prev, prev), dim=1)

        T_joint = self.mine_model(prev_cat, label)
        T_marginal = torch.exp(self.mine_model(prev_cat, label_tilde))

        mi_tensor = T_joint - torch.log(T_marginal + 1e-10)
        mi_tensor /= math.log(2)
        prev_val = torch.mean(mi_tensor).item()

        if reverse == False:
            mi_diff = prev_val - cur_val
        else:
            mi_diff = cur_val - prev_val
        self.cum_diff += mi_diff

        return self.cum_diff

    def get_conditional_mi_mine(self, xt, prev, labels, shuffled_labels):
        xt, prev, label, label_tilde = self.prepare_mine_data(xt, prev, labels, shuffled_labels)

        xt_concat = torch.cat((xt, xt), dim=1)
        prev_concat = torch.cat((prev, prev), dim=1)

        T_joint = self.mine_model(xt_concat, label)
        T_marginal = self.mine_model(prev_concat, label)
        expT = torch.exp(self.mine_model(xt_concat, label_tilde))

        cmi_tensor = T_joint - T_marginal - torch.log(expT.clamp(min=1e-10))
        cmi_tensor /= math.log(2)  # Normalize to bits
        return torch.mean(cmi_tensor).item()
    
####################################### MINE #######################################

####################################### IMAGES #######################################
    def saveImage(self, img, msg):
        convertToImage = ConvertToImage()
        if len(img.shape) == 4:
            img = img[0, :, :, :]

        if self.channel_count == 1:
            plt.imshow(convertToImage(img), cmap="gray")
        elif self.channel_count == 3:
            plt.imshow(convertToImage(img))
        plt.title("Loss = " + str("{:.6f}".format(self.loss)))
        plt.axis("off")
        plt.savefig(msg)
        plt.close()

    def saveImageSize(self, img, msg, size):
        convertToImage = ConvertToImage()
        if len(img.shape) == 4:
            img = img[0]
        image_data = convertToImage(img)
        if image_data.max() <= 1.0:
            image_data = (image_data * 255).astype(np.uint8)
        fig, ax = plt.subplots(figsize=(size / plt.rcParams['figure.dpi'], size / plt.rcParams['figure.dpi']), dpi=plt.rcParams['figure.dpi'])
        if self.channel_count == 1:
            ax.imshow(image_data[:, :, 0], cmap="gray")
        elif self.channel_count == 3:
            ax.imshow(image_data)
        plt.axis("off")
        plt.savefig(msg, bbox_inches='tight', pad_inches=0)
        plt.close(fig)

    @torch.no_grad()
    def saveDenoisingImageExamples(self, XT, labels, saveFolderPath):
        os.makedirs(saveFolderPath, exist_ok=True)
        [os.makedirs(os.path.join(saveFolderPath, f'class_{label}'), exist_ok=True) for label in range(10)]

        unique_label_indices = {label: None for label in range(10)}
        for i, label in enumerate(labels):
            label = label.item()
            if unique_label_indices[label] is None:
                unique_label_indices[label] = i

        for step in tqdm(reversed(range(self.steps)), desc="Saving Image Denoising Process"):
            XT = self.p_sample(XT, torch.full((self.batch_size,), step, device=device, dtype=torch.long))
            for label, index in unique_label_indices.items():
                x = XT[index]
                image_path = os.path.join(saveFolderPath, f'class_{label}', f'step_{step}.png')
                self.saveImageSize(x, image_path, self.image_size)
        return XT

####################################### IMAGES #######################################

####################################### PLOTTING #######################################

    def plotGraph(self, arr, label, x_label, y_label, title, savePath, reverse):
        plt.figure(figsize=(10, 10))
        plt.plot(arr, label=label, markersize=2)
        plt.legend()
        self.plotGraphHelper(x_label, y_label, title, savePath, reverse)

    def plotGraphHelper(self, x_label, y_label, title, savePath, reverse):
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.title(title)
        if reverse:
            plt.gca().invert_xaxis()
            # Fixing plots so reverse looks the same as forward plot padding
            padding = 0.05
            new_min = 0 - (self.steps - 0) * padding
            new_max = self.steps + (self.steps - 0) * padding
            plt.xlim(new_max, new_min)
        plt.savefig(savePath)
        plt.close()

    def diffusionSubplotting(self, img, step, ax):
        ax.set_title(f"Time = {step}")
        ax.axis("off")
        if self.channel_count == 1:
            img_data = (img[0].cpu().squeeze().numpy() + 1.0) / 2.0
            img_data = np.clip(img_data, 0, 1)  # Ensure data is in the range [0, 1]
            ax.imshow(img_data, cmap="gray")
        elif self.channel_count == 3:
            img_data = np.transpose((img.cpu().numpy() + 1.0) / 2.0, (1, 2, 0))
            img_data = np.clip(img_data, 0, 1)  # Ensure data is in the range [0, 1]
            ax.imshow(img_data)


    def plotGraphVariance(self, arr1, arr2, arr3, arr4, arr5, avg_var, avg_label, x_label, y_label, title, savePath, reverse=False):
        plt.figure(figsize=(10, 10))
        data_stack = np.stack((arr1, arr2, arr3, arr4, arr5))
        std_dev = np.std(data_stack, axis=0)
        lower_bound = avg_var - std_dev
        upper_bound = avg_var + std_dev
        plt.fill_between(range(len(avg_var)), lower_bound, upper_bound, color='#add8e6', alpha=0.5, label='Uncertainty')

        plt.plot(avg_var, label=avg_label, color='#e6bbad', alpha=1.0, markersize=7)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.title(title)
        plt.legend()

        if reverse:
            plt.gca().invert_xaxis()
            padding = 0.05
            new_min = 0 - (self.steps - 0) * padding
            new_max = self.steps + (self.steps - 0) * padding
            plt.xlim(new_max, new_min)

        plt.savefig(savePath)
        plt.close()

    @torch.no_grad()
    def getNoiseHistogram(self, prediction_noise, noise, bins):
        savePath = "images/" + str(self.load_data) + "/" + str(self.scheduler) + "/charts/noise/Epoch = " + str(self.current_epoch) + ""
        plt.figure(figsize=(7, 7))
        np_noise, np_prediction_noise, bins = self.getNormHistData(noise, prediction_noise, binCount=bins)
        plt.hist(np_noise, density=True, bins=bins, alpha=.60, label="True Noise")
        plt.hist(np_prediction_noise, density=True, bins=bins, alpha=.60, label="Prediction Noise")
        self.plotGraphHelper("Noise", "Frequency", "Noise Histogram", savePath, reverse=False)

    @staticmethod
    def getNormHistData(data_0, data_1, binCount):
        data_0 = data_0.cpu().detach().numpy().flatten()
        data_1 = data_1.cpu().detach().numpy().flatten()
        bin_range = (min(np.min(data_0), np.min(data_1)), max(np.max(data_0), np.max(data_1)))
        bin_interval = (bin_range[1] - bin_range[0]) / binCount
        bins = np.arange(bin_range[0], bin_range[1] + bin_interval, bin_interval)
        return data_0, data_1, bins

    def histogramPlotHelper(self, model, img, xlabel, ylabel, title, legend1, legend2, t, bins, ax):
        prediction = model(img, t)
        norm_img, norm_pred, bins = self.getNormHistData(img, prediction, binCount=bins)
        ax.hist(norm_img, density=True, bins=bins, alpha=.60, label=legend1)
        ax.hist(norm_pred, density=True, bins=bins, alpha=.60, label=legend2)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)

    @torch.no_grad()
    def plotDistributionHistogram(self, XT, t):
        num_subplots = (self.steps // self.series_frequency) + 1
        fig, axes = plt.subplots(nrows=1, ncols=num_subplots, figsize=((self.images_per_sample / 2) * 15, 5))
        XT = torch.randn_like(XT, device=device)
        params = {'model': self.ddpm_model, 'img': XT, 'xlabel': "Noise", 'ylabel': "Frequency", 'legend1': "Noise Distribution", 'legend2': "Predicted Distribution", 't': t, 'bins': 128, 'title': f"Distribution at Time = {self.steps} {self.scheduler}"}
        self.histogramPlotHelper(ax=axes[0], **params)
        for idx, step in enumerate(tqdm(reversed(range(0, self.steps, self.series_frequency)), desc="Plotting Distribution Histograms")):
            XT = self.p_sample(XT, torch.full((self.batch_size,), step, device=device, dtype=torch.long))
            params['title'] = f"Distribution at Time = {step} ({self.scheduler})"
            self.histogramPlotHelper(ax=axes[idx + 1], **params)
        buffer = io.BytesIO()
        plt.savefig(buffer, format='jpg')
        buffer.seek(0)
        save_path = f"images/{self.load_data}/{self.scheduler}/sequence plots/distribution series/Epoch = {self.current_epoch}.jpg"
        with open(save_path, 'wb') as f:
            f.write(buffer.read())
        plt.close()

    @torch.no_grad()
    def getDiffuionGif(self, XT):
        convertToImage = ConvertToImage()
        fig, ax = plt.subplots()
        imgs = []
        display_mode = 'gray' if self.channel_count == 1 else 'rgb'
        for step in tqdm(reversed(range(0, self.steps)), desc="Generating Diffusion Process Gif"):
            XT = self.p_sample(XT, torch.full((self.batch_size,), step, device=device, dtype=torch.long))
            img_np = convertToImage(XT[0])
            if display_mode == 'gray':
                imgs.append([ax.imshow(img_np, cmap="gray")])
            else:
                imgs.append([ax.imshow(img_np)])
        imgs.extend([imgs[-1]] * 500)
        ax.set_title("" + str(self.load_data) + " diffusion")
        ax.axis("off")
        animate = ArtistAnimation(fig, imgs, interval=30, blit=True, repeat_delay=50, repeat=True)
        animate.save("images/" + str(self.load_data) + "/" + str(self.scheduler) + "/" + str(self.load_data) + " diffusion.gif")
        plt.close(fig)

    ####################################### PLOTTING #######################################

    ####################################### SAMPLE LOOPS #######################################
    @torch.no_grad()
    def simpleForward(self, XT):
        for step in tqdm(range(0, self.steps)):
            XT = self.p_sample(XT, torch.full((self.batch_size,), step, device=device, dtype=torch.long))
        return XT

    @torch.no_grad()
    def simpleReversed(self, XT):
        for step in tqdm(reversed(range(0, self.steps))):
            XT = self.p_sample(XT, torch.full((self.batch_size,), step, device=device, dtype=torch.long))
        return XT

    @torch.no_grad()
    def trajectory(self, init_image, label, label_tilde, mi_method, reverse, description):
        XT = init_image.clone()
        mine_mi = [None] * self.steps
        steps_range = range(self.steps - 1, -1, -1) if reverse else range(0, self.steps)

        for step in tqdm(steps_range, desc=description):
            prev = XT
            XT = self.p_sample(XT, torch.full((self.batch_size,), step, device=XT.device, dtype=torch.long))
            mine_mi[step] = self.get_mi_type(XT, prev, label, label_tilde, mi_method, reverse)
        return mine_mi, XT
    
    @torch.no_grad()
    def plotForwardProcessImageCorruption(self, init_image, savePath):
        XT = init_image.clone()
        num_subplots = (self.steps // self.series_frequency) + 1
        fig, axes = plt.subplots(nrows=1, ncols=num_subplots, figsize=((self.images_per_sample / 2) * 5, 4))
        counter = 0
        for step in tqdm(range(0, self.steps), desc="Forward Process Image Corruption"):
            if step % self.series_frequency == 0:
                self.diffusionSubplotting(XT[0], step, ax=axes[counter])  # Plot image at each step
                counter += 1
            XT = self.p_sample(XT, torch.full((self.batch_size,), step, device=device, dtype=torch.long))
        self.diffusionSubplotting(XT[0], self.steps, ax=axes[counter])  # Plot final image
        plt.savefig(savePath)
        plt.close()
        return XT

    @torch.no_grad()
    def plotReverseProcessImageSynthesis(self, init_image, savePath):
        XT = init_image.clone()
        num_subplots = (self.steps // self.series_frequency) + 1
        fig, axes = plt.subplots(nrows=1, ncols=num_subplots, figsize=((self.images_per_sample / 2) * 5, 4))
        self.diffusionSubplotting(XT[0], self.steps, ax=axes[0])  # Plot initial noise
        counter = 1
        for step in tqdm(reversed(range(0, self.steps)), desc="Reverse Process Image Synthesis"):
            XT = self.p_sample(XT, torch.full((self.batch_size,), step, device=device, dtype=torch.long))
            if step % self.series_frequency == 0:
                self.diffusionSubplotting(XT[0], step, ax=axes[counter])  # Plot image at each step
                counter += 1
        plt.savefig(savePath)
        plt.close()
        return XT

    ####################################### SAMPLE LOOPS #######################################

    # def vlb(self, X0, Xt, t):  # Variational lower bound
    #     posterior_mean, posterior_model_variance, _ = self.q_posterior_mean_variance(X0, Xt, t)
    #     prior_mean, prior_model_variance, _ = self.p_prior_mean_variance(Xt, t)
    #     log_likelihood = self.log_likelihood(X0, Xt, t)
    #     kl_divergence = self.kl_divergence(posterior_mean, posterior_model_variance, prior_mean, prior_model_variance)
    #     return (log_likelihood - kl_divergence).cpu().numpy()

    ####################################### DIFFUSION #######################################

    @staticmethod
    def getExtract(tensor: torch.Tensor, t: torch.Tensor, X):
        return tensor.gather(-1, t).view(t.shape[0], *((1,) * (len(X) - 1)))

    @staticmethod
    def kl_divergence(posterior_mean, posterior_variance, prior_mean, prior_variance):
        return 0.5 * (prior_variance / posterior_variance + ((posterior_mean - prior_mean) ** 2) / posterior_variance - 1 - torch.log(prior_variance / posterior_variance))

    def log_likelihood(self, X0, Xt, t):  # log p(x_t | x_{t-1}, x_0)
        mean, variance = self.p_posterior_mean_variance(X0, Xt, t)
        return -0.5 * torch.log(2 * math.pi * variance) - 0.5 * (Xt - mean) ** 2 / variance

    @torch.no_grad()
    def p_sample(self, sample, t):  # Sample from p_{theta}(x_{t-1} | x_t) = N(x_{t-1}; UNet(x_{t}, t), sigma_bar_t * I)
        mean, posterior_variance, posterior_log_variance = self.p_prior_mean_variance(sample, t)  # Sample from p_{theta}(x_{t-1} | x_t)
        noise = torch.randn_like(sample, device=device)  # Sample from N(0, I)
        nonzero_mask = ((t != 0).float().view(-1, *([1] * (len(sample.shape) - 1))))  # Mask for t != 0
        return mean + nonzero_mask * torch.exp(0.5 * posterior_log_variance) * noise  # Sample from p_{theta}(x_{t-1} | x_t)

    # TODO: Update p_prior_mean_variance to account for predicting_X0 and predicting_V from the target objective.
    def p_prior_mean_variance(self, Xt, t):  # Sample from p_{theta}(x_{t-1} | x_t) & q(x_{t-1} | x_t, x_0) = N(posterior_mean, posterior_model_variance)
        prediction = self.pred_X0_from_XT(Xt.float(), self.ddpm_model(Xt.float(), t), t)  # p(x_{t-1} | x_t)
        prediction = prediction.clamp(-1.0, 1.0)  # Clamp to [-1, 1]
        model_mean, posterior_variance, posterior_log_variance = self.q_posterior_mean_variance(prediction, Xt, t)  # Sample from q(x_{t-1} | x_t, x_0)
        return model_mean, posterior_variance, posterior_log_variance

    def p_posterior_mean_variance(self, X0, Xt, t):  # p(x_{t-1} | x_t, x_0) = N(posterior_mean, posterior_model_variance)
        posterior_mean = self.getExtract(self.Posterior1, t, Xt.shape) * X0 + self.getExtract(self.Posterior2, t, Xt.shape) * Xt  # p(x_{t-1} | x_t, x_0) = N(posterior1_mean, posterior_model)
        posterior_model_variance = self.getExtract(self.Posterior_Variance, t, Xt.shape)  # p(x_{t-1} | x_t, x_0) = N(posterior2_mean, posterior_model_variance)
        return posterior_mean, posterior_model_variance

    def q_sample(self, X0, t):  # Sample from q(Xt | X0) = N(x_t; sqrt(alpha_bar_t) * x_0, sqrt(1 - alpha_bar_t) * noise, t)
        noise = torch.randn_like(X0, device=device)  # Sample from N(0, I)
        QSample = self.getExtract(self.Sqrt_Alpha_Cumprod, t, X0.shape) * X0 + self.getExtract(self.Sqrt_1_Minus_Alpha_Cumprod, t, X0.shape) * noise  # Sample from q(Xt | X0)
        return QSample, noise

    def q_posterior_mean_variance(self, X0, Xt, t):  # q(x_{t-1} | x_t, x_0) = N(posterior_mean, posterior_model_variance)
        posterior_mean = self.getExtract(self.Posterior1, t, Xt.shape) * X0 + self.getExtract(self.Posterior2, t, Xt.shape) * Xt
        posterior_model_variance = self.getExtract(self.Posterior_Variance, t, Xt.shape)
        posterior_model_log_variance_clamp = self.getExtract(self.Posterior_Log_Variance_Clamp, t, Xt.shape)
        return posterior_mean, posterior_model_variance, posterior_model_log_variance_clamp

    ####################################### DIFFUSION #######################################

    def trainingLoop(self, dataset, optimizer):
        for idx, (X0, labels) in enumerate(dataset):  # TRAINING 1: Repeated Loop
            X0, labels = X0.to(device), labels.to(device)

            optimizer.zero_grad()

            loss = self.gradientDescent(labels, X0, idx)  # Compute the loss

            loss.backward()
            optimizer.step()

            if int(idx % 10) == 0:
                print(f"\t  T: {idx:05d}/{len(dataset)} | Loss: {round(self.loss.item(), 11)}")

    def gradientDescent(self, batched_labels, X0, idx):
        t = torch.randint(0, self.steps, (X0.shape[0],), device=device).long()
        XT, noise = self.q_sample(X0, t)

        target, snr = self.getTargetObjective(X0, noise, t)

        with autocast():
            self.ddpm_model.resetEmpowermentValues()
            prediction_noise = self.ddpm_model(XT, t)

        loss = self.getLoss(XT, prediction_noise, target, t, snr)

        if self.empower > 0.0:
            loss = loss - self.empower * np.mean(self.ddpm_model.getEmpowermentValues())

        self.loss = loss

        # When idx == 0, we are at the start of the epoch to print images  |  When idx == [after last batch], we are at the end of the epoch to print images
        if idx == 0 and self.current_epoch % self.plot_every == 0 and self.current_epoch != 0:
            self.getSampleImages(batched_labels, X0, noise, prediction_noise, t)

        return loss

    def run(self):

        self.generateFolders()
        self.printSystemDynamics()
        self.printTrainingInfo()

        # Get Data
        print("\n\t\t...Loading Data...")
        train_data, dataset, label = self.getDataset()
        X0 = next(iter(dataset))[0].to(device)

        # Get Models
        print("\n\t\t...Loading Models...")
        self.ddpm_model, optimizer = self.getDDPMModel()
        self.ddpm_model = self.ddpm_model.to(device)

        self.mine_model = self.getMINEModel()
        self.mine_model = self.mine_model.to(device)

        # Waiting for Triton to support Windows
        # self.ddpm_model = torch.compile(self.ddpm_model)
        # self.mine_model = torch.compile(self.mine_model)

        # Example of Forward Diffusion Process
        if self.collect_sequence_plots:
           self.saveImage(X0, "images/" + str(self.load_data) + "/Example An Image Input.jpg")
           print("\n\t\t...Generating Example Corruption Sequence...")
           savePath = "images/" + str(self.load_data) + "/" + str(self.scheduler) + "/Example Deconstruction Sequence.jpg"
           final_noise = self.plotForwardProcessImageCorruption(X0, savePath)
           self.saveImage(final_noise, "images/" + str(self.load_data) + "/" + str(self.scheduler) + "/Example Corruption.jpg")

        # Print Model Information
        # self.PrintModelInfo(model)

        startTime = time.time()
        print("\nTraining Start Time: " + str(datetime.datetime.now()))

        while self.current_epoch < self.epochs:
            print(f"\n-------------- Epoch {self.current_epoch} --------------")
            self.trainingLoop(dataset, optimizer)
            self.loss_list.append(format(self.loss.item()) if self.loss_metric == "KL" else self.loss.item())
            if self.save_model and self.current_epoch in self.model_checkpoints:
                self.saveModel()
            self.current_epoch += 1

        if self.current_epoch == self.epochs:
            print(f"\n-------------- Epoch {self.epochs - 1} --------------")
            print(f"\tFinal Model Loss: \t{round(self.loss.item(), 10)}")
            print(f"\tAverage Model Loss: \t{round(sum(self.loss_list) / len(self.loss_list), 10)}")
            print(f"\tMinimum Model Loss: \t{round(min(self.loss_list), 10)}")
            if self.save_model and self.current_epoch in self.model_checkpoints:
                self.saveModel()

        endTime = time.time()
        print("\nTraining Completion Time: " + str(datetime.datetime.now()))
        print(f"Total Training Time: {(endTime - startTime) / 60:.2f} mins")

        # Generate Data Plots
        self.plotGraph(self.loss_list, "Loss", "Epoch", "Loss", "Epoch = " + str(self.current_epoch) + "Training Loss", "images/" + str(self.load_data) + "/" + str(self.scheduler) + "/charts/Training Loss.jpg", reverse=False)

        # Generate Final Example Reconstruction
        if self.collect_sequence_plots:
            print("\n\t\t...Generating Example Reconstruction Sequence...")
            X_recon = self.plotReverseProcessImageSynthesis(final_noise, "images/" + str(self.load_data) + "/Example Reconstruction Sequence.jpg")
            self.saveImage(X_recon, "images/" + str(self.load_data) + "/" + str(self.scheduler) + "/Example Zn Image Output.jpg")

        if self.generate_gif:
            print("\n\t   ...Generating Gif... ")
            final_noise = self.simpleForward(X0)
            self.getDiffuionGif(final_noise)

    # TODO Demonstrates skipping timesteps
    # https://github.com/kxh001/ITdiffusion/blob/main/benchmark/improved-diffusion/improved_diffusion/respace.py


if __name__ == "__main__":
    main()
