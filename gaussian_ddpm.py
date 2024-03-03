#  Author: Paul-Jason Mello
#  Date: June 5th, 2023


#  General Libraries
import math
import re

import matplotlib
import numpy as np
import matplotlib.style as mplstyle
from matplotlib import pyplot as plt
from matplotlib.animation import ArtistAnimation

#  Torch Libraries
import torch
import torch.utils.data
import torch.nn.functional as func
from torch import optim
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler

#  Misc. Libraries
import os
import time
import datetime
from tqdm import tqdm

#  Model Libraries
from mlp import mlp

matplotlib.use("Agg")
device = "cuda" if torch.cuda.is_available() else "cpu"
torch.cuda.set_device(torch.cuda.current_device())

#  Set Seed for Reproducibility
# seed = 3407  # https://arxiv.org/abs/2109.08203
seed = np.random.randint(0, 1_000_000)  # random seed
np.random.seed(seed)
torch.manual_seed(seed)

mplstyle.use(["dark_background", "fast"])


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


def main():
    lr = 2e-4
    epochs = 1000  # Epochs

    batch_size = 512  # Batch Size, Test higher Batch Sizes

    sample_size = 100000  # Sample Size
    dimensionality = 2  # Dimensionality
    num_classes = 10  # Number of Classes

    steps = 1000
    t_steps = steps

    dataset = "gaussian"  # "gaussian"

    # TODO: fix "Not Working" noise schedulers
    schedule = "linear"  # "auto", "linear", "cosine", "sigmoid", "geometric"   | Not Working: "snr"  | "auto" > "linear" > "cosine" > "sigmoid" > "geometric"

    # TODO: fix "Not Working" loss metrics
    loss_metric = "MSE"  # Working: "MSE", "L1", "PSNR" | Not Working: "SCORE", "KL", "ELBO", "DSM", "ADSM"        | "MSE" > "L1" > "PSNR"

    # Score based loss functions ("SCORE", "ELBO", "DSM", "ADSM") might work with X0 / V predictions. Score has been shown to create something w/ V.
    # predict_noise: standard diffusion model prediction objective for training
    # predict_X0: predict the initial image x_0
    # predict_V: predict the initial image x_0 and the noise vector v_0
    prediction_objective = "predict_noise"  # Options: "predict_noise", "predict_X0", "predict_V"   | "predict_noise" > "predict_X0" > "predict_V"

    # Pretrained Model
    pretrained_model = True

    # Number of Samples to take from the diffusion process for sequence generation
    images_per_sample = 5

    # dims = [2, 5, 10, 25, 50, 100, 250, 500, 1000]
    # ddpm_epoch = [1, 5, 10, 20, 50, 100, 200, 500, 750, 1000]

    # dims = [250, 500, 1000]
    # ratios = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1500, 2000, 3000, 4000, 5000]
    # for d in dims:
    #     dimensionality = d
    #     for num in ratios:
    #         steps = num
    #         t_steps = steps
    #         ddpm_instance = gaussian_ddpm(loss_metric, dataset, num_classes, steps, t_steps, epochs, batch_size, lr, images_per_sample, sample_size, dimensionality, schedule, prediction_objective, pretrained_model)
    #         ddpm_instance.run()

    ddpm_instance = gaussian_ddpm(loss_metric, dataset, num_classes, steps, t_steps, epochs, batch_size, lr, images_per_sample, sample_size, dimensionality, schedule, prediction_objective, pretrained_model)
    ddpm_instance.run()


class gaussian_ddpm:
    def __init__(self, loss_metric, dataset, num_classes, steps, t_steps, epochs, batch_size, lr, images_per_sample, sample_size, dimensionality, schedule, prediction_objective, pretrained_model):
        super().__init__()

        self.ddpm_model = None
        self.mine_model = None

        self.save_model = False
        self.model_checkpoints = [1, 5, 10, 25, 50, 100, 250, 500, 750, 1000]

        self.lr = lr
        self.steps = steps
        self.epochs = epochs + 1  # + 1 for readability
        self.t_steps = t_steps
        self.load_data = dataset
        self.scheduler = schedule
        self.batch_size = batch_size
        self.sample_size = sample_size
        self.loss_metric = loss_metric
        self.num_classes = num_classes
        self.dimensionality = dimensionality

        self.pretrained_model = pretrained_model  # Use Pretrained Model
        self.prediction_objective = prediction_objective

        self.plot_every = 10
        self.images_per_sample = images_per_sample
        self.series_frequency = int(self.t_steps / self.images_per_sample)

        self.collect_mi_variance_charts = True
        self.collect_mi_trajectory_charts = True
        self.collect_conditional_mi_charts = True
        self.collect_incremental_information = True

        self.saveGif = True  # Generate Gif
        self.collectSequencePlots = True  # Collect Sequence Plots (Distribution / Image Series)

        self.current_epoch = 0

        self.loss = 0.0
        self.loss_list = []

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
                ddpm_dimension_path = os.path.join(path, self.load_data, f"{self.dimensionality}D")
                scheduler_path = os.path.join(ddpm_dimension_path, self.scheduler)
                steps_path = os.path.join(scheduler_path, str(self.steps))
                self.folderCheck(steps_path)

            elif subfolder == 'mine model':
                mine_model_dimension_path = os.path.join(path, self.load_data, f"{self.dimensionality}D")
                self.folderCheck(mine_model_dimension_path)

        images_dimension_path = os.path.join('images', self.load_data, f"{self.dimensionality}D", self.scheduler)
        self.folderCheck(images_dimension_path)

        images_subfolders = ['charts', 'noise', 'sequence plots']
        for subfolder in images_subfolders:
            subfolder_path = os.path.join(images_dimension_path, subfolder)
            self.folderCheck(subfolder_path)

            if subfolder == 'charts':
                for chart_folder in ['mi variance test', 'mi trajectory test', 'conditional mi test', 'mi increment test', 'noise']:
                    chart_path = os.path.join(subfolder_path, chart_folder)
                    self.folderCheck(chart_path)

            elif subfolder == 'sequence plots':
                for sequence_folder in ['distribution series', 'image series']:
                    sequence_path = os.path.join(subfolder_path, sequence_folder)
                    self.folderCheck(sequence_path)

    def getDDPMModel(self):
        if self.pretrained_model:
            pattern = re.compile(r'ddpm E(\d+)\.pt$')  # Pattern to find epoch number in the filename
            for epoch in range(10000, 0, -1):
                model_path = f"./models/ddpm model/{self.load_data}/{self.dimensionality}D/{self.scheduler}/{self.steps}/ddpm E{epoch}.pt"
                if os.path.isfile(model_path):
                    print(f"Loading DDPM Model from: {model_path}")
                    model = torch.load(model_path, map_location=torch.device('cuda'))  # Assuming CUDA is used
                    self.current_epoch = self.epochs
                    match = pattern.search(model_path)
                    if match:
                        self.current_epoch = int(match.group(1))
                    break
            else:
                print("No pretrained model found. Building new model.")
                model = mlp(input_dim=self.dimensionality, hidden_dim=int(self.batch_size / 2), output_dim=self.dimensionality, emb_dim=self.steps, dropout_prob=.1).to(device)
        else:
            model = mlp(input_dim=self.dimensionality, hidden_dim=int(self.batch_size / 2), output_dim=self.dimensionality, emb_dim=self.steps, dropout_prob=.1).to(device)
        # model = nn.DataParallel(model)  # Parallelize Data when Multi-GPU Applicable (Untested)
        optimizer = optim.Adam(model.parameters(), lr=self.lr, eps=1e-8)
        return model, optimizer

    def getMINEModel(self):
        mine_model_dir = f"./models/mine model/{self.load_data}/{self.dimensionality}D"
        highest_epoch = 0
        mine_model_path = None

        # Search for the latest model based on the highest epoch number
        if os.path.isdir(mine_model_dir):
            for filename in os.listdir(mine_model_dir):
                match = re.match(r"MI E(\d+).pt", filename)
                if match:
                    epoch = int(match.group(1))
                    if epoch > highest_epoch:
                        highest_epoch = epoch
                        mine_model_path = os.path.join(mine_model_dir, f"MI E{epoch}.pt")

        # Load the model if found
        if mine_model_path and os.path.isfile(mine_model_path):
            mine_model = torch.load(mine_model_path).to(device)
            print(f"Loading Mine Model from: {mine_model_path}")
        else:
            raise Exception(f"No MINE model found for {self.dimensionality}D Gaussian data.")

        return mine_model

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

    def getLoss(self, XT, prediction_noise, target, t):
        loss = None
        target = target.to(torch.float16)
        prediction_noise = prediction_noise.to(torch.float16)
        if self.loss_metric == "MSE":
            loss = func.mse_loss(target, prediction_noise)
        elif self.loss_metric == "L1":
            loss = func.l1_loss(target, prediction_noise)
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

        return loss

    def denoising_score_matching_loss(self, XT, prediction_noise, t):
        XT.requires_grad_(True)
        perturbed_x = XT + prediction_noise.detach()
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
        mse = torch.mean((epsilon_theta - target) ** 2)
        epsilon = 1e-10
        max_val = torch.max(target)
        psnr_like = 20 * torch.log10(max_val / torch.sqrt(mse + epsilon))
        return -psnr_like

    def score_loss(self, X0, prediction_noise):
        true_score = X0 - prediction_noise
        mse_loss = func.mse_loss(prediction_noise, true_score.half())
        return mse_loss

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
        # label = label.view(-1, 1).float()
        # label_tilde = label_tilde.view(-1, 1).float()
        # if self.single_image:
        #     xt = xt[:1]
        #     prev = prev[:1]
        #     label_tensor = label_tensor[:1]
        #     label_tilde_tensor = label_tilde_tensor[:1]
        return xt, prev, label, label_tilde

    def get_mi_type(self, XT, prev, label, label_tilde, mi_method):
        if mi_method == 'mutual information':
            return self.get_mi_mine(XT, prev, label, label_tilde)
        elif mi_method == 'incremental information':
            return self.get_inc_mi_mine(XT, prev, label, label_tilde)
        elif mi_method == 'conditional mutual information':
            return self.get_conditional_mi_mine(XT, prev, label, label_tilde)

    def get_mi_mine(self, xt, prev, labels, shuffled_labels):

        # Ensure labels and label_tilde are 2D and match the shape of xt and prev
        labels = labels.unsqueeze(-1).expand(-1, 2)  # Expands the shape from [512] to [512, 2]
        shuffled_labels = shuffled_labels.unsqueeze(-1).expand(-1, 2)  # Same for label_tilde

        xt, prev, label, label_tilde = self.prepare_mine_data(xt, prev, labels, shuffled_labels)

        # print(xt.shape)
        # print(prev.shape)
        # print(label.shape)
        # print(label_tilde.shape)

        T_joint = self.mine_model(xt, label)
        T_marginal = torch.exp(self.mine_model(xt, label_tilde))

        mi_tensor = T_joint - torch.log(T_marginal + 1e-10)
        mi_tensor /= math.log(2)
        return torch.mean(mi_tensor).item()

    def get_inc_mi_mine(self, xt, prev, labels, shuffled_labels):
        # Ensure labels and label_tilde are 2D and match the shape of xt and prev
        # labels = labels.unsqueeze(-1)  # Expands the shape from [512] to [512, 2]
        # shuffled_labels = shuffled_labels.unsqueeze(-1)  # Same for label_tilde

        xt, prev, label, label_tilde = self.prepare_mine_data(xt, prev, labels, shuffled_labels)

        # label = torch.cat((label, label), dim=1)
        # label_tilde = torch.cat((label_tilde, label_tilde), dim=1)

        channel_tensor = torch.cat((xt, prev), dim=1)

        T_joint = self.mine_model(channel_tensor, label)
        T_marginal = torch.exp(self.mine_model(channel_tensor, label_tilde))

        mi_tensor = T_joint - torch.log(T_marginal + 1e-10)
        mi_tensor /= math.log(2)
        channel_val = torch.mean(mi_tensor).item()

        prev_cat = torch.cat((prev, prev), dim=1)

        T_joint = self.mine_model(prev_cat, label)
        T_marginal = torch.exp(self.mine_model(prev_cat, label_tilde))

        mi_tensor = T_joint - torch.log(T_marginal + 1e-10)
        mi_tensor /= math.log(2)
        prev_val = torch.mean(mi_tensor).item()

        return channel_val - prev_val

    def get_conditional_mi_mine(self, xt, prev, labels, shuffled_labels):
        # First, prepare the data as before
        xt, prev, label, label_tilde = self.prepare_mine_data(xt, prev, labels, shuffled_labels)

        # No need to unsqueeze and then repeat, as we directly expand to the desired shape
        # Adjust labels to match the shape of xt and prev after concatenation
        label = labels.unsqueeze(-1).expand(-1, 2)  # Directly expand to [512, 2] to match xt_concat and prev_concat
        label_tilde = shuffled_labels.unsqueeze(-1).expand(-1, 2)  # Same for label_tilde

        # Concatenate xt and prev to double their dimensionality
        xt_concat = torch.cat((xt, xt), dim=1)
        prev_concat = torch.cat((prev, prev), dim=1)

        # Ensure the dimensions are printed for verification
        print("xt_concat shape:", xt_concat.shape)
        print("prev_concat shape:", prev_concat.shape)
        print("label shape:", label.shape)
        print("label_tilde shape:", label_tilde.shape)

        # Proceed with model computation as before
        T_joint = self.mine_model(xt_concat, label)
        T_marginal = self.mine_model(prev_concat, label)
        expT = torch.exp(self.mine_model(xt_concat, label_tilde))

        # Compute the conditional mutual information tensor
        cmi_tensor = T_joint - T_marginal - torch.log(expT.clamp(min=1e-10))
        cmi_tensor /= math.log(2)  # Normalize to bits

        return torch.mean(cmi_tensor).item()

####################################### MINE #######################################

    def process_mi_trajectory(self, X0_batch, current_labels_batch, current_labels_marginal_batch, savePath, mi_method):
        plt.figure(figsize=(10, 10))
        self.saveImage(X0_batch,  title=None, filename=f'{savePath} A Input.jpg')

        mine_fp_mi, final_noise = self.trajectory(X0_batch, current_labels_batch, current_labels_marginal_batch, mi_method, reverse=False, description=f'Forward Process')
        self.plotGraph(arr=mine_fp_mi, label="MINE Predicted MI", x_label="Timestep", y_label="Mutual Information", title=f'MINE Forward Process', savePath=f'{savePath} B FP MINE.jpg', reverse=False)
        plt.figure(figsize=(10, 10))
        self.saveImage(final_noise,  title=None, filename=f'{savePath} C Corrupted.jpg')

        mine_rp_mi, noise_recon = self.trajectory(final_noise, current_labels_batch, current_labels_marginal_batch, mi_method, reverse=True, description=f'Reverse Process')
        self.plotGraph(arr=mine_rp_mi, label="MINE Predicted MI", x_label="Timestep", y_label="Mutual Information", title=f'MINE Reverse Process, start from FP output corruption', savePath=f'{savePath} E RP MINE Noise.jpg', reverse=True)
        plt.figure(figsize=(10, 10))
        self.saveImage(noise_recon, title=None, filename=f'{savePath} F Reconstructed Noise.jpg')

    def process_conditional_mi(self, X0_batch, current_labels_batch, current_labels_marginal_batch, savePath, mi_method):
        plt.figure(figsize=(10, 10))
        self.saveImage(X0_batch,  title=None, filename=f'{savePath} A Input.jpg')

        mine_fp_mi, final_noise = self.trajectory(X0_batch, current_labels_batch, current_labels_marginal_batch, mi_method, reverse=False, description=f'Forward Process Expiraments')
        self.plotGraph(arr=mine_fp_mi, label="MINE Predicted MI", x_label="Timestep", y_label="Conditional Mutual Information", title=f'Experiment: MINE Forward Process, Conditional Mutual Information', savePath=f'{savePath} B FP MINE.jpg', reverse=False)
        plt.figure(figsize=(10, 10))
        self.saveImage(final_noise, title=None, filename= f'{savePath} C Corrupted.jpg')

        mine_rp_mi, noise_recon = self.trajectory(final_noise, current_labels_batch, current_labels_marginal_batch, mi_method, reverse=True, description=f'Reverse Process Expiraments')
        self.plotGraph(arr=mine_rp_mi, label="MINE Predicted MI", x_label="Timestep", y_label="Conditional Mutual Information", title=f'Experiment: MINE Reverse Process, Conditional Mutual Information', savePath=f'{savePath} E RP MINE Noise.jpg', reverse=True)
        plt.figure(figsize=(10, 10))
        self.saveImage(noise_recon, title=None, filename=f'{savePath} F Reconstructed Noise.jpg')

    def process_incremental_information(self, X0_batch, current_labels_batch, current_labels_marginal_batch, savePath, mi_method):
        plt.figure(figsize=(10, 10))
        self.saveImage(X0_batch, title=None, filename=f'{savePath} A Input.jpg')

        mine_fp_mi, final_noise = self.trajectory(X0_batch, current_labels_batch, current_labels_marginal_batch, mi_method, reverse=False, description=f'Forward Process')
        self.plotGraph(arr=mine_fp_mi, label="MINE Prediction", x_label="Timestep", y_label="Incremental Information", title=f'MINE FP Incremental Information Calculation Chart', savePath=f'{savePath} B FP MINE.jpg', reverse=False)
        plt.figure(figsize=(10, 10))
        self.saveImage(final_noise, title=None, filename=f'{savePath} C Corrupted.jpg')

        mine_rp_mi, noise_recon = self.trajectory(final_noise, current_labels_batch, current_labels_marginal_batch, mi_method, reverse=True, description=f'Reverse Process')
        self.plotGraph(arr=mine_rp_mi, label="MINE Prediction", x_label="Timestep", y_label="Incremental Information", title=f'MINE RP Incremental Information Calculation Chart', savePath=f'{savePath} E RP MINE Noise.jpg', reverse=True)
        plt.figure(figsize=(10, 10))
        self.saveImage(noise_recon, title=None, filename=f'{savePath} F Reconstructed Noise.jpg')

    def process_mi_variance(self, X0_batch, current_labels_batch, current_labels_marginal_batch, savePath, mi_method):
        plt.figure(figsize=(10, 10))
        self.saveImage(X0_batch,  title=None, filename=f'{savePath} A Input.jpg')

        mine_fp_mi = [[] for _ in range(5)]
        mine_rp_mi = [[] for _ in range(5)]
        final_noise = None
        recon_noise = None

        for i in range(5):
            mine_fp_mi[i], final_noise = self.trajectory(X0_batch, current_labels_batch, current_labels_marginal_batch, mi_method, reverse=False, description=f'Forward Process Variance')
        fp_avg_variance = self.getAverageVariance(*mine_fp_mi)
        self.plotGraphVariance(arr1=mine_fp_mi[0], arr2=mine_fp_mi[1], arr3=mine_fp_mi[2], arr4=mine_fp_mi[3], arr5=mine_fp_mi[4], avg_var=fp_avg_variance, avg_label="Avg. Var.", x_label="Timestep", y_label="Mutual Information", title=f'MINE Forward Process Variance', savePath=f'{savePath} B FP MINE.jpg', reverse=False)
        plt.figure(figsize=(10, 10))
        self.saveImage(final_noise, title=None, filename=f'{savePath} C Corrupted.jpg')

        for i in range(5):
            mine_rp_mi[i], recon_noise = self.trajectory(final_noise, current_labels_batch, current_labels_marginal_batch, mi_method, reverse=True, description=f'Reverse Process Variance')
        rp_avg_variance = self.getAverageVariance(*mine_rp_mi)
        self.plotGraphVariance(arr1=mine_rp_mi[0], arr2=mine_rp_mi[1], arr3=mine_rp_mi[2], arr4=mine_rp_mi[3], arr5=mine_rp_mi[4], avg_var=rp_avg_variance, avg_label="Avg. Var.", x_label="Timestep", y_label="Mutual Information", title=f'MINE Reverse Process Variance', savePath=f'{savePath} E RP MINE Noise.jpg', reverse=True)
        plt.figure(figsize=(10, 10))
        self.saveImage(recon_noise,  title=None, filename=f'{savePath} F Reconstructed Noise.jpg')
        # TODO IDEA: Implement a graph which shows the MI of the MINE model stretched over time, and under the associated image series.
        # TODO Above Example:
        # A series plot of two rows where the top row is the MI through time from forward and backward process and the bottom is the respective images at an interval of timesteps.

    def getAverageVariance(self, arr1, arr2, arr3, arr4, arr5):
        avg_var = []
        for i in range(len(arr1)):
            avg_var.append((arr1[i] + arr2[i] + arr3[i] + arr4[i] + arr5[i]) / 5)
        return avg_var

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

    def getSampleImages(self, batched_labels, X0, XT, noise, prediction_noise, t):
        savePathMI = f"images/{self.load_data}/{self.dimensionality}D/{self.scheduler}/charts/mi trajectory test/Epoch = {self.current_epoch}"
        savePathVar = f"images/{self.load_data}/{self.dimensionality}D/{self.scheduler}/charts/mi variance test/Epoch = {self.current_epoch}"
        savePathCondMI = f"images/{self.load_data}/{self.dimensionality}D/{self.scheduler}/charts/conditional mi test/Epoch = {self.current_epoch}"
        savePathIncMI = f"images/{self.load_data}/{self.dimensionality}D/{self.scheduler}/charts/mi increment test/Epoch = {self.current_epoch}"
        savePathSeq = f"images/{self.load_data}/{self.dimensionality}D/{self.scheduler}/sequence plots/image series/"

        current_labels = batched_labels.clone().detach()
        X0_batch, current_labels_batch, current_labels_marginal_batch = self.sample_batch_from_X0_and_labels(X0, current_labels, self.batch_size)

        # print("X0_BATCH: ", X0_batch)
        # print("XO_BATCH", X0_batch.shape)
        # print("current_labels_batch", current_labels_batch.shape)
        # print("current_labels_marginal_batch", current_labels_marginal_batch.shape)

        # Collect Process Trajectories Charts
        if self.collect_mi_trajectory_charts:
            print("\n\t\t...Generating Trajectories...")
            self.process_mi_trajectory(X0_batch, current_labels_batch, current_labels_marginal_batch, savePathMI, mi_method="mutual information")

        # Collect Conditional MI Charts
        if self.collect_conditional_mi_charts:
            print("\n\t\t...Generating Conditional MI Experiment...")
            #self.process_conditional_mi(X0_batch, current_labels_batch, current_labels_marginal_batch, savePathCondMI, mi_method="conditional mutual information")

        # Collect Process Variance Charts
        if self.collect_mi_variance_charts:
            print("\n\t\t...Generating Variance...")
            self.process_mi_variance(X0_batch, current_labels_batch, current_labels_marginal_batch, savePathVar, mi_method="mutual information")

        # Collect Incremental Information Charts
        if self.collect_incremental_information:
            print("\n\t\t...Generating Incremental Charts...")
            #self.process_incremental_information(X0_batch, current_labels_batch, current_labels_marginal_batch, savePathIncMI, mi_method="incremental information")

        # Reverse Trajectory | Start from initial training image to reconstruction
        # rp_noise_MI, X0_recon = self.ReverseTrajectoryX0(X0_batch, current_labels_batch, current_labels_marginal_batch, X0_batch)

        # model.plot_empowerment_dynamics(savePath + " Model Empowerment Dynamics.jpg")

        if self.collectSequencePlots:
            print("\t\t...Generating Image Sequences...")
            final_noise = self.plotForwardProcessCorruption(X0_batch)
            self.saveImage(final_noise, title=None, filename="" + str(savePathSeq) + "Epoch=" + str(self.current_epoch) + " FP.jpg")
            final_recon = self.plotReverseProcessReconstruction(final_noise)
            self.saveImage(final_recon, title=None, filename="" + str(savePathSeq) + "Epoch=" + str(self.current_epoch) + " RP.jpg")

    # def vlb(self, X0, Xt, t):  # Variational lower bound
    #     posterior_mean, posterior_model_variance, _ = self.q_posterior_mean_variance(X0, Xt, t)
    #     prior_mean, prior_model_variance, _ = self.p_prior_mean_variance(Xt, t)
    #     log_likelihood = self.log_likelihood(X0, Xt, t)
    #     kl_divergence = self.kl_divergence(posterior_mean, posterior_model_variance, prior_mean, prior_model_variance)
    #     return (log_likelihood - kl_divergence).cpu().numpy()

    @torch.no_grad()
    def trajectory(self, init_image, label, label_tilde, mi_method, reverse, description):
        XT = init_image.clone()
        mine_mi = [None] * self.steps
        steps_range = range(self.steps - 1, -1, -1) if reverse else range(0, self.steps)

        for step in tqdm(steps_range, desc=description):
            prev = XT
            XT = self.p_sample(XT, torch.full((self.batch_size,), step, device=XT.device, dtype=torch.long))
            mine_mi[step] = self.get_mi_type(XT, prev, label, label_tilde, mi_method)
        return mine_mi, XT

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

    def log_likelihood(self, X0, Xt, t):  # log p(x_t | x_{t-1}, x_0)
        mean, variance = self.p_posterior_mean_variance(X0, Xt, t)
        return -0.5 * torch.log(2 * math.pi * variance) - 0.5 * (Xt - mean) ** 2 / variance

    @staticmethod
    def kl_divergence(posterior_mean, posterior_variance, prior_mean, prior_variance):
        return 0.5 * (prior_variance / posterior_variance + ((posterior_mean - prior_mean) ** 2) / posterior_variance - 1 - torch.log(prior_variance / posterior_variance))

    def p_posterior_mean_variance(self, X0, Xt, t):  # p(x_{t-1} | x_t, x_0) = N(posterior_mean, posterior_model_variance)
        posterior_mean = self.getExtract(self.Posterior1, t, Xt.shape) * X0 + self.getExtract(self.Posterior2, t, Xt.shape) * Xt  # p(x_{t-1} | x_t, x_0) = N(posterior1_mean, posterior_model)
        posterior_model_variance = self.getExtract(self.Posterior_Variance, t, Xt.shape)  # p(x_{t-1} | x_t, x_0) = N(posterior2_mean, posterior_model_variance)
        return posterior_mean, posterior_model_variance

    @staticmethod
    def getExtract(tensor: torch.Tensor, t: torch.Tensor, X):
        return tensor.gather(-1, t).view(t.shape[0], *((1,) * (len(X) - 1)))

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
        model_mean, posterior_variance, posterior_log_variance = self.q_posterior_mean_variance(prediction, Xt, t)  # Sample from q(x_{t-1} | x_t, x_0)
        return model_mean, posterior_variance, posterior_log_variance

    @torch.no_grad()
    def p_sample(self, sample, t):  # Sample from p_{theta}(x_{t-1} | x_t) = N(x_{t-1}; UNet(x_{t}, t), sigma_bar_t * I)
        mean, posterior_variance, posterior_log_variance = self.p_prior_mean_variance(sample, t)  # Sample from p_{theta}(x_{t-1} | x_t)
        noise = torch.randn_like(sample, device=device)  # Sample from N(0, I)
        nonzero_mask = ((t != 0).float().view(-1, *([1] * (len(sample.shape) - 1))))  # Mask for t != 0
        return mean + nonzero_mask * torch.exp(0.5 * posterior_log_variance) * noise  # Sample from p_{theta}(x_{t-1} | x_t)

    def getDiffusionGif(self, img):
        fig, ax = plt.subplots()
        img = torch.randn_like(img, device=device)
        imgs = ax.scatter(img[:, 0].cpu(), img[:, 1].cpu(), c="b", linewidths=1, s=5)
        frames = [[imgs]]
        for step in tqdm(reversed(range(0, self.steps))):
            img = self.p_sample(img, torch.full((self.batch_size,), step, device=device, dtype=torch.long))
            new_img = ax.scatter(img[:, 0].cpu(), img[:, 1].cpu(), c="b", linewidths=1, s=5)
            frames.append([new_img])
        plt.title(str(self.dimensionality) + "D Gaussian diffusion")
        animate = ArtistAnimation(fig, frames, interval=25, blit=True, repeat_delay=3000, repeat=True)
        animate.save(f"images/{self.load_data}/{self.dimensionality}D/{self.scheduler}/F {self.dimensionality}D Gaussian diffusion.gif", dpi=80, writer="pillow")
        plt.close(fig)

    def diffusionSubplotting(self, XT, step, string):
        if string == "reverse":
            subplot_position = self.images_per_sample - int(step / self.series_frequency) + 1
        elif string == "forward":
            subplot_position = int(step / self.series_frequency) + 1
        plt.subplot(1, self.images_per_sample + 1, subplot_position)
        plt.title(f"Time = {step}")
        XT = XT.cpu().detach().numpy()

        for i in range(XT.shape[0]):  # Iterate over the batch
            plt.scatter(XT[i, 0], XT[i, 1], c='b', alpha=0.6)

        plt.xlabel('Dimension 1')
        plt.ylabel('Dimension 2')

    @torch.no_grad()
    def plotReverseProcessReconstruction(self, XT):
        plt.figure(figsize=((self.images_per_sample / 2) * 12, 5))
        self.diffusionSubplotting(XT, self.steps, "reverse")
        for step in tqdm(reversed(range(0, self.steps)), desc="Reverse Process"):
            XT = self.p_sample(XT, torch.full((self.batch_size,), step, device=device, dtype=torch.long))
            if step % self.series_frequency == 0:
                self.diffusionSubplotting(XT, step, "reverse")
        return XT

    @torch.no_grad()
    def plotForwardProcessCorruption(self, XT):
        plt.figure(figsize=((self.images_per_sample / 2) * 12, 5))
        for step in tqdm(range(0, self.steps), desc="Forward Process"):
            if step % self.series_frequency == 0:
                self.diffusionSubplotting(XT, step, "forward")
            XT = self.p_sample(XT, torch.full((self.batch_size,), step, device=device, dtype=torch.long))
        self.diffusionSubplotting(XT, self.steps, "forward")
        return XT

    def getTargetObjective(self, X0, noise, t):
        target = snr = None
        if self.prediction_objective == "predict_noise":
            target = noise
        elif self.prediction_objective == "predict_X0":
            target = X0
        elif self.prediction_objective == "predict_V":
            target = self.pred_V(X0, t, noise)
        return target, snr

    def printSystemDynamics(self):
        print("Cuda: ", torch.cuda.is_available())
        print("Device Count: ", torch.cuda.device_count())
        print("Device: ", device)
        print("Device Name: ", torch.cuda.get_device_name(device))
        print("\nImages Series Count: ", self.images_per_sample)
        print("Steps Between Images: ", self.series_frequency)

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

    def saveImage(self, data, title, filename):
        # if data.shape[1] != 2:
        #     raise ValueError("Data is not 2D and cannot be plotted.")

        data_cpu = data.cpu() if data.is_cuda else data

        plt.scatter(data_cpu[:, 0].numpy(), data_cpu[:, 1].numpy(), c='b', alpha=0.6)
        if title is not None:
            plt.title(f"{title} - Loss: {self.loss:.6f}")
        plt.xlabel("Dimension 1")
        plt.ylabel("Dimension 2")
        plt.savefig(filename)
        plt.close()

    def saveModel(self):
        print("\n\t\t...Saving Model...")
        torch.save(self.ddpm_model, f"models/ddpm model/{self.load_data}/{self.dimensionality}D/{self.scheduler}/{self.steps}/ddpm E{self.current_epoch}.pt")

    def gradientDescent(self, batched_labels, X0, idx):
        t = torch.randint(0, self.steps, (X0.shape[0],), device=device).long()
        XT, noise = self.q_sample(X0, t)

        target, snr = self.getTargetObjective(X0, noise, t)

        prediction_noise = self.ddpm_model(XT, t)

        self.loss = loss = self.getLoss(XT, prediction_noise, target, t)

        # When idx == 0, we are at the start of the epoch to print images  |  When idx == [after last batch], we are at the end of the epoch to print images
        if idx == 0 and self.current_epoch % self.plot_every == 0 and self.current_epoch != 0:
            self.getSampleImages(batched_labels, X0, XT, noise, prediction_noise, t)

        return loss

    def trainingLoop(self, dataset, scaler, optimizer):
        for idx, (X0, labels) in enumerate(dataset):  # TRAINING 1: Repeated Loop
            X0, labels = X0.to(device), labels.to(device)

            optimizer.zero_grad()

            loss = self.gradientDescent(labels, X0, idx)  # TRAINING 4: θ ← θ − α ∇θL(θ)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            if int(idx % 10) == 0:
                print(f"\t  T: {idx:05d}/{len(dataset)} | Loss: {round(self.loss.item(), 11)}")

    def run(self):

        # Generate Folder Structure
        self.generateFolders()

        # Print System Dynamics
        self.printSystemDynamics()

        # Print Training Information
        self.printTrainingInfo()

        # Get Data
        print("\n\t\t...Loading Data...")
        print("\nDimensionality: " + str(self.dimensionality) + "")
        dataset = GaussianDataset(dimensionality=self.dimensionality, sample_size=self.sample_size, num_classes=self.num_classes)

        # Normalize between [-1, 1]
        # We don't want to normalize the data, because we want to see the effect of the noise
        # data = (data - data.min())/(data.max() - data.min()) * 2 - 1

        dataset = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, pin_memory=True, num_workers=4, pin_memory_device="cuda", persistent_workers=True)
        img = next(iter(dataset))[0].to(device)

        # Get Models
        print("\t\t...Loading Models...")
        self.ddpm_model, optimizer = self.getDDPMModel()

        self.mine_model = self.getMINEModel()

        # Waiting for Triton to support Windows
        # model = torch.compile(model)
        # mine_model = torch.compile(mine_model)

        # Example of Forward Diffusion Process, Same as in plotSampleQuality but not reversed
        if self.collectSequencePlots:
            self.saveImage(img, title="Example Input", filename="images/" + str(self.load_data) + "/" + str(self.dimensionality) + "D/" + str(self.scheduler) + "/A Example Input.jpg")  # Single Image
            print("\n\t\t...Generating Example Corruption Sequence...")
            final_noise = self.plotForwardProcessCorruption(img)
            self.saveImage(final_noise, title=None, filename="images/" + str(self.load_data) + "/" + str(self.dimensionality) + "D/" + str(self.scheduler) + "/B Example Corruption.jpg")  # Series Plot
            self.saveImage(final_noise, title="Final Corrupt", filename="images/" + str(self.load_data) + "/" + str(self.dimensionality) + "D/" + str(self.scheduler) + "/C Final Corruption.jpg")  # Series Plot

        startTime = time.time()
        print("\nTraining Start Time: " + str(datetime.datetime.now()))
        scaler = GradScaler()

        while self.current_epoch != self.epochs:
            print(f"\n     -------------- Epoch {self.current_epoch} -------------- ")
            self.trainingLoop(dataset, scaler, optimizer)  # Sampling done intermittently during training
            self.loss_list.append(format(self.loss.item()) if self.loss_metric == "KL" else self.loss.item())
            if self.save_model and self.current_epoch in self.model_checkpoints:
                self.saveModel()
            self.current_epoch += 1

        if self.save_model and self.current_epoch in self.model_checkpoints:
            self.saveModel()

        print(f"\n     -------------- Epoch {self.current_epoch} -------------- ")
        print(f"\t  Final   Model Loss: \t{round(self.loss.item(), 10)}")
        print(f"\t  Average Model Loss: \t{round(sum(self.loss_list) / len(self.loss_list), 10)}")
        print(f"\t  Minimum Model Loss: \t{round(min(self.loss_list), 10)}")

        endTime = time.time()
        print("\nTraining Completion Time: " + str(datetime.datetime.now()))
        print(f"Total Training Time: {(endTime - startTime) / 60:.2f} mins")

        if self.collectSequencePlots:
            print("\n\t\t...Generating Example Reconstruction Sequence...")
            final_recon = self.plotReverseProcessReconstruction(final_noise)
            self.saveImage(final_recon, title=None, filename=f"images/{self.load_data}/{self.dimensionality}D/{self.scheduler}/D Example Reconstruction.jpg")  # Series Plot
            self.saveImage(final_recon, title="Final Recon", filename=f"images/{self.load_data}/{self.dimensionality}D/{self.scheduler}/E Final Reconstruction.jpg")

        if self.collectSequencePlots & self.saveGif:
            print("\n\t   ...Generating Gif... ")
            self.getDiffusionGif(final_noise)


# class GaussianDataset(torch.utils.data.Dataset):
#     def __init__(self, dimensionality, sample_size, num_classes, rho):
#         self.dimensionality = dimensionality
#         self.sample_size = sample_size
#         self.num_classes = num_classes
#         self.rho = rho  # Correlation coefficient
#         self.data, self.labels = self.getLabeledDataset()
#
#     def getCovarianceMatrix(self, dim, rho):
#         # Create a matrix with all elements as 1-Rho
#         covariance_matrix = torch.full((dim, dim), 1 - rho)
#         # Adjust the diagonal elements to be 1
#         covariance_matrix.fill_diagonal_(1.0)
#         return covariance_matrix
#
#     def getMultiGauss(self, mean, covariance):
#         mvn = MultivariateNormal(mean, covariance)
#         return mvn.sample((self.sample_size,))
#
#     def getLabeledDataset(self):
#         mean = torch.zeros(self.dimensionality)
#         covariance_matrix = self.getCovarianceMatrix(self.dimensionality, self.rho)
#         data = self.getMultiGauss(mean, covariance_matrix)
#
#         # Eigenvalue decomposition for labeling (unchanged)
#         eigenvalues, _ = torch.linalg.eigh(covariance_matrix)
#         scaled_eigenvalues = torch.floor(self.num_classes * (eigenvalues / eigenvalues.max()))
#
#         labels = scaled_eigenvalues.repeat(self.sample_size // self.dimensionality)
#         return data, labels
#
#     def __len__(self):
#         return len(self.data)
#
#     def __getitem__(self, idx):
#         return self.data[idx], self.labels[idx]

# class GaussianDataset(torch.utils.data.Dataset):
#     def __init__(self, dimensionality, sample_size, num_classes):
#         self.dimensionality = dimensionality
#         self.sample_size = sample_size
#         self.num_classes = num_classes
#         self.data, self.labels = self.getLabeledDataset()
#
#     def getPositiveDefiniteMatrix(self, dim):
#         random_matrix = torch.rand(dim, dim)
#         return random_matrix @ random_matrix.t()
#
#     def getMultiGauss(self, mean, covariance):
#         mvn = MultivariateNormal(mean, covariance)
#         return mvn.sample((self.sample_size,))
#
#     def getLabeledDataset(self):
#         mean = torch.zeros(self.dimensionality)
#         covariance_matrix = self.getPositiveDefiniteMatrix(self.dimensionality)
#         data = self.getMultiGauss(mean, covariance_matrix)
#
#         # Eigenvalue decomposition
#         eigenvalues, _ = torch.linalg.eigh(covariance_matrix)
#         scaled_eigenvalues = torch.floor(self.num_classes * (eigenvalues / eigenvalues.max()))
#
#         labels = scaled_eigenvalues.repeat(self.sample_size // self.dimensionality)
#         return data, labels
#
#     def __len__(self):
#         return len(self.data)
#
#     def __getitem__(self, idx):
#         return self.data[idx], self.labels[idx]

class GaussianDataset(torch.utils.data.Dataset):
    def __init__(self, dimensionality, sample_size, num_classes):
        self.dimensionality = dimensionality
        self.sample_size = sample_size
        self.num_classes = num_classes
        self.data, self.labels = self.getLabeledDataset()

    def getPositiveDefiniteMatrix(self, dim):
        random_matrix = torch.rand(dim, dim)
        return random_matrix @ random_matrix.t()

    def getMultiGauss(self, mean, covariance):
        mvn = MultivariateNormal(mean, covariance)
        return mvn.sample((self.sample_size,))

    def getLabeledDataset(self):
        mean = torch.zeros(self.dimensionality)
        covariance_matrix = self.getPositiveDefiniteMatrix(self.dimensionality)
        data = self.getMultiGauss(mean, covariance_matrix)

        # Eigenvalue decomposition
        eigenvalues, _ = torch.linalg.eigh(covariance_matrix)
        min_eigenvalue, max_eigenvalue = eigenvalues.min(), eigenvalues.max()

        # Normalize eigenvalues to the range [0, 1] and then scale to the range [0, num_classes-1]
        normalized_eigenvalues = (eigenvalues - min_eigenvalue) / (max_eigenvalue - min_eigenvalue)
        scaled_eigenvalues = torch.floor(normalized_eigenvalues * self.num_classes).long()

        # Ensure labels are within 0 to num_classes-1
        labels = scaled_eigenvalues % self.num_classes
        labels = labels.repeat(self.sample_size // self.dimensionality)

        return data, labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


if __name__ == "__main__":
    main()
