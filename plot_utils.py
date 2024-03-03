import io
import torch
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from image_transform import ConvertToImage
from matplotlib.animation import ArtistAnimation

device = "cuda" if torch.cuda.is_available() else "cpu"


class plot_utils:
    def __init__(self, ddpm_model, batch_size, scheduler, steps, loss, numChannels, images_per_sample, series_frequency, load_data):
        self.ddpm_model = ddpm_model
        self.batch_size = batch_size
        self.scheduler = scheduler
        self.steps = steps
        self.loss = loss
        self.load_data = load_data
        self.numChannels = numChannels
        self.images_per_sample = images_per_sample
        self.series_frequency = series_frequency
        self.epochCounter =

# IMAGES:
    def saveImageSize(self, img, msg, size):
        convertToImage = ConvertToImage()

        if len(img.shape) == 4:
            img = img[0]

        # Convert tensor to image format
        image_data = convertToImage(img)

        # Scale pixel values to [0, 255] if necessary
        if image_data.max() <= 1.0:
            image_data = (image_data * 255).astype(np.uint8)

        # Plotting using matplotlib
        fig, ax = plt.subplots(figsize=(size / plt.rcParams['figure.dpi'], size / plt.rcParams['figure.dpi']),
                               dpi=plt.rcParams['figure.dpi'])
        if self.numChannels == 1:
            ax.imshow(image_data[:, :, 0], cmap="gray")
        elif self.numChannels == 3:
            ax.imshow(image_data)

        plt.axis("off")

        # Save using plt.savefig
        plt.savefig(msg, bbox_inches='tight', pad_inches=0)
        plt.close(fig)

    def saveImage(self, img, msg):
        convertToImage = ConvertToImage()
        if len(img.shape) == 4:
            img = img[0, :, :, :]

        if self.numChannels == 1:
            plt.imshow(convertToImage(img), cmap="gray")
        elif self.numChannels == 3:
            plt.imshow(convertToImage(img))
        plt.title("Loss = " + str("{:.6f}".format(self.loss)))
        plt.axis("off")
        plt.savefig(msg)
        plt.close()

# PLOTTING:
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
    def plotForwardProcessImageCorruption(self, X0):
        XT = X0
        num_subplots = (self.steps // self.series_frequency) + 1
        fig, axes = plt.subplots(nrows=1, ncols=num_subplots, figsize=((self.images_per_sample / 2) * 5, 4))
        counter = 0
        for step in tqdm(range(0, self.steps), desc="Forward Process Image Corruption"):
            if step % self.series_frequency == 0:
                self.diffusionSubplotting(XT[0], step, ax=axes[counter])  # Plot image at each step
                counter += 1
            XT = ddpm.p_sample(XT, torch.full((self.batch_size,), step, device=device, dtype=torch.long))
        self.diffusionSubplotting(XT[0], self.steps, ax=axes[counter])  # Plot final image
        plt.savefig("images/" + str(self.load_data) + "/" + str(self.scheduler) + "/Example Deconstruction Sequence.jpg")
        plt.close()
        return XT

    @torch.no_grad()
    def plotReverseProcessImageSynthesis(self, XT, savePath):
        num_subplots = (self.steps // self.series_frequency) + 1
        fig, axes = plt.subplots(nrows=1, ncols=num_subplots, figsize=((self.images_per_sample / 2) * 5, 4))
        self.diffusionSubplotting(XT[0], self.steps, ax=axes[0])  # Plot initial noise
        counter = 1
        for step in tqdm(reversed(range(0, self.steps)), desc="Reverse Process Image Synthesis"):
            XT = ddpm.p_sample(XT, torch.full((self.batch_size,), step, device=device, dtype=torch.long))
            if step % self.series_frequency == 0:
                self.diffusionSubplotting(XT[0], step, ax=axes[counter])  # Plot image at each step
                counter += 1
        plt.savefig(savePath)
        plt.close()
        return XT

    def diffusionSubplotting(self, img, step, ax):
        ax.set_title(f"Time = {step}")
        ax.axis("off")
        if self.numChannels == 1:
            img_data = (img[0].cpu().squeeze().numpy() + 1.0) / 2.0
            img_data = np.clip(img_data, 0, 1)  # Ensure data is in the range [0, 1]
            ax.imshow(img_data, cmap="gray")
        elif self.numChannels == 3:
            img_data = np.transpose((img.cpu().numpy() + 1.0) / 2.0, (1, 2, 0))
            img_data = np.clip(img_data, 0, 1)  # Ensure data is in the range [0, 1]
            ax.imshow(img_data)

# HISTOGRAMS:
    @torch.no_grad()
    def getNoiseHistogram(self, prediction_noise, noise, bins):
        savePath = "images/" + str(self.load_data) + "/" + str(self.scheduler) + "/charts/noise/Epoch = " + str(self.epochCounter) + ""
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
        params = {'model': self.ddpm_model, 'img': XT, 'xlabel': "Noise", 'ylabel': "Frequency",
                  'legend1': "Noise Distribution", 'legend2': "Predicted Distribution", 't': t, 'bins': 128,
                  'title': f"Distribution at Time = {self.steps} {self.scheduler}"}
        self.histogramPlotHelper(ax=axes[0], **params)
        for idx, step in enumerate(
                tqdm(reversed(range(0, self.steps, self.series_frequency)), desc="Plotting Distribution Histograms")):
            XT = ddpm.p_sample(XT, torch.full((self.batch_size,), step, device=device, dtype=torch.long))
            params['title'] = f"Distribution at Time = {step} ({self.scheduler})"
            self.histogramPlotHelper(ax=axes[idx + 1], **params)
        buffer = io.BytesIO()
        plt.savefig(buffer, format='jpg')
        buffer.seek(0)
        save_path = f"images/{self.load_data}/{self.scheduler}/sequence plots/distribution series/Epoch = {self.epochCounter}.jpg"
        with open(save_path, 'wb') as f:
            f.write(buffer.read())
        plt.close()


# GIFS:
    @torch.no_grad()
    def getDiffuionGif(self, XT):
        convertToImage = ConvertToImage()
        fig, ax = plt.subplots()
        imgs = []
        display_mode = 'gray' if self.numChannels == 1 else 'rgb'
        XT = torch.randn_like(XT, device=device)  # Noise for initial image
        for step in tqdm(reversed(range(0, self.steps)), desc="Generating Diffusion Process Gif"):
            XT = ddpm.p_sample(XT, torch.full((self.batch_size,), step, device=device, dtype=torch.long))
            img_np = convertToImage(XT[0])
            if display_mode == 'gray':
                imgs.append([ax.imshow(img_np, cmap="gray")])
            else:
                imgs.append([ax.imshow(img_np)])
        imgs.extend([imgs[-1]] * 500)
        ax.set_title("" + str(self.load_data) + " diffusion")
        ax.axis("off")
        animate = ArtistAnimation(fig, imgs, interval=30, blit=True, repeat_delay=50, repeat=True)
        animate.save(
            "images/" + str(self.load_data) + "/" + str(self.scheduler) + "/" + str(self.load_data) + " diffusion.gif")
        plt.close(fig)