import io
import os
import tqdm
import base64
import pickle
import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt

from PIL import Image
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, MDS, Isomap

from dataset_classifier import Net, NetRGB

SEED = 0
np.random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.backends.cudnn.deterministic = torch.cuda.is_available()
torch.backends.cudnn.benchmark = False

print(f"Using device: {DEVICE}")


class ManifoldVisualization:
    def __init__(self, dataset):
        self.dataset = dataset
        self.model = None

    def create_directory_structure(self, dataset_name, epochs, dimensions, method):
        base_dir = f"models/manifold model/{dataset_name}/{method}/{dimensions}"
        os.makedirs(os.path.join(base_dir, "base"), exist_ok=True)
        for epoch in epochs:
            os.makedirs(os.path.join(base_dir, f"E{epoch}"), exist_ok=True)

    def save_model(self, model, dataset_name, num_samples, dimensions, method, epoch=None):
        base_dir = f"models/manifold model/{dataset_name}/{method}/{dimensions}"
        model_file_name = f"{dataset_name}_{num_samples}_model.pkl"
        model_file_path = os.path.join(base_dir, 'base' if epoch is None else f"E{epoch}", model_file_name)
        with open(model_file_path, 'wb') as file:
            pickle.dump(model, file)
        print(f"{method} model saved at {model_file_path}")

    def apply_reduction(self, features, dimensions, method, perplexity=15, n_neighbors=5):
        if method == 'pca':
            model = PCA(n_components=dimensions)
        elif method == 'tsne':
            model = TSNE(n_components=dimensions, perplexity=perplexity, random_state=SEED)
        elif method == 'mds':
            model = MDS(n_components=dimensions, random_state=SEED)
        elif method == 'isomap':
            model = Isomap(n_components=dimensions, n_neighbors=n_neighbors)
        else:
            raise ValueError(f"Unknown method: {method}")

        transformed_features = model.fit_transform(features)
        return transformed_features

    def load_model(self, dataset_name, num_samples, dimensions, method):
        model_file_path = f"models/manifold model/{dataset_name}/{method}/{dimensions}/base/{dataset_name}_{num_samples}_model.pkl"
        if os.path.exists(model_file_path):
            with open(model_file_path, 'rb') as file:
                self.model = pickle.load(file)
            print(f"Model loaded from {model_file_path}")
        else:
            print(f"Model file not found: {model_file_path}")
            self.model = None

    def visualize_3d_plotly(self, dataset_name, original_features, new_features, base64_images, original_labels, new_labels, new_colors, new_hover_texts=None, cluster_means=None, cluster_std_devs=None, distances_to_true_centers=None):
        fig = go.Figure()
        cmap = plt.get_cmap('tab10')

        # Original Data
        fig.add_trace(
            go.Scatter3d(
                x=original_features[:, 0],
                y=original_features[:, 1],
                z=original_features[:, 2],
                name=f"{dataset_name} Data",
                mode='markers',
                marker=dict(size=5, color=[cmap(label) for label in original_labels], opacity=0.8),
                hovertext=[f'Data: {dataset_name}, Label: {label}' for label in original_labels]))

        if new_features.size > 0:
            hover_data = [f'Data: Diffusion, True Label: {label} (Dist: {dist:.2f}), {hover_text}' for label, dist, hover_text in zip(new_labels, distances_to_true_centers, new_hover_texts)]
            fig.add_trace(
                go.Scatter3d(
                    x=new_features[:, 0],
                    y=new_features[:, 1],
                    z=new_features[:, 2],
                    name="Diffusion Data: Reverse Trajectory",
                    mode='markers',
                    marker=dict(size=5, color=new_colors, opacity=0.8),
                    hovertext=hover_data,
                    customdata=base64_images))

        if cluster_means is not None and cluster_std_devs is not None:
            for label, center in cluster_means.items():
                std_dev = cluster_std_devs[label]
                mean_x, mean_y, mean_z = center
                center_str = f"{mean_x:.2f}, {mean_y:.2f}, {mean_z:.2f}"
                std_dev_str = f"{std_dev[0]:.2f}, {std_dev[1]:.2f}, {std_dev[2]:.2f}"
                fig.add_trace(
                    go.Scatter3d(
                        x=[mean_x],
                        y=[mean_y],
                        z=[mean_z],
                        mode='markers',
                        marker=dict(size=15, color='black', symbol='diamond'),  # Black color, diamond shape
                        name=f"Cluster Center",
                        hovertext=f"Data: Center, Label: {label}, Mean: {center_str}, Deviation: {std_dev_str}",
                        showlegend=False
                    ))

        # Update layout
        fig.update_layout(
            legend=dict(
                y=0.5,
                x=1),
            margin=dict(l=0, r=0, b=0, t=0),
            scene=dict(aspectratio=dict(x=1, y=1, z=0.7), aspectmode='manual'))

        fig.show()

    def sample_dataset(self, num_samples, dataset_name):
        dataset_size = len(self.dataset)
        if num_samples > dataset_size:
            print(f"Warning: Requested num_samples ({num_samples}) exceeds the dataset size ({dataset_size}). Sampling {dataset_size} data points from {dataset_name}.")
            num_samples = dataset_size

        indices = torch.randperm(dataset_size)[:num_samples]
        return np.array([self.dataset[idx][0].view(-1).numpy() for idx in indices]), np.array([self.dataset[idx][1] for idx in indices]), np.array([self.dataset[idx][0].numpy() for idx in indices])


def calculate_distances(new_data_points, cluster_means, true_labels, predicted_labels):
    distances_to_true_centers = []
    distances_to_predicted_centers = []

    for point, true_label, predicted_label in zip(new_data_points, true_labels, predicted_labels):
        # Distance to the center of the true label cluster
        true_center = cluster_means[true_label]
        distance_to_true_center = np.linalg.norm(point - true_center)
        distances_to_true_centers.append(distance_to_true_center)

        # Distance to the center of the predicted label cluster
        predicted_center = cluster_means[predicted_label]
        distance_to_predicted_center = np.linalg.norm(point - predicted_center)
        distances_to_predicted_centers.append(distance_to_predicted_center)

    return np.array(distances_to_true_centers), np.array(distances_to_predicted_centers)


def compute_cluster_statistics(features, labels, std_threshold):
    unique_labels = np.unique(labels)
    cluster_means = {}
    cluster_std_devs = {}
    for label in unique_labels:
        cluster_features = features[labels == label]
        mean = np.mean(cluster_features, axis=0)
        std_dev = np.std(cluster_features, axis=0)
        z_scores = np.abs((cluster_features - mean) / std_dev)
        filtered_features = cluster_features[np.all(z_scores < std_threshold, axis=1)]
        cluster_means[label] = np.mean(filtered_features, axis=0)
        cluster_std_devs[label] = np.std(filtered_features, axis=0)
    return cluster_means, cluster_std_devs


def determine_colors(labels, color_setting):
    cmap = plt.get_cmap('tab10')
    if color_setting == 'labels':
        return [cmap(label) for label in labels]
    elif color_setting == 'black':
        return ['black'] * len(labels)
    elif color_setting == 'gradient':
        max_label = max(labels)
        return [f'rgb({int(255 * label / max_label)}, 0, {int(255 * (1 - label / max_label))})' for label in labels]
    return ['rgb(128, 128, 128)'] * len(labels)


def img_to_base64_str(img):
    pil_img = Image.fromarray((img.squeeze() * 255).astype(np.uint8))
    buff = io.BytesIO()
    pil_img.save(buff, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buff.getvalue()).decode("utf-8")


def calculate_color(step, max_step, class_num, color_mode):
    if color_mode == 'gradient':
        gradient_value = int(255 * step / max_step)
        return f'rgb({gradient_value}, {gradient_value}, {gradient_value})'
    elif color_mode == 'black':
        return 'black'
    elif color_mode == 'labels':
        cmap = plt.get_cmap('tab10')
        label_color = cmap(class_num)
        return plt.colors.rgb2hex(label_color)


def add_images_to_manifold(manifold_data, classifier_model, sampled_features, sampled_labels, cluster_means, cluster_std_devs, dataset_name, trajectory_dataset, dimensions, epoch, classes, timesteps, scheduler, modulus, color, method):
    new_data_points = []
    new_labels = []  # True labels
    new_colors = []
    new_hover_texts = []
    predicted_labels = []  # Predicted labels
    image_cache = {}

    # Ensure the model is on GPU and set to evaluation mode
    classifier_model.to(DEVICE)
    classifier_model.eval()

    for timestep in timesteps:
        for class_num in classes:
            dir_path = os.path.join("images", trajectory_dataset, scheduler, "manifold", f"E{epoch}", f"T{timestep}", f"class_{class_num}")
            step_files = [f for f in os.listdir(dir_path) if f.startswith('step_') and f.endswith('.png')]
            steps = sorted([int(f.split('_')[1].split('.')[0]) for f in step_files])

            if not steps:
                continue

            max_step = steps[-1]
            for step in steps:
                if step % modulus == 0:
                    image_path = os.path.join(dir_path, f"step_{step}.png")

                    if image_path in image_cache:
                        image_tensor = image_cache[image_path]
                    else:
                        try:
                            with Image.open(image_path) as image:
                                image = image.convert('L').resize((28, 28))
                                image_tensor = transforms.ToTensor()(image).view(1, 1, 28, 28).to(DEVICE)
                                image_cache[image_path] = image_tensor
                        except Exception as e:
                            print(f"Error processing {image_path}: {e}")
                            continue

                    prediction = classifier_model(image_tensor)
                    predicted_class = torch.argmax(prediction, dim=1).item()
                    predicted_labels.append(predicted_class)

                    new_data_points.append(image_tensor.cpu().numpy().flatten())
                    new_labels.append(class_num)
                    new_hover_texts.append(f"Step: {step}, Series: {max_step + 1}, Epoch: {epoch}")

                    new_colors.append(calculate_color(step, max_step, class_num, color))

    if not new_data_points:
        print("No new data points found.")
        return

    new_data_points = np.array(new_data_points)
    new_labels = np.array(new_labels)
    predicted_labels = np.array(predicted_labels)

    combined_features = np.vstack([sampled_features, new_data_points])
    combined_transformed_features = manifold_data.apply_reduction(combined_features, dimensions, method)

    original_transformed_features = combined_transformed_features[:len(sampled_features)]
    new_transformed_features = combined_transformed_features[len(sampled_features):]

    # Calculate distances to true and predicted label centers
    distances_to_true_centers, distances_to_predicted_centers = calculate_distances(new_transformed_features, cluster_means, new_labels, predicted_labels)

    for i in range(len(new_data_points)):
        step_info = f"Pred Label: {predicted_labels[i]} (Dist: {distances_to_predicted_centers[i]:.2f}), "
        new_hover_texts[i] = step_info + new_hover_texts[i]

    new_base64_images = [img_to_base64_str(img.reshape(28, 28, 1)) for img in new_data_points]

    manifold_data.visualize_3d_plotly(dataset_name, original_transformed_features, new_transformed_features, new_base64_images, sampled_labels, new_labels, new_colors, new_hover_texts, cluster_means, cluster_std_devs, distances_to_true_centers)


def load_dataset(name):
    transform = transforms.ToTensor()

    dataset_options = {
        'mnist': lambda **kwargs: torchvision.datasets.MNIST(root='./datasets', train=True, download=True, transform=transform, **kwargs),
        'fashionmnist': lambda **kwargs: torchvision.datasets.FashionMNIST(root='./datasets', train=True, download=True, transform=transform, **kwargs),
        'cifar': lambda **kwargs: torchvision.datasets.CIFAR10(root='./datasets', train=True, download=True, transform=transform, **kwargs),
        'emnist': lambda **kwargs: torchvision.datasets.EMNIST(root='./datasets', split='digits', download=True, transform=transform, **kwargs)  # 'train' argument is not needed for EMNIST
    }

    if name not in dataset_options:
        raise ValueError(f"Dataset {name} not supported.")

    return dataset_options[name]()


def prepare_visualization_data(manifold_data, sampled_features, sampled_labels, sampled_images, dimensions, method):
    transformed_features = manifold_data.apply_reduction(sampled_features, dimensions, method)
    mnist_colors = determine_colors(sampled_labels, 'labels')
    base64_images = [img_to_base64_str(img) for img in sampled_images]
    return transformed_features, mnist_colors, base64_images


def process_epochs(manifold_data, classifier_model, sampled_features, sampled_labels, cluster_means, cluster_std_devs, dataset_name, trajectory_dataset, num_samples, dimensions, epochs, classes, timesteps, scheduler, modulus, color, method):
    for epoch in tqdm.tqdm(epochs, desc="Processing Epoch Data"):
        add_images_to_manifold(manifold_data, classifier_model, sampled_features, sampled_labels, cluster_means, cluster_std_devs, dataset_name, trajectory_dataset, dimensions, epoch, classes, timesteps, scheduler, modulus, color, method)
        manifold_data.save_model(manifold_data.model, dataset_name, num_samples, dimensions, method, epoch=epoch)


def main():
    CONFIG = {
        'noisy_classification_model': True,
        'generate_all_at_once': False,
        'use_precomputed_plot': False,
        'method': 'tsne',  # "pca", "tsne", "mds", "isomap"
        'dimensions': 10 if 'method' == 'pca' else 3,

        # mnist is a subset of emnist. so if you train ddpm on mnist or emnist you can use the classifcation model for either because we only train the models on digits.
        'dataset_name': 'mnist',  # "mnist", "cifar", "fashionmnist", "emnist
        'trajectory_dataset': 'mnist',  # "mnist", "cifar", "fashionmnist", "emnist"              |   'emnist' is set to digits only
        'classification_model': 'mnist',  # "mnist", "cifar", "fashionmnist", "emnist"     |   'emnist' is set to digits only

        'num_samples': 15000,  # Number of samples to use from the dataset                  |   'mnist': 60000, 'cifar': 50000, 'fashionmnist': 60000, 'emnist': (240,000 digits)

        'epochs': [1000],  # Epochs to use for visualization  |  Full: [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
        'timesteps': [100, 500, 1000, 5000],  # Timesteps to use for visualization  |  Full: [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1500, 2000, 3000, 4000, 5000]
        'classes': [0],  # Classes to use for visualization  |  Full: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

        'color': 'gradient',  # "gradient", "black", "labels"
        'plot_data_every_num_steps': 5,

        'scheduler': 'linear',
    }

    std_threshold = 3

    try:
        if CONFIG['noisy_classification_model']:
            classifier_model = torch.load(f"././models/classifier model/noisy {CONFIG['classification_model']} classifier E1000.pt").to(DEVICE)
            print(f"\nNoisy {CONFIG['dataset_name'].capitalize()} Classifier Loaded\n")
        else:
            classifier_model = torch.load(f"././models/classifier model/{CONFIG['classification_model']} classifier E1000.pt").to(DEVICE)
            print(f"\n{CONFIG['dataset_name'].capitalize()} Classifier Loaded\n")
    except FileNotFoundError:
        print(f"Classifier model not found for {CONFIG['dataset_name'].capitalize()} @ 1000 Epochs. Stopping execution.")
        exit()

    classifier_model.eval()

    manifold_data = ManifoldVisualization(load_dataset(CONFIG['dataset_name']))
    manifold_data.create_directory_structure(CONFIG['dataset_name'], CONFIG['epochs'], CONFIG['dimensions'], CONFIG['method'])

    sampled_features, sampled_labels, sampled_images = manifold_data.sample_dataset(CONFIG['num_samples'], CONFIG['dataset_name'])
    CONFIG['num_samples'] = len(sampled_features)
    manifold_data.save_model(manifold_data, CONFIG['dataset_name'], CONFIG['num_samples'], CONFIG['dimensions'], CONFIG['method'], epoch=None)

    if CONFIG['generate_all_at_once'] and CONFIG['use_precomputed_plot']:
        print("Cannot use precomputed plot when generate_all_at_once is True.")
    elif CONFIG['generate_all_at_once']:
        label_colors = determine_colors(sampled_labels, 'labels')
        combined_features, combined_labels = sampled_features, sampled_labels
        base64_images = [img_to_base64_str(img) for img in sampled_images]
    elif CONFIG['generate_all_at_once']:
        manifold_data.load_model(CONFIG['dataset_name'], CONFIG['num_samples'], CONFIG['dimensions'], CONFIG['method'])
        if manifold_data.model is None:
            print("Base model not found. Stopping execution.")
            exit()
    else:
        features, labels = np.array([]), np.array([])

    transformed_features, label_colors, base64_images = prepare_visualization_data(manifold_data, sampled_features, sampled_labels, sampled_images, CONFIG['dimensions'], CONFIG['method'])
    cluster_means, cluster_std_devs = compute_cluster_statistics(transformed_features, sampled_labels, std_threshold)
    # manifold_data.visualize_3d_plotly(transformed_features, np.array([]).reshape(0, CONFIG['dimensions']), base64_images, sampled_labels, [], label_colors)
    process_epochs(manifold_data, classifier_model, sampled_features, sampled_labels, cluster_means, cluster_std_devs, CONFIG['dataset_name'], CONFIG['trajectory_dataset'], CONFIG['num_samples'],
                   CONFIG['dimensions'], CONFIG['epochs'], CONFIG['classes'],CONFIG['timesteps'], CONFIG['scheduler'], CONFIG['plot_data_every_num_steps'], CONFIG['color'], CONFIG['method'])


if __name__ == '__main__':
    main()
