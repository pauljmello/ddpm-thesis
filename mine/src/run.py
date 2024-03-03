import os
import mine
import utils
import matplotlib.pyplot as plt
import numpy as np
import torch


seed = 3407  # https://arxiv.org/abs/2109.08203
np.random.seed(seed)
torch.manual_seed(seed)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.cuda.set_device(torch.cuda.current_device())

args = utils.get_args()

model_directories = []
image_directory = f"../images/{args.example}/"

if args.example == "gaussian":
    gaussian_model_dir = f"{args.d}D"
    model_directories = [f"../models/{args.example}/{gaussian_model_dir}/",
                         f"../../models/mine model/{args.example}/{gaussian_model_dir}/"]
else:
    model_directories = [f"../models/{args.example}/", f"../../models/mine model/{args.example}/"]

for directory in model_directories:
    os.makedirs(directory, exist_ok=True)
os.makedirs(image_directory, exist_ok=True)


if args.example == "gaussian":

    sampleSize = 250_000

    n_rhos = args.n_rhos
    d = args.d
    MI_real = np.empty(n_rhos)
    MI_mine = np.empty(n_rhos)
    mine_network = None

    rhos = np.linspace(-0.99, 0.99, n_rhos)

    for i, rho in enumerate(rhos):

        print(f"Computing MI for rho {rho:.2f}")
        
        # Compute the real MI
        cov = np.eye(2*d)
        cov[d:2*d, 0:d] = rho * np.eye(d) 
        cov[0:d, d:2*d] = rho * np.eye(d)
        MI_real[i] = - 0.5 * np.log(np.linalg.det(cov)) / np.log(2)
        
        # Compute the estimation of the MI 
        dataset = utils.BiVariateGaussianDatasetForMI(d, rho, sampleSize)
        mine_network = mine.MINE(d, d, network_type="mlp", hidden_size=512).to(device)
        tensor_val = mine_network.train(dataset, learning_rate=args.learning_rate, batch_size=args.batch_size, n_iterations=args.n_iterations, n_verbose=args.n_verbose, n_window=args.n_window, save_progress=args.save_progress, dims=args.d, args=args)
        MI_mine[i] = tensor_val[-1].item()
        print(f"Rho {rho:.2f}: Real ({MI_real[i]:.3f}) - Estimated ({MI_mine[i]:.3f})")

    plt.plot(rhos, MI_real, "black", label="Real MI")
    plt.plot(rhos, MI_mine, "orange", label="Estimated MI")
    plt.legend()
    plt.title(f"{d}D Gaussian MI")
    plt.savefig(f"../images/{args.example}/{d}D Gaussian MI.png")
    torch.save(mine_network, f"../../mine/models/gaussian/{d}D Gaussian MI E{args.n_iterations}.pt")
    torch.save(mine_network, f"../../models/mine model/gaussian/{d}D Gaussian MI E{args.n_iterations}.pt")


if args.example == "mnist":
    dataset = utils.MNISTForMI()
    mine_network = mine.MINE((28, 28), 1, network_type="cnn").to(device)
    MI_mine = mine_network.train(dataset, learning_rate=args.learning_rate, batch_size=args.batch_size, n_iterations=args.n_iterations, n_verbose=args.n_verbose, n_window=args.n_window, save_progress=args.save_progress, dims=args.d, args=args)
    iterations = np.linspace(args.save_progress, args.n_iterations, int(args.n_iterations/args.save_progress))
    plt.plot(iterations, np.log2(10)*np.ones_like(iterations), "black", label="Real MI")
    plt.plot(iterations, MI_mine, "orange", label="Estimated MI")
    plt.legend()
    plt.title(f"mnist MI")
    plt.savefig(f"../images/{args.example}/{args.example} MI.png")
    torch.save(mine_network, f"../models/mnist/MI E{args.n_iterations}.pt")
    torch.save(mine_network, f"../../models/mine model/mnist/MI E{args.n_iterations}.pt")

if args.example == "fashion-mnist":
    dataset = utils.FashionMNISTForMI()
    mine_network = mine.MINE((28, 28), 1, network_type="cnn").to(device)
    MI_mine = mine_network.train(dataset, learning_rate=args.learning_rate, batch_size=args.batch_size, n_iterations=args.n_iterations, n_verbose=args.n_verbose, n_window=args.n_window, save_progress=args.save_progress, dims=args.d, args=args)
    iterations = np.linspace(args.save_progress, args.n_iterations, int(args.n_iterations/args.save_progress))
    plt.plot(iterations, np.log2(10)*np.ones_like(iterations), "black", label="Real MI")
    plt.plot(iterations, MI_mine, "orange", label="Estimated MI")
    plt.legend()
    plt.title(f"fashion-mnist MI")
    plt.savefig(f"../images/{args.example}/{args.example} MI.png")
    torch.save(mine_network, f"../models/fashion-mnist/MI E{args.n_iterations}.pt")
    torch.save(mine_network, f"../../models/mine model/fashion-mnist/MI E{args.n_iterations}.pt")

if args.example == "cifar10":
    dataset = utils.CIFAR10ForMI()
    mine_network = mine.MINE((32, 32), 1, network_type="cnnRGB").to(device)
    MI_mine = mine_network.train(dataset, learning_rate=args.learning_rate, batch_size=args.batch_size, n_iterations=args.n_iterations, n_verbose=args.n_verbose, n_window=args.n_window, save_progress=args.save_progress, dims=args.d, args=args)
    iterations = np.linspace(args.save_progress, args.n_iterations, int(args.n_iterations/args.save_progress))
    plt.plot(iterations, np.log2(10)*np.ones_like(iterations), "black", label="Real MI")
    plt.plot(iterations, MI_mine, "orange", label="Estimated MI")
    plt.legend()
    plt.title(f"cifar10 MI")
    plt.savefig(f"../images/{args.example}/{args.example} MI.png")
    torch.save(mine_network, f"../models/cifar10/MI E{args.n_iterations}.pt")
    torch.save(mine_network, f"../../models/mine model/cifar10/MI E{args.n_iterations}.pt")

# TODO: Fix CelebA by creating a new RGB CNN class for it specifically
if args.example == "celeba":
    dataset = utils.CelebAForMI()
    mine_network = mine.MINE((64, 64), 1, network_type="cnnRGB").to(device)
    MI_mine = mine_network.train(dataset, learning_rate=args.learning_rate, batch_size=args.batch_size, n_iterations=args.n_iterations, n_verbose=args.n_verbose, n_window=args.n_window, save_progress=args.save_progress, dims=args.d, args=args)
    iterations = np.linspace(args.save_progress, args.n_iterations, int(args.n_iterations/args.save_progress))
    plt.plot(iterations, np.log2(10)*np.ones_like(iterations), "black", label="Real MI")
    plt.plot(iterations, MI_mine, "orange", label="Estimated MI")
    plt.legend()
    plt.title(f"celeba MI")
    plt.savefig(f"../images/{args.example}/{args.example} MI.png")
    torch.save(mine_network, f"../models/celeba/MI E{args.n_iterations}.pt")
    torch.save(mine_network, f"../../models/mine model/celeba/MI E{args.n_iterations}.pt")

