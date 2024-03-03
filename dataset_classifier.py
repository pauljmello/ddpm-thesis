import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import matplotlib.pyplot as plt
from tqdm import tqdm

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.cuda.set_device(torch.cuda.current_device())

model = None

def save_classification_model(model, noise, dataset_name, epochs):
    if noise:
        file_path = f"./models/classifier model/noisy {dataset_name} classifier E{epochs}.pt"
    else:
        file_path = f"./models/classifier model/{dataset_name} classifier E{epochs}.pt"
    torch.save(model, file_path)
    print(f"Saved model to {file_path}")

# Function to load datasets
def load_datasets(dataset_name, batch_size):
    if dataset_name in ["mnist", "fashion-mnist", "emnist"]:
        transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5,), (0.5,))])  # Normalize
    elif dataset_name == "cifar10":
        transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])  # Normalize

    if dataset_name == "mnist":
        Dataset = torchvision.datasets.MNIST
    elif dataset_name == "fashion-mnist":
        Dataset = torchvision.datasets.FashionMNIST
    elif dataset_name == "emnist":
        Dataset = lambda **kwargs: torchvision.datasets.EMNIST(split='digits', **kwargs)
    elif dataset_name == "cifar10":
        Dataset = torchvision.datasets.CIFAR10

    train = Dataset(root='./datasets', train=True, download=True, transform=transform)
    training_data = torch.utils.data.DataLoader(train, batch_size=batch_size, pin_memory=True, num_workers=8, pin_memory_device="cuda", persistent_workers=True)

    test = Dataset(root='./datasets', train=False, download=True, transform=transform)
    testing_data = torch.utils.data.DataLoader(test, batch_size=batch_size, pin_memory=True, num_workers=8, pin_memory_device="cuda", persistent_workers=True)

    return training_data, testing_data


# Display images
def imshow(img, dataset_name):
    img = img * 0.5 + 0.5
    if dataset_name in ["mnist", "fashion-mnist", "emnist"]:
        plt.imshow(torchvision.utils.make_grid(img).permute(1, 2, 0), cmap='gray')
    elif dataset_name == "cifar10":
        plt.imshow(torchvision.utils.make_grid(img).permute(1, 2, 0))
    plt.show()


# Function to train the network
def train_network(model, training_data, epochs, optimizer, criterion, noise_bool):
    model.to(device)

    for epoch in range(epochs):
        running_loss = 0.0
        with tqdm(enumerate(training_data), total=len(training_data)) as t:
            for i, data in t:
                inputs, labels = data[0].to(device), data[1].to(device)
                batch_size = inputs.size(0)

                if noise_bool:
                    scaling_factors = np.random.uniform(0.01, 0.99, size=batch_size).reshape(-1, 1, 1, 1)
                    noise = torch.randn_like(inputs) * torch.tensor(scaling_factors, dtype=torch.float32, device=inputs.device)
                    batched_training_data = inputs + noise
                else:
                    batched_training_data = inputs

                optimizer.zero_grad()
                outputs = model(batched_training_data)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                t.set_description(f'Epoch {epoch + 1}/{epochs}')

        print("\033[K", end='\r')  # Clear tqdm line
        average_loss = running_loss / len(training_data)
        print(f"Loss: {average_loss:.6f}\n")
    print("Finished Training")


# Function to test the network
def test_network(model, testing_data, batch_size, dataset_name):
    model.to(device)

    # Define classes based on the dataset
    if dataset_name in ["mnist", "fashion-mnist", "emnist"]:
        classes = [str(i) for i in range(10)]  # 10 classes: 0-9
    elif dataset_name == "cifar10":
        classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']  # CIFAR-10 classes
    else:
        classes = []  # For other datasets, define class labels accordingly

    class_correct = list(0. for i in range(len(classes)))
    class_total = list(0. for i in range(len(classes)))

    with torch.no_grad():
        for data in testing_data:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            c = (predicted == labels).squeeze()
            for i in range(min(batch_size, labels.size(0))):
                label = labels[i].item()
                class_correct[label] += c[i].item()
                class_total[label] += 1

    # Print accuracy for each class
    for i in range(len(classes)):
        if class_total[i] > 0:
            print(f"Accuracy of {classes[i]} : {100 * class_correct[i] / class_total[i]:.2f} %")
        else:
            print(f"Accuracy of {classes[i]} : N/A (no samples)")

    for i in range(10):
        print("Accuracy of %5s : %2d %%" % (classes[i], 100 * class_correct[i] / class_total[i]))


def main():
    dataset_name = "emnist"  # "mnist, "fashion-mnist", 'emnist' | "cifar10"    | 'emnist' is set to digits only
    epochs = 1000
    batch_size = 512
    noised_training_data = True
    model = None

    if dataset_name == "mnist" or dataset_name == "fashion-mnist" or dataset_name == "emnist":
        model = Net().to(device)
    elif dataset_name == "cifar10":
        model = NetRGB().to(device)
    else:
        print("Invalid dataset name")
        exit()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.0001, eps=1e-08)

    training_data, testing_data = load_datasets(dataset_name, batch_size)
    train_network(model, training_data, epochs, optimizer, criterion, noised_training_data)
    test_network(model, testing_data, batch_size, dataset_name)

    save_classification_model(model, noised_training_data, dataset_name, epochs)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv4 = nn.Conv2d(128, 256, 3, padding=1)

        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)

        self.dropout = nn.Dropout(0.2)

        self.fc1 = nn.Linear(256, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.pool(F.relu(self.bn4(self.conv4(x))))
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class NetRGB(nn.Module):
    def __init__(self):
        super(NetRGB, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv4 = nn.Conv2d(128, 256, 3, padding=1)

        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)

        self.dropout = nn.Dropout(0.2)

        self.fc1 = nn.Linear(256, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.pool(F.relu(self.bn4(self.conv4(x))))
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


if __name__ == '__main__':
    main()
