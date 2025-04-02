import torch
import torchvision
from torchvision import transforms


def load_mnist_dataset():
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Lambda(lambda x: x.flatten())]
    )
    dataset = torchvision.datasets.MNIST(
        root="data", train=True, download=True, transform=transform
    )
    return dataset


def load_cifar10_dataset():
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Lambda(lambda x: x.flatten())]
    )
    dataset = torchvision.datasets.FashionMNIST(
        root="data", train=True, download=True, transform=transform
    )

    return dataset


def load_cifar100_dataset():
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Lambda(lambda x: x.flatten())]
    )
    dataset = torchvision.datasets.CIFAR100(
        root="data", train=True, download=True, transform=transform
    )
    return dataset


def determine_input_size(dataset):
    input_size = 1
    for shape in dataset.data[0].shape:
        input_size *= shape

    return input_size


def prepare_data(
    dataset,
    num_imgs=5,
    preprocess_sensory=True,
    noise_level="medium",
    across_dataset=True,
    device=None,
):
    data = dataset.train_data.flatten(1)[:num_imgs].float().to(device)
    if preprocess_sensory:
        if across_dataset:
            data = (data - data.mean(dim=0)) / (data.std(dim=0) + 1e-8)
        else:
            for i in range(len(data)):
                data[i] = (data[i] - data[i].mean()) / data[i].std()

        # noising the data
        # data = data.float()
    if noise_level == "none":
        return data, data
    elif noise_level == "low":
        random_noise = torch.zeros_like(data).uniform_(-1, 1)
    elif noise_level == "medium":
        random_noise = torch.zeros_like(data).uniform_(-1.25, 1.25)
    elif noise_level == "high":
        random_noise = torch.zeros_like(data).uniform_(-1.5, 1.5)
    noisy_data = data + random_noise

    return data, noisy_data


def prepare_random_data(noise_scale=0.1, num_imgs=5):
    data = torch.randn((num_imgs, 784))
    noise = noise_scale * torch.randn((num_imgs, 784))
    return data, data + noise


def load_prepare_mnist_data(
    num_imgs=5,
    preprocess_sensory=True,
    noise_level="medium",
    use_fix=False,
):
    dataset = load_mnist_dataset()
    return prepare_data(
        dataset,
        num_imgs=num_imgs,
        preprocess_sensory=preprocess_sensory,
        noise_level=noise_level,
        use_fix=use_fix,
    )
