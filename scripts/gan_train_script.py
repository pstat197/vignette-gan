"""
Train a simple Generative Adversarial Network (GAN) on MNIST and Fashion-MNIST
using PyTorch. This script is the linear, fully reproducible version of the
notebooks in the `notebooks/` folder.
"""

# ----- Imports -----
# Standard library
import os

# Third-party imports: PyTorch, torchvision, matplotlib
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, utils as vutils

import matplotlib.pyplot as plt


# Device configuration
# Use GPU if available; otherwise fall back to CPU.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Hyperparameters
# Size of the latent noise vector z.
Z_DIM = 64

# Image size: MNIST and Fashion-MNIST are 28x28 grayscale.
IMG_SIZE = 28 * 28
IMG_CHANNELS = 1

# Training settings.
NUM_EPOCHS_MNIST = 5        # can increase for better digits
NUM_EPOCHS_FASHION = 5      # can adjust depending on time
BATCH_SIZE = 64
LR = 2e-4                   # learning rate for both networks
PRINT_EVERY = 200           # how often to print training progress

# Output directory for plots and generated images (relative to project root).
IMG_DIR = "img"


# Data loading
def get_dataloader(dataset_name: str, batch_size: int) -> DataLoader:
    """
    Create a DataLoader for either MNIST or Fashion-MNIST.

    Parameters
    ----------
    dataset_name : str
        Either "mnist" or "fashion".
    batch_size : int
        Batch size for training.

    Returns
    -------
    DataLoader
        PyTorch DataLoader for the chosen dataset.
    """
    # Transform: convert images to tensors and scale to [-1, 1].
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),  # from [0,1] to [-1,1]
        ]
    )

    # Choose dataset based on the name.
    if dataset_name.lower() == "mnist":
        dataset = datasets.MNIST(
            root="data",
            train=True,
            download=True,
            transform=transform,
        )
    elif dataset_name.lower() == "fashion":
        dataset = datasets.FashionMNIST(
            root="data",
            train=True,
            download=True,
            transform=transform,
        )
    else:
        raise ValueError("dataset_name must be 'mnist' or 'fashion'.")

    # Wrap in DataLoader for batching and shuffling.
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    return loader


# Model definitions
class Generator(nn.Module):
    """
    Generator network.

    Takes random noise (z) as input and outputs a fake image tensor of shape
    (batch_size, 1, 28, 28). This is the same simple MLP structure as in the
    Intro_GAN notebook.
    """

    def __init__(self, z_dim: int, img_size: int):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(z_dim, 128),
            nn.ReLU(True),
            nn.Linear(128, 256),
            nn.ReLU(True),
            nn.Linear(256, img_size),
            nn.Tanh(),  # output values in [-1, 1]
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # Pass the noise through the MLP.
        x = self.model(z)
        # Reshape to image tensor (batch_size, channels, height, width).
        return x.view(-1, IMG_CHANNELS, 28, 28)


class Discriminator(nn.Module):
    """
    Discriminator network.

    Takes an image as input and outputs a single probability in [0, 1]:
    - 1 means "real"
    - 0 means "fake"
    """

    def __init__(self, img_size: int):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(img_size, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, 1),
            nn.Sigmoid(),  # output probability
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Flatten image to a vector of length img_size.
        x = x.view(x.size(0), -1)
        return self.model(x)


# Training function
def train_gan(
    dataset_name: str,
    num_epochs: int,
    z_dim: int = Z_DIM,
    lr: float = LR,
    batch_size: int = BATCH_SIZE,
    print_every: int = PRINT_EVERY,
):
    """
    Train a GAN on the specified dataset.

    Parameters
    ----------
    dataset_name : str
        "mnist" or "fashion".
    num_epochs : int
        Number of training epochs.
    z_dim : int
        Dimension of the noise vector.
    lr : float
        Learning rate for both generator and discriminator.
    batch_size : int
        Batch size.
    print_every : int
        How often (in batches) to print training status.

    Returns
    -------
    generator : nn.Module
        Trained generator model.
    G_losses : list[float]
        Generator loss values recorded during training.
    D_losses : list[float]
        Discriminator loss values recorded during training.
    fixed_noise : torch.Tensor
        Fixed noise used for visualizing training progress.
    """
    # Create DataLoader for chosen dataset.
    dataloader = get_dataloader(dataset_name, batch_size)

    # Instantiate generator and discriminator, then move them to device.
    generator = Generator(z_dim, IMG_SIZE).to(device)
    discriminator = Discriminator(IMG_SIZE).to(device)

    # Loss function: Binary Cross Entropy for real vs fake.
    criterion = nn.BCELoss()

    # Optimizers for both networks.
    optimizer_G = optim.Adam(generator.parameters(), lr=lr)
    optimizer_D = optim.Adam(discriminator.parameters(), lr=lr)

    # Fixed noise vector for visualizing progress across epochs.
    fixed_noise = torch.randn(32, z_dim).to(device)

    # Lists to store loss values for plotting later.
    G_losses = []
    D_losses = []

    step = 0  # counts training steps across epochs

    print(f"Starting training on {dataset_name} for {num_epochs} epochs...")

    # Outer loop: epochs.
    for epoch in range(num_epochs):
        # Inner loop: mini-batches from the DataLoader.
        for batch_idx, (real_imgs, _) in enumerate(dataloader):
            # Move real images to the selected device.
            real_imgs = real_imgs.to(device)
            batch_size_curr = real_imgs.size(0)

            # 1. Train Discriminator
            # Zero out discriminator gradients.
            optimizer_D.zero_grad()

            # Real images: label = 1.
            labels_real = torch.ones(batch_size_curr, 1).to(device)
            output_real = discriminator(real_imgs)
            loss_D_real = criterion(output_real, labels_real)

            # Fake images: label = 0.
            noise = torch.randn(batch_size_curr, z_dim).to(device)
            fake_imgs = generator(noise)
            labels_fake = torch.zeros(batch_size_curr, 1).to(device)
            # Detach so gradients do not flow to generator when training D.
            output_fake = discriminator(fake_imgs.detach())
            loss_D_fake = criterion(output_fake, labels_fake)

            # Total discriminator loss: sum of real and fake losses.
            loss_D = loss_D_real + loss_D_fake
            # Backpropagate and update discriminator parameters.
            loss_D.backward()
            optimizer_D.step()

            # 2. Train Generator
            # Zero out generator gradients.
            optimizer_G.zero_grad()

            # Generate new fake images.
            noise = torch.randn(batch_size_curr, z_dim).to(device)
            fake_imgs = generator(noise)

            # For the generator, we want the discriminator to think
            # the fake images are real, so we use label = 1.
            labels_for_generator = torch.ones(batch_size_curr, 1).to(device)
            output = discriminator(fake_imgs)
            loss_G = criterion(output, labels_for_generator)

            # Backpropagate and update generator parameters.
            loss_G.backward()
            optimizer_G.step()

            # Record losses for plotting.
            G_losses.append(loss_G.item())
            D_losses.append(loss_D.item())

            # Print training progress occasionally.
            if batch_idx % print_every == 0:
                print(
                    f"[{dataset_name}] "
                    f"Epoch [{epoch + 1}/{num_epochs}] "
                    f"Batch [{batch_idx}/{len(dataloader)}] "
                    f"Loss D: {loss_D.item():.4f}, Loss G: {loss_G.item():.4f}"
                )

            step += 1

    print(f"Training on {dataset_name} complete.")
    return generator, G_losses, D_losses, fixed_noise


# Helper functions to save outputs
def ensure_img_dir():
    """Create the image directory if it does not already exist."""
    os.makedirs(IMG_DIR, exist_ok=True)


def save_loss_plot(G_losses, D_losses, dataset_name: str):
    """Save a plot of generator and discriminator losses."""
    ensure_img_dir()
    plt.figure(figsize=(6, 4))
    plt.plot(G_losses, label="Generator loss")
    plt.plot(D_losses, label="Discriminator loss")
    plt.xlabel("Training step")
    plt.ylabel("Loss")
    plt.title(f"GAN Training Losses ({dataset_name})")
    plt.legend()
    out_path = os.path.join(IMG_DIR, f"losses_{dataset_name}.png")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"Saved loss plot to {out_path}")


def save_generated_samples(generator, fixed_noise, dataset_name: str):
    """Save a grid of generated sample images."""
    ensure_img_dir()
    generator.eval()
    with torch.no_grad():
        fake_samples = generator(fixed_noise).cpu()

    # Convert from [-1, 1] back to [0, 1] and make a grid.
    fake_samples = (fake_samples + 1) / 2
    grid = vutils.make_grid(fake_samples[:16], nrow=4, padding=2)

    plt.figure(figsize=(6, 6))
    plt.axis("off")
    plt.title(f"Generated samples ({dataset_name})")
    plt.imshow(grid.permute(1, 2, 0).numpy(), cmap="gray")
    out_path = os.path.join(IMG_DIR, f"samples_{dataset_name}.png")
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"Saved generated samples to {out_path}")
    generator.train()


# Main entry point
if __name__ == "__main__":
    # Train on MNIST (digits).
    gen_mnist, G_mnist, D_mnist, fixed_noise_mnist = train_gan(
        dataset_name="mnist",
        num_epochs=NUM_EPOCHS_MNIST,
    )
    save_loss_plot(G_mnist, D_mnist, dataset_name="mnist")
    save_generated_samples(gen_mnist, fixed_noise_mnist, dataset_name="mnist")

    # Train on Fashion-MNIST (clothing).
    gen_fashion, G_fashion, D_fashion, fixed_noise_fashion = train_gan(
        dataset_name="fashion",
        num_epochs=NUM_EPOCHS_FASHION,
    )
    save_loss_plot(G_fashion, D_fashion, dataset_name="fashion")
    save_generated_samples(gen_fashion, fixed_noise_fashion, dataset_name="fashion")