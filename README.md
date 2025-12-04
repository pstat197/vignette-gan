# Generative Adversarial Networks: An Introductory Vignette


## Overview

This repository provides an accessible, introductory walkthrough of Generative Adversarial Networks (GANs) using two classic image datasets: MNIST and Fashion-MNIST. The goal of the vignette is to help students intuitively understand how GANs work—both conceptually and in practice—through clear explanations, step-by-step code, and visualizations of training progress.  

The centerpiece of the project is the Jupyter notebook Intro_GAN.ipynb, which introduces the generator–discriminator framework, explains the adversarial training process, and demonstrates how a simple GAN learns to produce increasingly realistic synthetic images. A second notebook, GAN_More_Examples.ipynb, offers additional experiments for further exploration.

For full reproducibility, the repository also includes a standalone training script (gan_train_script.py) that implements the same GAN architecture used in the notebooks.

Background

A Generative Adversarial Network consists of two neural networks trained together:

Generator (G): Produces synthetic images from random noise.

Discriminator (D): Attempts to distinguish real images from fake ones.

During training, the generator improves by trying to fool the discriminator, while the discriminator becomes better at detecting synthetic images. This adversarial dynamic—introduced by Goodfellow et al. (2014)—allows GANs to learn complex data distributions and generate new samples that resemble real data.

This vignette focuses on clarity and intuition, using simplified model architectures and small datasets to make GAN training dynamics easy to visualize and understand.

Repository Structure
vignette-gan/
├── README.md                  # Project introduction (this file)
├── LICENSE                    # Usage and sharing permissions
│
├── notebooks/
│   ├── Intro_GAN.ipynb        # Main vignette with explanations + PyTorch code
│   └── GAN_More_Examples.ipynb# Additional GAN experiments
│
├── scripts/
│   └── gan_train_script.py    # Fully reproducible GAN training script
│
├── data/
│   └── raw/                   # MNIST/Fashion-MNIST downloads (handled automatically)
│
└── img/
    └── Intro_GAN_training_progress/   # Generated samples + loss curves

Model Architecture

The GAN in this vignette uses intentionally simple fully connected (MLP) networks to emphasize conceptual understanding rather than engineering complexity.

Generator

Input: 64-dimensional random noise vector

Hidden layers: 128 → 256

Output: 28×28 image scaled with tanh

Discriminator

Input: Flattened 28×28 image

Hidden layers: 256 → 128

Output: Real/fake probability via sigmoid

This architecture is identical in both the notebooks and the training script.

Notebooks
Intro_GAN.ipynb

The primary educational vignette.
Includes:

Conceptual explanation of GANs

Diagrams and visual training intuition

PyTorch implementation of generator and discriminator

Step-by-step adversarial training loop

Visualizations of generated samples across epochs

Discussion of training behavior and common challenges

GAN_More_Examples.ipynb

Provides optional extensions, including:

Additional training experiments

Comparisons between dataset results

Exploration of hyperparameters

Training Script

The file scripts/gan_train_script.py provides a clean, linear PyTorch implementation of the GAN training routine. It replicates the results from the notebooks and automatically saves:

Generator output samples

Loss plots

Training progress images

It can be run directly using:

python scripts/gan_train_script.py


GPU acceleration is used if available.

## References

Goodfellow, Ian, et al. 2014. Generative Adversarial Nets.
https://arxiv.org/abs/1406.2661

PyTorch DCGAN Tutorial.
https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html

Fashion-MNIST Dataset.
https://github.com/zalandoresearch/fashion-mnist

Research and Application Analysis of Generative Adversarial Network Technology.
https://www.researchgate.net/publication/392254607_Research_and_Application_Analysis_of_Generative_Adversarial_Network_Technology

## Contributors

Aidan Frazier
Anna Gornyitzki
Charles Yang
Justin Zhou
Sam Caruthers

### An educational project for PSTAT 197A demonstrating how to build, train, and interpret a basic Generative Adversarial Network (GAN) using PyTorch.
