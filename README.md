# Generative Adversarial Networks: An Introductory Vignette
Vignette on building and training a basic Generative Adversarial Network (GAN) using MNIST and Fashion-MNIST; created as a class project for PSTAT197A in Fall 2025.

## Contributors

Sam Caruthers
Aidan Frazier
Anna Gornyitzki
Charles Yang
Justin Zhou

## Vignette abstract

This vignette introduces the fundamentals of Generative Adversarial Networks (GANs) and demonstrates how to build and train a simple GAN using PyTorch (A Machine Learning Library). Using the MNIST and Fashion-MNIST datasets, the vignette explains the generator–discriminator framework of GANs, outlines the adversarial training process, and visualizes how synthetic images improve over the course of training. The primary notebook, Intro_GAN.ipynb, provides step-by-step explanations alongside executable code, while a secondary notebook, GAN_More_Examples.ipynb, extends the analysis with additional experiments. A standalone training script reproduces the results programmatically. This vignette serves as an accessible introduction for students learning how GANs operate and how they can be applied to image generation tasks.

### Background on GANs
A Generative Adversarial Network consists of two neural networks trained together:
Generator (G): Produces synthetic images from random noise.
Discriminator (D): Attempts to distinguish real images from fake ones.
During training, the generator improves by trying to fool the discriminator, while the discriminator becomes better at detecting synthetic images. This adversarial dynamic—introduced by Goodfellow et al. (2014)—allows GANs to learn complex data distributions and generate new samples that resemble real data.

### Limitations of GANs
It is important to note that since GANs were developed in 2014 and are sometimes seen as rather outdated due to its main limitation. The main limitation is mode collapse, which one should always keep in mind when implementing GANs. Mode collapse is when the generator produces only a small number of distinct outputs (or even a single output) despite many possible modes in the real data distribution. This takes place when: the generator discovers one pattern that consistently fools the discriminator, it may keep generating that pattern rather than exploring the full data distribution. Depsite this flaw GANs are a powerful method if this can be avoided

## Repository contents
vignette-gan/  
├── README.md  
├── LICENSE  
│  
├── notebooks/  
│   ├── GAN_More_Examples.ipynb  
│   └── Intro_GAN.ipynb  
│  
├── scripts/  
│   ├── gan_train_script.py  
│   └── data/  
│   └── img/  
│  
├── data/  
│   └── raw/  
│  
└── img/  
│   └── Intro_GAN_training_progress/  



## References

Goodfellow, Ian, et al. 2014. Generative Adversarial Nets. https://arxiv.org/abs/1406.2661

PyTorch DCGAN Tutorial. https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html

Fashion-MNIST Dataset. https://github.com/zalandoresearch/fashion-mnist

Research and Application Analysis of Generative Adversarial Network Technology. https://www.researchgate.net/publication/392254607_Research_and_Application_Analysis_of_Generative_Adversarial_Network_Technology











# Generative Adversarial Networks: An Introductory Vignette


## Overview

This repository provides an accessible, introductory walkthrough of Generative Adversarial Networks (GANs) using two classic image datasets: MNIST and Fashion-MNIST. The goal of the vignette is to help students intuitively understand how GANs work—both conceptually and in practice—through clear explanations, step-by-step code, and visualizations of training progress. The centerpiece of the project is the Jupyter notebook Intro_GAN.ipynb, which introduces the generator–discriminator framework, explains the adversarial training process, and demonstrates how a simple GAN learns to produce increasingly realistic synthetic images. A second notebook, GAN_More_Examples.ipynb, offers additional experiments for further exploration. For full reproducibility, the repository also includes a standalone training script (gan_train_script.py) that implements the same GAN architecture used in the notebooks.



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



vignette-gan/ ├── .gitignore # Files and folders ignored by Git ├── LICENSE # License for using and sharing this code ├── README.md # Project description and guide │ ├── notebooks/ # Educational and exploratory notebooks │ ├── Intro_GAN.ipynb # Main vignette notebook │ └── GAN_More_Examples.ipynb # Additional GAN experiments │ ├── scripts/ # Standalone scripts that replicate results │ └── gan_train_script.py # Linear, annotated script to reproduce the main notebook │ ├── data/ # Data directory │ └── raw/ # Raw datasets (MNIST and Fashion-MNIST downloaded via PyTorch) │ └── img/ # Saved figures generated by the notebooks or scripts └── Intro_GAN_training_progress/ # Generator output at different training epochs

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
