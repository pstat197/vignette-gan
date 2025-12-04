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

```
vignette-gan/  
├── README.md          # Project introduction  
├── LICENSE                 
│  
├── notebooks/  
│   ├── GAN_More_Examples.ipynb      # Additional GAN experiments  
│   └── Intro_GAN.ipynb    # Main vignette with explanations + PyTorch code  
│  
├── scripts/  
│   ├── gan_train_script.py       # Fully reproducible GAN training script  
│   └── data/    
│   │   └── raw/       # MNIST/Fashion-MNIST downloads (handled automatically)  
│   └── img/           # Generated samples + loss curves  
│  
└── img/  
│   └── Intro_GAN_training_progress/     # Generated samples runs and improvements upon additional executions  
```


## References

Goodfellow, Ian, et al. 2014. Generative Adversarial Nets. https://arxiv.org/abs/1406.2661

PyTorch DCGAN Tutorial. https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html

Fashion-MNIST Dataset. https://github.com/zalandoresearch/fashion-mnist

Research and Application Analysis of Generative Adversarial Network Technology. https://www.researchgate.net/publication/392254607_Research_and_Application_Analysis_of_Generative_Adversarial_Network_Technology











