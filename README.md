WGAN CIFAR-10 Explorer

This project implements a Wasserstein Generative Adversarial Network (WGAN) trained on the CIFAR-10 dataset, along with an interactive web application for exploring generated images and training behavior.

The system allows users to generate synthetic images, visualize how outputs improve over training epochs, and explore latent space interpolation using an intuitive interface built with Gradio.

Features
Generate CIFAR-10 style images using a trained WGAN model
Visualize training progression across epochs
Explore latent space interpolation between generated samples
Control generation using adjustable parameters such as image count and seed
Interface
Generate Images

<img width="1571" height="868" alt="generate" src="https://github.com/user-attachments/assets/24c153cf-87bd-44d8-9d0b-38b4739940be" />



Latent Space Interpolation

<img width="1499" height="806" alt="interpolate" src="https://github.com/user-attachments/assets/51d86076-f8ed-4aa6-8fd4-d3eacb1cd2a9" />


Epoch Progress Visualization

<img width="1766" height="871" alt="epoch" src="https://github.com/user-attachments/assets/5c8e958b-0b00-4a48-a1c3-8a90b62668d2" />


Model Details
Architecture: Deep Convolutional Wasserstein GAN (WGAN)
Dataset: CIFAR-10
Training Duration: 50 epochs
Framework: PyTorch
How It Works

The model is trained using the CIFAR-10 dataset. During training, images are periodically saved at different epochs to track the improvement in generation quality.

The web interface loads these saved outputs and allows users to:

View image quality progression using an epoch slider
Generate new samples from the trained generator
Interpolate between latent vectors to observe smooth transitions
Project Structure
WGAN-CIFAR10/
│── models/           # Generator and Critic architectures
│── utils/            # Dataset loading and image utilities
│── outputs/          # Saved images across epochs (local use)
│── assets/           # UI screenshots for README
│── app.py            # Gradio interface
│── train.py          # Training script
│── config.py         # Configuration parameters
│── requirements.txt
Installation and Usage

Clone the repository:

git clone https://github.com/Jayderino/WGAN-CIFAR10-Explorer.git
cd WGAN-CIFAR10-Explorer

Install dependencies:

pip install -r requirements.txt

Run the application:

python app.py
Notes
The CIFAR-10 dataset is not included due to size constraints and is downloaded automatically during training
Model weights and full training outputs are not included in the repository
Sample outputs are used to demonstrate training progression in the interface
Author

Jaden Castelino
