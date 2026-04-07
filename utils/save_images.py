import torchvision.utils as vutils
import os

def save_images(fake, epoch):
    os.makedirs("outputs", exist_ok=True)
    vutils.save_image(fake, f"outputs/epoch_{epoch}.png", normalize=True)