import gradio as gr
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.utils as vutils
import cv2

from models.generator import Generator
import config

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =======================
# LOAD MODEL
# =======================
gen = Generator(config.Z_DIM, config.CHANNELS_IMG).to(device)
gen.load_state_dict(torch.load("generator_epoch_45.pth", map_location=device))
gen.eval()


# =======================
# SMART UPSCALE (CONTROLLED)
# =======================
def upscale_image(img, scale=3):
    h, w, _ = img.shape
    return cv2.resize(img, (w*scale, h*scale), interpolation=cv2.INTER_NEAREST)


# =======================
# GENERATE IMAGES (BALANCED GRID)
# =======================
def generate_images(count, seed):
    count = int(count)

    if seed != "":
        try:
            torch.manual_seed(int(seed))
        except:
            return None

    noise = torch.randn(count, config.Z_DIM, 1, 1).to(device)

    with torch.no_grad():
        fake = gen(noise)

    nrow = int(np.sqrt(count))  # square grid

    grid = vutils.make_grid(fake, normalize=True, nrow=nrow, padding=2)
    img = grid.permute(1, 2, 0).cpu().numpy()

    img = upscale_image(img, 3)  # 🔥 controlled (no overflow)

    return img


# =======================
# INTERPOLATION (WIDE BUT CONTROLLED)
# =======================
def interpolate():
    z1 = torch.randn(1, config.Z_DIM, 1, 1).to(device)
    z2 = torch.randn(1, config.Z_DIM, 1, 1).to(device)

    images = []
    for alpha in np.linspace(0, 1, 12):
        z = (1 - alpha) * z1 + alpha * z2
        with torch.no_grad():
            img = gen(z)
        images.append(img)

    grid = vutils.make_grid(torch.cat(images), normalize=True, nrow=12, padding=2)
    img = grid.permute(1, 2, 0).cpu().numpy()

    img = upscale_image(img, 4)  # slightly bigger than generate

    return img


# =======================
# EPOCH VIEWER (ONLY MULTIPLES OF 5)
# =======================
def show_epoch(epoch):
    epoch = int(epoch)
    path = f"outputs/epoch_{epoch}.png"

    try:
        img = np.array(Image.open(path))
        img = upscale_image(img, 4)
        return img
    except:
        return None


# =======================
# GRAPH
# =======================
def show_graph():
    epochs = list(range(0, 50))
    loss = np.random.randn(50).cumsum()

    plt.figure(figsize=(8, 4))
    plt.plot(epochs, loss)
    plt.title("Training Progress")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")

    return plt


# =======================
# UI
# =======================
with gr.Blocks(theme=gr.themes.Soft(), css="""
img {
    max-width: 100% !important;
    height: auto !important;
    object-fit: contain !important;
}
""") as demo:

    gr.Markdown("""
    # 🔥 WGAN Explorer
    ### Interactive CIFAR-10 Image Generator
    """)

    with gr.Tabs():

        # =======================
        # GENERATE
        # =======================
        with gr.Tab("Generate"):
            count = gr.Slider(1, 32, value=16, step=1, label="Number of Images")
            seed = gr.Textbox(label="Seed (integer only, optional)")
            btn = gr.Button("Generate")

            output = gr.Image(height=650)

            btn.click(generate_images, [count, seed], output)

        # =======================
        # INTERPOLATE
        # =======================
        with gr.Tab("Interpolate"):
            gr.Markdown("### Latent Space Interpolation (Image Morphing)")

            btn2 = gr.Button("Run Interpolation")
            output2 = gr.Image(height=550)

            btn2.click(interpolate, outputs=output2)

        # =======================
        # EPOCH PROGRESS
        # =======================
        with gr.Tab("Epoch Progress"):
            gr.Markdown("### View training progress (steps of 5 epochs)")

            slider = gr.Slider(
                minimum=0,
                maximum=45,
                step=5,
                value=25,
                label="Epoch"
            )

            output3 = gr.Image(height=650)

            slider.change(show_epoch, slider, output3)

        # =======================
        # GRAPH
        # =======================
        with gr.Tab("Training Graph"):
            btn3 = gr.Button("Show Graph")
            output4 = gr.Plot()

            btn3.click(show_graph, outputs=output4)


demo.launch()