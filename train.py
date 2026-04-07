import torch
import torch.optim as optim
from tqdm import tqdm

from models.generator import Generator, initialize_weights as init_gen
from models.critic import Critic, initialize_weights as init_critic
from utils.dataset import get_loader
from utils.save_images import save_images
import config

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

loader = get_loader(config.BATCH_SIZE)

gen = Generator(config.Z_DIM, config.CHANNELS_IMG).to(device)
gen.load_state_dict(torch.load("generator_epoch_30.pth"))
print("Loaded checkpoint from epoch 30")
critic = Critic(config.CHANNELS_IMG).to(device)

init_gen(gen)
init_critic(critic)

opt_gen = optim.Adam(gen.parameters(), lr=config.LEARNING_RATE, betas=(0.0, 0.9))
opt_critic = optim.Adam(critic.parameters(), lr=config.LEARNING_RATE, betas=(0.0, 0.9))


def gradient_penalty(critic, real, fake):
    batch_size, C, H, W = real.shape
    epsilon = torch.rand((batch_size, 1, 1, 1)).repeat(1, C, H, W).to(device)

    interpolated = real * epsilon + fake * (1 - epsilon)
    interpolated.requires_grad_(True)

    mixed_scores = critic(interpolated)

    gradient = torch.autograd.grad(
        inputs=interpolated,
        outputs=mixed_scores,
        grad_outputs=torch.ones_like(mixed_scores),
        create_graph=True,
        retain_graph=True
    )[0]

    gradient = gradient.view(gradient.shape[0], -1)
    return torch.mean((gradient.norm(2, dim=1) - 1) ** 2)


fixed_noise = torch.randn(32, config.Z_DIM, 1, 1).to(device)

start_epoch = 30
for epoch in range(start_epoch, config.NUM_EPOCHS):
    for batch_idx, (real, _) in enumerate(tqdm(loader)):
        real = real.to(device)

        # TRAIN CRITIC
        for _ in range(config.CRITIC_ITERATIONS):
            noise = torch.randn(real.shape[0], config.Z_DIM, 1, 1).to(device)
            fake = gen(noise)

            critic_real = critic(real)
            critic_fake = critic(fake.detach())

            gp = gradient_penalty(critic, real, fake)

            loss_critic = -(torch.mean(critic_real) - torch.mean(critic_fake)) + config.LAMBDA_GP * gp

            critic.zero_grad()
            loss_critic.backward()
            opt_critic.step()

        # TRAIN GENERATOR
        noise = torch.randn(real.shape[0], config.Z_DIM, 1, 1).to(device)
        fake = gen(noise)

        loss_gen = -torch.mean(critic(fake))

        gen.zero_grad()
        loss_gen.backward()
        opt_gen.step()

    print(f"Epoch {epoch} | Loss D: {loss_critic:.4f}, Loss G: {loss_gen:.4f}")

    if epoch % config.SAVE_INTERVAL == 0:
        with torch.no_grad():
            fake = gen(fixed_noise)
            save_images(fake, epoch)

        torch.save(gen.state_dict(), f"generator_epoch_{epoch}.pth")


torch.save(gen.state_dict(), "generator.pth")