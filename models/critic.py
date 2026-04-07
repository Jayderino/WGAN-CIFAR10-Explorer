import torch.nn as nn

class Critic(nn.Module):
    def __init__(self, channels_img):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(channels_img, 64, 4, 2, 1),   # 32 → 16
            nn.LeakyReLU(0.2),

            nn.Conv2d(64, 128, 4, 2, 1),            # 16 → 8
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2),

            nn.Conv2d(128, 256, 4, 2, 1),           # 8 → 4
            nn.InstanceNorm2d(256),
            nn.LeakyReLU(0.2),

            nn.Conv2d(256, 512, 4, 2, 1),           # 4 → 2
            nn.InstanceNorm2d(512),
            nn.LeakyReLU(0.2),

            nn.Conv2d(512, 1, 2, 1, 0)              # 2 → 1
        )

    def forward(self, x):
        return self.net(x).reshape(-1)


def initialize_weights(model):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.normal_(m.weight.data, 0.0, 0.02)