
import torch
import torch.nn as nn

class Flatten(torch.nn.Module):
    def forward(self, x):
        ret = x.view(x.size(0), -1)
        return ret

class AutoEncoder(nn.Module):
    """
    Assembled with the help of this resource: https://medium.com/@syed_hasan/autoencoders-theory-pytorch-implementation-a2e72f6f7cb7
    """
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3 , 3*3, kernel_size=5),
            nn.ReLU(True),
            nn.Conv2d(9,16, kernel_size=5),
            nn.ReLU(True),
            Flatten(),
            nn.Linear(16 * 408 * 408, 32)
            )
        self.decoder = nn.Sequential( 
            nn.Linear(32, 16 * 408 * 408),
            nn.Unflatten(1, [16, 408, 408]),
            nn.ConvTranspose2d(16,9,kernel_size=5),
            nn.ReLU(True),
            nn.ConvTranspose2d(9,3,kernel_size=5),
            nn.ReLU(True))
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x