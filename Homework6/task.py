import torch
from torch import nn
from torch.nn import functional as F


# Task 1

class Encoder(nn.Module):
    def __init__(self, img_size=128, latent_size=512, start_channels=16,
                 downsamplings=5, linear_hidden_size = 128):
        super().__init__()
        self.ModuleList = nn.ModuleList([nn.Conv2d(3, start_channels,kernel_size=1,stride=1,padding=0)])

        for i in range(downsamplings):
            self.ModuleList.extend(nn.ModuleList([nn.Conv2d((2**i)*start_channels,
                    (2**(i+1)) * start_channels, kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm2d((2**(i+1)) * start_channels), 
                    nn.ReLU()]))

        last_layers = nn.ModuleList([nn.Flatten(),
                        nn.Linear(int(start_channels*img_size*img_size/(2**downsamplings)),linear_hidden_size),
                        nn.ReLU(),
                        nn.Linear(linear_hidden_size,2*latent_size)])

        self.ModuleList.extend(last_layers)
        #self.cuda()

    def forward(self, x):
        # x = x.cuda()
        for module in self.ModuleList:
            x = module(x)

        mu = x[:, :int(x.size()[1]/2)]
        sigma = x[:, int(x.size()[1]/2):]
        sigma = torch.exp(sigma)

        #embedding = mu + torch.randn((mu.size()[0], mu.size()[1])).mul(sigma)
        embedding = mu + torch.randn((mu.size()[0], mu.size()[1]), device=x.device).mul(sigma)
        return embedding, (mu, sigma)


# Task 2

class Decoder(nn.Module):
    def __init__(self, img_size=128, latent_size=512,
                 end_channels=16, upsamplings=5, linear_hidden_size = 2*128):
        super().__init__()
        self.ModuleList = nn.ModuleList([
            nn.Linear(latent_size, linear_hidden_size),
            nn.ReLU(),
            nn.Linear(linear_hidden_size, int(end_channels*(img_size**2)/(2**upsamplings))),
            nn.Unflatten(1, (end_channels*(2**upsamplings), int(img_size/(2**upsamplings)),
                             int(img_size/(2**upsamplings))))])

        for i in range(upsamplings, 0, -1):
            self.ModuleList.extend(nn.ModuleList([
                nn.ConvTranspose2d(end_channels*(2**i), end_channels*(2**(i-1)),
                                   kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(end_channels*(2**(i-1))),
                nn.ReLU()]))

        last_layers = nn.ModuleList([nn.ConvTranspose2d(end_channels, 3, kernel_size=1, stride=1),
                                     nn.Tanh()])

        self.ModuleList.extend(last_layers)
        #self.cuda()

    def forward(self, z):
        #z = z.cuda()
        for module in self.ModuleList:
            z = module(z)
        return z


# Task 3

class VAE(nn.Module):
    def __init__(self, img_size=128, downsamplings=3, latent_size=512, down_channels=3, up_channels=6):
        super().__init__()
        self.encoder = Encoder(img_size=img_size, latent_size=latent_size,
                               start_channels=down_channels,downsamplings=downsamplings)
        self.decoder = Decoder(img_size=img_size, latent_size=latent_size,
                               end_channels=up_channels, upsamplings=downsamplings)
        #self.cuda()

    def forward(self, x):
        z, (mu, sigma) = self.encoder.forward(x)
        x_pred = self.decoder.forward(z)
        #kld = 0.5*torch.sum(sigma**2 + mu**2 - torch.log(sigma**2) - 1)
        kld = 0.5 * (sigma**2 + mu**2 - torch.log(sigma ** 2) - 1)
        return x_pred, kld

    def encode(self, x):
        z, (mu, sigma) = self.encoder.forward(x)
        return z

    def decode(self, z):
        x_pred = self.decoder.forward(z)
        return x_pred

    def save(self):
        torch.save(self.state_dict(), "Model.pth")

    def load(self):
        #self.load_state_dict(torch.load(__file__[:-7] + "Model.pth",
                                        #map_location=lambda storage, loc: storage))
        self.load_state_dict(torch.load(__file__[:-7] + "Model.pth"))



