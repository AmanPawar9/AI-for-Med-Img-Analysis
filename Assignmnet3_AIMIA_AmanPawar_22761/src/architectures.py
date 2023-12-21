import torch
import torch.nn as nn

# Define Generator architecture
class Generator(nn.Module):
    def __init__(self, input_size=2352, output_channels=3, ngf=16):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(input_size, ngf * 4, kernel_size=3, stride=2, padding=0, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf * 4, ngf * 2, kernel_size=3, stride=2, padding=0, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf * 2, ngf, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf, output_channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh()
        )

    def forward(self, x):
        return self.main(x.view(x.size(0), x.size(1), 1, 1))
    


# Define Discriminator architecture
class Discriminator(nn.Module):
    def __init__(self, input_channels=3, ndf=16):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(input_channels, ndf, kernel_size=3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf, ndf * 2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf * 2, ndf * 4, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # Additional linear layer
        self.fc = nn.Sequential(
            nn.Linear(ndf * 4 * 28 * 28, 1),  # Adjust the input size based on your image dimensions
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.main(x)
        x = x.view(x.size(0), -1)  # Flatten the tensor before passing it to the linear layer
        x = self.fc(x)
        return x





# Define the Variational Autoencoder model
class VAE(nn.Module):
    def __init__(self, latent_dim):
        super(VAE, self).__init__()
        self.latent_dim = latent_dim

        # Encoder layers
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28 * 3, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU()
        )

        # Latent space representation
        self.fc_mu = nn.Linear(128, latent_dim)
        self.fc_logvar = nn.Linear(128, latent_dim)

        # Decoder layers
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 28 * 28 * 3),  # Adjust output size to match input size
            nn.Sigmoid()  # Sigmoid for output in the range [0, 1]
        )

    def encode(self, x):
        x = self.encoder(x)
        mu = self.fc_mu(x)
        log_var = self.fc_logvar(x)
        return mu, log_var

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        reconstructed = self.decode(z)
        return reconstructed, mu, log_var