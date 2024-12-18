import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from residual import ResidualUpBlock, ResidualDownBlock


class Generator(nn.Module):
    def __init__(self, img_size, latent_dim, dim, residual=False, gumbel_latent=False):
        super(Generator, self).__init__()
        self.dim = dim
        self.latent_dim = latent_dim
        self.img_size = img_size
        self.feature_sizes = tuple(map(int, (self.img_size[0] / 16, self.img_size[1] / 16)))
        self.gumbel_latent = gumbel_latent

        self.latent_to_features = nn.Sequential(
            nn.Linear(latent_dim, 8 * dim * self.feature_sizes[0] * self.feature_sizes[1]),
            nn.ReLU()
        )

        if residual:
            self.features_to_image = nn.Sequential(
                ResidualUpBlock(8 * dim, 4 * dim, (4, 4), 2, 1),
                ResidualUpBlock(4 * dim, 2 * dim, (4, 4), 2, 1),
                ResidualUpBlock(2 * dim, dim, (4, 4), 2, 1),
                nn.ConvTranspose2d(dim, self.img_size[2], 4, 2, 1),
                nn.Sigmoid() # makes MNIST better, less sure for ERA5
            )
        else:
            self.features_to_image = nn.Sequential(
                nn.ConvTranspose2d(8 * dim, 4 * dim, 4, 2, 1),
                nn.ReLU(),
                nn.BatchNorm2d(4 * dim),
                nn.ConvTranspose2d(4 * dim, 2 * dim, 4, 2, 1),
                nn.ReLU(),
                nn.BatchNorm2d(2 * dim),
                nn.ConvTranspose2d(2 * dim, dim, 4, 2, 1),
                nn.ReLU(),
                nn.BatchNorm2d(dim),
                nn.ConvTranspose2d(dim, self.img_size[2], 4, 2, 1),
                nn.Sigmoid()
            )


    def forward(self, input_data):
        x = self.latent_to_features(input_data)
        x = x.view(-1, 8 * self.dim, self.feature_sizes[0], self.feature_sizes[1])
        return self.features_to_image(x)

    def sample_latent(self, num_samples):
        if self.gumbel_latent:
            latent = torch.rand((num_samples, self.latent_dim))
            return -torch.log(-torch.log(latent))
        else:
            return torch.randn((num_samples, self.latent_dim))


class Discriminator(nn.Module):
    def __init__(self, img_size, dim, residual=False):
        """
        img_size : (int, int, int)
            Height and width must be powers of 2.  E.g. (32, 32, 1) or
            (64, 128, 3). Last number indicates number of channels, e.g. 1 for
            grayscale or 3 for RGB
        """
        super(Discriminator, self).__init__()

        self.img_size = img_size

        if residual:
            self.image_to_features = nn.Sequential(
                ResidualDownBlock(self.img_size[2], dim, (4,4), 2, 1),
                ResidualDownBlock(dim, 2 * dim, (4,4), 2, 1),
                ResidualDownBlock(2 * dim, 4 * dim, (4,4), 2, 1),
                nn.Conv2d(4 * dim, 8 * dim, 4, 2, 1),
                nn.Sigmoid()
            )
        else:
            self.image_to_features = nn.Sequential(
                nn.Conv2d(self.img_size[2], dim, 4, 2, 1),
                nn.LeakyReLU(0.2),
                nn.Conv2d(dim, 2 * dim, 4, 2, 1),
                nn.LeakyReLU(0.2),
                nn.Conv2d(2 * dim, 4 * dim, 4, 2, 1),
                nn.LeakyReLU(0.2),
                nn.Conv2d(4 * dim, 8 * dim, 4, 2, 1),
                nn.Sigmoid()
            )


        # 4 convolutions of stride 2, i.e. halving of size everytime
        # So output size will be 8 * (img_size / 2 ^ 4) * (img_size / 2 ^ 4)
        self.output_size = int(8 * dim * (img_size[0] / 16) * (img_size[1] / 16))
        self.features_to_prob = nn.Sequential(
            nn.Linear(self.output_size, 1),
            nn.Sigmoid() # apparently this shouldn't be used for WGAN-GP? Try removing?
        )
    
    def forward(self, input_data):
        batch_size = input_data.size()[0]
        x = self.image_to_features(input_data)
        x = x.view(batch_size, -1)
        return self.features_to_prob(x)
    

class SanDiscriminator(Discriminator):
    """Don't think this will work with the gradient penalty loss."""
    def __init__(self, img_size, dim, residual=False):
        super(Discriminator, self).__init__(img_size, dim, residual)
        self.W = nn.Parameter(torch.randn(1, self.output_size))
        self.features_to_prob = nn.Sigmoid()

    def forward(self, input_data, training:bool):
        hidden_feature = self.image_to_features(input_data)
        hidden_feature = torch.flatten(hidden_feature, start_dim=1) # same as x = x.view(batch_size, -1)?
        weights = self.W
        direction = F.normalize(weights, dim=1)
        scale = torch.norm(weights, dim=1).unsqueeze(1) 
        hidden_feature = hidden_feature * scale

        if training:
            out_fun = (hidden_feature * direction.detach()).sum(dim=1)
            out_dir = (hidden_feature.detach() * direction).sum(dim=1)
            x = dict(fun=out_fun, dir=out_dir)
        else:
            x = (hidden_feature * direction).sum(dim=1)
        
        return self.features_to_prob(x)

    