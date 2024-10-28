# %%
import os
import numpy as np
import imageio
import torch
import torch.optim as optim
from torchvision.utils import make_grid
import dataloaders
from models import Generator, Discriminator
from training import Trainer
import matplotlib.pyplot as plt

# %%
def notify(title, subtitle, message):
    os.system("""
              osascript -e 'display notification "{}" with title "{}" subtitle "{}" beep'
              """.format(message, title, subtitle))

if not torch.backends.mps.is_available():
    if not torch.backends.mps.is_built():
        print("MPS not available because the current PyTorch install was not "
              "built with MPS enabled.")
    else:
        print("MPS not available because the current MacOS version is not 12.3+ "
              "and/or you do not have an MPS-enabled device on this machine.")

GUMBEL   = False
RESIDUAL = True
DATASET  = 'era5'
CHANNELS = 1
DIM      = 16 # default 16
img_size = (32, 32, CHANNELS)

# automate naming of files
gstr = "gumbel" if GUMBEL else "gaussian"
rstr = "residual" if RESIDUAL else ""

get_dataloaders = getattr(dataloaders, f"get_{DATASET}_dataloaders")
data_loader, _ = get_dataloaders(batch_size=64,
                                 img_size=(img_size[0], img_size[1]),
                                 gumbel=GUMBEL,
                                 nvars=CHANNELS) # need to add to mps


generator = Generator(img_size=img_size, latent_dim=100, dim=DIM, residual=RESIDUAL, gumbel_latent=GUMBEL)
discriminator = Discriminator(img_size=img_size, dim=DIM, residual=RESIDUAL)

print(generator)
print(discriminator)

# Initialize optimizers
lr = 1e-4
betas = (.9, .99)
G_optimizer = optim.Adam(generator.parameters(), lr=lr, betas=betas)
D_optimizer = optim.Adam(discriminator.parameters(), lr=lr, betas=betas)

# %% temporary: make training image grid
train_sample = next(iter(data_loader))[0]
img_grid = make_grid(train_sample)
img_grid = np.transpose(img_grid.numpy(), (1, 2, 0))
im = (img_grid * 255).astype(np.uint8)
imageio.imwrite(f'gifs/{DATASET}_train.png', im)

# %% Train model
epochs = 100
trainer = Trainer(generator, discriminator, G_optimizer, D_optimizer,
                  device="mps", label=f'{DATASET}_{rstr}_{gstr}')
trainer.train(data_loader, epochs, save_training_gif=True)

# Save models
name = f'{DATASET}_{rstr}_{gstr}'
torch.save(trainer.G.state_dict(), 'models/gen_' + name + '.pt')
torch.save(trainer.D.state_dict(), 'models/dis_' + name + '.pt')
notify("Process finished", "Python script", "Finished training GAN!")

# %% check out loss curves
G_loss = torch.Tensor(trainer.losses['G']).to('cpu')
D_loss = torch.Tensor(trainer.losses['D']).to('cpu')
GP = torch.Tensor(trainer.losses['GP']).to('cpu')
gradient_norm = torch.Tensor(trainer.losses['gradient_norm']).to('cpu')

critic_iters = trainer.critic_iterations
N = len(D_loss)

# plt.plot(np.arange(0, N, critic_iters), G_loss, label='Generator loss')
plt.plot(np.linspace(0, N, len(G_loss)), G_loss, label='Generator loss')
plt.plot(np.arange(0, N), D_loss, label='Discriminator loss')
plt.legend()
plt.savefig('imgs/latest_loss_curve.png', dpi=300)

# %% ---END---
x = np.load(f'arrs/{DATASET}_{rstr}_{gstr}_{epochs}_epochs.npz')['data']
plt.imshow(x[0, ..., 0])
plt.colorbar()
# %%
