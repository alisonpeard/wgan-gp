# %%
import torch
import torch.optim as optim
import dataloaders
from models import Generator, Discriminator
from training import Trainer

# %%
if not torch.backends.mps.is_available():
    if not torch.backends.mps.is_built():
        print("MPS not available because the current PyTorch install was not "
              "built with MPS enabled.")
    else:
        print("MPS not available because the current MacOS version is not 12.3+ "
              "and/or you do not have an MPS-enabled device on this machine.")
GUMBEL = False
RESIDUAL = True
DATASET = 'mnist'

# automate naming of files
gstr = "gumbel" if GUMBEL else "gaussian"
rstr = "residual" if RESIDUAL else ""

get_dataloaders = getattr(dataloaders, f"get_{DATASET}_dataloaders")
data_loader, _ = get_dataloaders(batch_size=64, gumbel=GUMBEL) # need to add to mps
img_size = (32, 32, 1)

generator = Generator(img_size=img_size, latent_dim=100, dim=16, residual=RESIDUAL, gumbel_latent=GUMBEL)
discriminator = Discriminator(img_size=img_size, dim=16, residual=RESIDUAL)

print(generator)
print(discriminator)

# Initialize optimizers
lr = 1e-4
betas = (.9, .99)
G_optimizer = optim.Adam(generator.parameters(), lr=lr, betas=betas)
D_optimizer = optim.Adam(discriminator.parameters(), lr=lr, betas=betas)

# Train model
epochs = 50
trainer = Trainer(generator, discriminator, G_optimizer, D_optimizer,
                  device="mps", label=f'{DATASET}_{rstr}_{gstr}', num_steps=3)

trainer.train(data_loader, epochs, save_training_gif=True)

# Save models
name = f'{DATASET}_{rstr}_{gstr}'
torch.save(trainer.G.state_dict(), './gen_' + name + '.pt')
torch.save(trainer.D.state_dict(), './dis_' + name + '.pt')

# %% ---END---
