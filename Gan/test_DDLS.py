from torchvision import datasets, transforms
import torch
from prdc import compute_prdc
from tqdm.notebook import tqdm
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from model import Generator, Discriminator
from utils import load_model
from torchvision.utils import save_image

if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using device: CUDA")
else:
        device = torch.device("cpu")
        print(f"Using device: CPU")
# Model Pipeline


G = Generator().to(device)
G = load_model(G, 'checkpoints', device,'G')
D = Discriminator().to(device)
D = load_model(D, 'checkpoints', device, 'D')
if torch.cuda.device_count() > 1:
    G = torch.nn.DataParallel(model)
G.eval()

print('Model loaded.')


def energy(z,y):
    d = z.shape[1]
    
    log_p_0 = -((z**2).sum(dim=1, keepdims=True))/2-(d/2)*torch.log((torch.tensor(2*torch.pi)))
    
    d_g = D(G(z,y),y)
    return -log_p_0 - d_g


n_iterations = 20000
eps = 5e-2
z = torch.randn(64,128).to(device)
y = torch.randint(low=0, high=10, size=(64,), device=device)
for i in range(n_iterations):
    z = z.detach().requires_grad_(True)
    E = energy(z,y).sum()
    grad, = torch.autograd.grad(E,z)
    noise = torch.randn_like(z)
    with torch.no_grad():

        z += -(eps/2)*grad + eps**0.5*noise

    if i%1000==0:
        fake = G(z,y).detach()
        fake = (fake*0.5)+0.5
        save_image(fake, f"samples/DDLS_iter_{i}.png", nrow=8)
        print("=================================================================")
        print(f"iteration {i}")
        
















