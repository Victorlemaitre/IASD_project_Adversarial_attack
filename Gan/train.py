import torch
import os
from tqdm import trange
import argparse
from torchvision import datasets, transforms
from torchvision.utils import save_image
from prdc import compute_prdc
import torch.nn as nn
import torch.optim as optim
from model import Generator, Discriminator,weights_init
from utils import D_train, G_train, save_models

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train GAN on MNIST.')
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs for training.")
    parser.add_argument("--lr", type=float, default=0.0002, help="Learning rate.")
    parser.add_argument("--batch_size", type=int, default=64, help="Size of mini-batches for SGD.")
    parser.add_argument("--gpus", type=int, default=-1, help="Number of GPUs to use (-1 for all available).")
    args = parser.parse_args()

    to_download=False
    if torch.cuda.is_available():
        device = torch.device("cuda")
        device_type = "cuda"
        print(f"Using device: CUDA")
        # Use all available GPUs if args.gpus is -1
        if args.gpus == -1:
            args.gpus = torch.cuda.device_count()
            print(f"Using {args.gpus} GPUs.")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        device_type = "mps"
        print(f"Using device: MPS (Apple Metal)")
    else:
        device = torch.device("cpu")
        device_type = "cpu"
        print(f"Using device: CPU")
        

    

    # Create directories
    os.makedirs('checkpoints', exist_ok=True)
    data_path = './data'
    if data_path is None:
        data_path = "data"
        to_download = True
    # Data Pipeline
    print('Dataset loading...')
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    train_dataset = datasets.CIFAR10(root=data_path, train=True, transform=transform, download=to_download)
    test_dataset = datasets.CIFAR10(root=data_path, train=False, transform=transform, download=to_download)

    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,  # Use multiple workers for data loading
        pin_memory=True  # Faster data transfer to GPU
    )
    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    print('Dataset loaded.')

    # Model setup
    print('Model loading...')
    
    G = Generator().to(device)
    D = Discriminator().to(device)
    #G.apply(weights_init)
    #D.apply(weights_init)
    # Wrap models in DataParallel if multiple GPUs are available
    if args.gpus > 1:
        G = torch.nn.DataParallel(G)
        D = torch.nn.DataParallel(D)
    print('Model loaded.')
    G_ema = Generator().to(device)
    G_ema.load_state_dict(G.state_dict()) # Initialize with G's weights
    G_ema.eval() # EMA model is only for inference

    # Helper function
    def update_ema(ema_model, model, decay=0.999):
        with torch.no_grad():
            for ema_param, param in zip(ema_model.parameters(), model.parameters()):
                ema_param.data.mul_(decay).add_(param.data, alpha=1 - decay)
            for ema_buffer, buffer in zip(ema_model.buffers(), model.buffers()):
                ema_buffer.copy_(buffer)
    # Loss and optimizers
    
    betas = (0.0, 0.9)
    G_optimizer = optim.Adam(G.parameters(), lr=1e-4,betas=betas)
    D_optimizer = optim.Adam(D.parameters(), lr=2e-4,betas=betas)
    # create the real features from the test_dataset. I had to use the loader in order to keep the transforms 
    os.makedirs("samples", exist_ok=True) 
    print('Start training:')
    n_epoch = args.epochs
    for epoch in range(0, n_epoch + 1):
        d_loss_list, g_loss_list = [],[]
        for batch_idx, (x, y) in enumerate(train_loader):
            x = x.to(device)
            d_loss_list.append( D_train(x,y, G, D, D_optimizer, device))
            if batch_idx%2==0 or epoch<5:
                g_loss_list.append(G_train(x, G, D, G_optimizer,  device))
                update_ema(G_ema, G)
        if epoch % 10 == 0:
            print(f"discr loss : {sum(d_loss_list)/len(d_loss_list):.2f}")
            print(f"gen loss : {sum(g_loss_list)/len(g_loss_list):.2f}")
            save_models(G_ema, D, 'checkpoints')
            
            z = torch.randn(64,128).to(device)
            y = torch.randint(low=0, high=10, size=(64,), device=device)
            fake_ema = G_ema(z,y).detach()
            fake_ema = (fake_ema*0.5)+0.5
            fake = G(z,y).detach()
            fake = (fake*0.5)+0.5
            save_image(fake, f"samples/epoch_{epoch}.png", nrow=8)
            save_image(fake_ema, f"samples/EMA_epoch_{epoch}.png", nrow=8)


    print('Training done.')
    
