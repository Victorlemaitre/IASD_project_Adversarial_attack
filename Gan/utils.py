import torch
import torch.nn.functional as F
import os



def D_train(x, y_label, G, D, D_optimizer, device):
    #=======================Train the discriminator=======================#
    D.zero_grad()

    # train discriminator on real
    x_real = x.to(device)
    y_label = y_label.to(device)
    # train discriminator on fake
    z = torch.randn(x.shape[0], 128,device=device)
    x_fake = G(z,y_label)

    real_logits = D(x_real,y_label)    # (N,)
    fake_logits = D(x_fake, y_label)

    d_loss_real = F.relu(1.0 - real_logits).mean()
    d_loss_fake = F.relu(1.0 + fake_logits).mean()
    D_loss = d_loss_real + d_loss_fake    

    # gradient backprop & optimize ONLY D's parameters
    D_loss.backward()
    D_optimizer.step()
        
    return  D_loss.data.item()


def G_train(x, G, D, G_optimizer, device):
    #=======================Train the generator=======================#
    G.zero_grad()

    z = torch.randn(x.shape[0], 128,device=device)
    y = torch.ones(x.shape[0], 1, device=device)
    random_label = torch.randint(low=0,high=10, size = (x.shape[0],), device = device)
    G_output = G(z,random_label)
    D_output = D(G_output, random_label)
    G_loss = -D_output.mean()

    # gradient backprop & optimize ONLY G's parameters
    G_loss.backward()
    G_optimizer.step()
        
    return G_loss.data.item()



def save_models(G, D, folder):
    torch.save(G.state_dict(), os.path.join(folder,'G.pth'))
    torch.save(D.state_dict(), os.path.join(folder,'D.pth'))


def load_model(G, folder, device, s):
    if s == 'G':
        ckpt_path = os.path.join(folder,'G.pth')
        ckpt = torch.load(ckpt_path, map_location=device)
        G.load_state_dict({k.replace('module.', ''): v for k, v in ckpt.items()})
        return G
    else:
        ckpt_path = os.path.join(folder,'D.pth')
        ckpt = torch.load(ckpt_path, map_location=device)
        G.load_state_dict({k.replace('module.', ''): v for k, v in ckpt.items()})
        return G
        
