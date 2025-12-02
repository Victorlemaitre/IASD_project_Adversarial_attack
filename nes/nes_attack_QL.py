import torch
import os, sys
import argparse
import torchvision.transforms as transforms
import torchvision
import matplotlib.pyplot as plt
import torchvision
import pandas as pd

sys.path.append("../")
from test_project import test_natural, get_validation_loader
from model_class import Net, BigNet

import re

def nes_est_grad_loss(model, x, y, sigma, n_samples, criterion, device):
    grad = torch.zeros_like(x, device=device)
    y_tensor = torch.tensor([y], device=device)

    for _ in range(n_samples):
        u = torch.randn_like(x, device=device)
        pos = x + sigma * u
        neg = x - sigma * u

        logits_pos = model(pos.unsqueeze(0))
        logits_neg = model(neg.unsqueeze(0))

        loss_pos = criterion(logits_pos, y_tensor)
        loss_neg = criterion(logits_neg, y_tensor)

        grad += (loss_pos - loss_neg) * u

    return grad / (2 * sigma * n_samples)

def nes_pgd_attack(model, images, labels, criterion, iters, sigma, n_samples, epsilon, delta, device):
    """
    NES+PGD black-box attack: PGD where the gradient is estimated with NES instead of backprop.
    """
    images = images.clone().detach().to(device)
    ori_images = images.data
    for it in range(iters):
        adv_images = images.clone().detach()
        
        grads = torch.zeros_like(adv_images, device=device)
        for i in range(adv_images.size(0)):
            x_i = adv_images[i]
            y_i = labels[i].item()
            grads[i] = nes_est_grad_loss(model=model, y=y_i, x=x_i, sigma=sigma, n_samples=n_samples, criterion=criterion, device=device)

        adv_images = adv_images + delta * grads.sign()
        clamped_attack = torch.clamp(adv_images - ori_images, min=-epsilon, max=epsilon)
        images = torch.clamp(ori_images + clamped_attack, min=0, max=1).detach_()

    return images

def test_clean(model, valid_loader, N_images, device) :
    clean_correct = 0
    count = 0
    for images, labels in valid_loader:
        for i in range(len(images)):
            if count >= N_images:
                clean_acc = clean_correct / N_images
                return clean_acc
            
            img = images[i].to(device)
            lbl = labels[i].item()

            # --- Clean prediction ---
            with torch.no_grad():
                pred_clean = model(img.unsqueeze(0)).argmax(1).item()
            if pred_clean == lbl:
                clean_correct += 1
            
            count += 1

    clean_acc = clean_correct / count
    return clean_acc

def test_nes_QL(model, valid_loader, N_images,
                 sigma, n_samples, iters,
                 epsilon, delta, device, criterion):

    adv_correct = 0
    count = 0

    for images, labels in valid_loader:
        for i in range(len(images)):

            if count >= N_images:
                adv_acc = adv_correct / N_images
                return adv_acc

            img = images[i].to(device)
            lbl = labels[i].item()

            # --- NES-PGD attack ---
            adv_img = nes_pgd_attack(
                model=model,
                images=img.unsqueeze(0),
                labels=torch.tensor([lbl], device=device),
                criterion=criterion,
                iters=iters,
                sigma=sigma,
                n_samples=n_samples,
                epsilon=epsilon,
                delta=delta,
                device=device
            )[0].detach()

            with torch.no_grad():
                pred_adv = model(adv_img.unsqueeze(0)).argmax(1).item()
            if pred_adv == lbl:
                adv_correct += 1

            count += 1

    # If loop ends early
    adv_acc = adv_correct / count
    return adv_acc


if __name__ == "__main__":

    # Parameter setups
    project_dir = "../"
    num_samples=1 # higher than 1 for randomized networks
    saved_examples_eps=4
    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    save_path = 'assets/'
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-file", default=Net.model_file,
                        help="Name of the file used to load or to sore the model weights."\
                        "If the file exists, the weights will be load from it."\
                        "If the file doesn't exists, or if --force-train is set, training will be performed, "\
                        "and the model weights will be stored in this file."\
                        "Warning: "+Net.model_file+" will be used for testing (see load_for_testing()).")
    args = parser.parse_args()
    weights_path = os.path.join(project_dir, args.model_file)
    os.makedirs(save_path, exist_ok=True)
    torch.manual_seed(42)
    criterion = torch.nn.NLLLoss()
    
    pattern = re.compile("big", re.IGNORECASE)
    if pattern.search(args.model_file):
        net = BigNet().to(device)
    else:
        net = Net().to(device)

    net.load(weights_path,device)
    net.eval()
    print(f"Loaded model weights from: {weights_path}")

    # CIFAR Loading
    transform = transforms.Compose([transforms.ToTensor()])
    cifar = torchvision.datasets.CIFAR10(project_dir + 'data/', download=True, transform=transform)
    batch_size=100
    valid_loader = get_validation_loader(cifar, batch_size=batch_size)

    # Param NES
    sigma   = 0.001
    n_samples = 50 

    epsilon = 0.1
    delta = 0.02

    # Number of images
    N_images = 200

    """L = 500
    iters = L // (2 * n_samples)
    print(f"PGD iteration : {iters}")
    epsilons = [0.01, 0.02, 0.03, 0.04, 0.05, 0.07, 0.1]

    results_sweep = []
    for epsilon in epsilons:

        adv_acc = test_nes_QL(
            model=net,
            valid_loader=valid_loader,
            N_images=N_images,
            sigma=sigma,
            n_samples=n_samples,
            iters=iters,
            epsilon=epsilon,
            delta=delta,
            device=device,
            criterion=criterion
            )
        
        print(f"Epsilon : {epsilon} | Accuracy Attacked : {adv_acc*100:>6.2f}%\n")
        results_sweep.append({
        "L": L,
        "iters": iters,
        "adv_acc": adv_acc,
        "epsilon" : epsilon,
        })
    
    df = pd.DataFrame(results_sweep)
    csv_filename = f"sweep_sigma_L{L}_ns{n_samples}_eps{epsilon}_delta{delta}_Nimages{N_images}.csv"
    df.to_csv(save_path+csv_filename, index=False)"""


    L_values = [500, 2000, 1000, 3000, 4000]
    results_sweep = []

    clean_acc = test_clean(model=net, valid_loader=valid_loader, N_images=N_images, device=device)
    print(f"Clean Accuracy : {clean_acc*100:>6.2f}‰")
    for L in L_values:
        iters = L // (2 * n_samples)
        
        print(f"Budget Limit : {L} queries")
        print(f"PGD iteration : {iters}")

        adv_acc = test_nes_QL(
            model=net,
            valid_loader=valid_loader,
            N_images=N_images,
            sigma=sigma,
            n_samples=n_samples,
            iters=iters,
            epsilon=epsilon,
            delta=delta,
            device=device,
            criterion=criterion
            )
        
        print(f"Accuracy Attacked : {adv_acc*100:>6.2f}%\n")
        
        results_sweep.append({
        "L": L,
        "iters": iters,
        "adv_acc": adv_acc
        })

    print("\n===== Sweep Summary =====")
    print(f"{'L (queries)':<15} {'iters':<10} {'NES+PGD Acc'}")
    print("-" * 60)
    for r in results_sweep:
        print(f"{r['L']:<15} {r['iters']:<10} {r['adv_acc']*100:>6.2f}%")
    print("-" * 60)

    df = pd.DataFrame(results_sweep)

    csv_filename = f"sweep_L_sigma{sigma}_ns{n_samples}_eps{epsilon}_delta{delta}_Nimages{N_images}.csv"
    df.to_csv(save_path+csv_filename, index=False)

    print("\nSaved CSV:", save_path+csv_filename)


