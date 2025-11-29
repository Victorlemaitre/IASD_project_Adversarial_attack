import argparse
import torch
import os, sys
import re
import torchvision.transforms as transforms
import torchvision
import matplotlib.pyplot as plt
import torchvision.utils as vutils
from tqdm import tqdm

sys.path.append("../")
from test_project import test_natural, get_validation_loader
from model_class import Net, BigNet

def nes_est_grad_loss(model, x, y, sigma, n_samples, criterion):
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

def nes_pgd_attack(model, images, labels, criterion, sigma=0.001, n_samples=5000, epsilon=0.3, delta=2/255, iters=30):
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
            grads[i] = nes_est_grad_loss(model=model, y=y_i, x=x_i, sigma=sigma, n_samples=n_samples, criterion=criterion)

        adv_images = adv_images + delta * grads.sign()
        clamped_attack = torch.clamp(adv_images - ori_images, min=-epsilon, max=epsilon)
        images = torch.clamp(ori_images + clamped_attack, min=0, max=1).detach_()

    return images


def test_nes_pgd(model, test_loader, epsilon, delta, iters, sigma, n_samples, criterion, save_path='assets/', max_examples=1):

    correct = 0
    total = 0
    saved = 0

    pbar = tqdm(test_loader, desc="NES-PGD Attack", ncols=100)

    for _, (images, labels) in enumerate(pbar):

        images = images.to(device)
        labels = labels.to(device)

        # NES-PGD Attack
        perturbed = nes_pgd_attack(model, images, labels, sigma=sigma, criterion=criterion, n_samples=n_samples, epsilon=epsilon, delta=delta, iters=iters)

        # Forward pass on adversarial images
        outputs_adv = model(perturbed)
        _, predicted = torch.max(outputs_adv.data, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        curr_acc = 100 * correct / total

        pbar.set_postfix({
            "acc": f"{curr_acc:.2f}%",
        })

        # Savings
        if saved < max_examples:
            clean_img = images[0]
            adv_img   = perturbed[0]

            with torch.no_grad():
                outputs_clean = net(images)
            clean_pred = outputs_clean.max(1)[1][0].item()
            adv_pred   = predicted[0].item()

            save_dir = os.path.join(save_path, f"eps_{epsilon}_delta_{delta}_iters_{iters}")
            save_image_pair(clean_img, adv_img, clean_pred, adv_pred, idx=saved, save_dir=save_dir)
            saved += 1

    acc = 100 * correct / total
    return acc

def get_validation_loader_v2(dataset, valid_size=1024, batch_size=32):
    indices = torch.randperm(len(dataset))[:valid_size]
    sampler = torch.utils.data.SubsetRandomSampler(indices.tolist())
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=sampler)


def save_image_pair(clean_img, adv_img, clean_pred, adv_pred, idx, save_dir):
    os.makedirs(save_dir, exist_ok=True)

    clean_path = os.path.join(save_dir, f"{idx:03d}_clean_pred_{clean_pred}.png")
    adv_path   = os.path.join(save_dir, f"{idx:03d}_adv_pred_{adv_pred}.png")

    vutils.save_image(clean_img.detach().cpu(), clean_path)
    vutils.save_image(adv_img.detach().cpu(),   adv_path)

if __name__ == "__main__":

    # Parameter setups
    project_dir = "../"
    batch_size=64
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
    
    # Model Loading
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
    valid_loader = get_validation_loader(cifar, batch_size=batch_size)

    # Natural accuracy
    acc_nat = test_natural(net, valid_loader, num_samples=num_samples)
    print(f"Clean accuracy: {acc_nat:.2f}%",)

    # Test NES estimate
    print(f"-- TEST NES ATTACK nes_pgd_attack--")
    
    # Param NES
    sigma   = 0.001
    n_samples = 1000

    # Param PGD
    epsilon = 0.01
    delta   = 0.01
    iters   = 10 # 1 FSGM
    
    print(f"Using device :{device}")
    acc_nespdg = test_nes_pgd(
        model=net,
        test_loader=valid_loader,
        epsilon=epsilon,
        delta=delta,
        iters=iters,
        sigma=sigma,
        n_samples=n_samples,
        save_path=save_path,
        criterion=criterion,
        max_examples=3
    )
    print(f"NES-PGD attacked accuracy = {acc_nespdg:.2f}%")


    """# Test NES estimate
    # --- Test NES ---
    images, labels = next(iter(valid_loader))
    images = images.to(device)
    labels = labels.to(device)

    x = images[0]
    y_true = labels[0].item()
    print(f"Label chosen : {y_true}")

    x_exact = x.clone().detach().unsqueeze(0).requires_grad_(True)
    out = net(x_exact)           
    score = out[0, y_true]       
    net.zero_grad()
    score.backward()
    grad_exact = x_exact.grad.squeeze(0).detach() 
    norm_exact = grad_exact.norm().item()
    print(f"Exact gradient norm : {norm_exact:.6f}")

    print(f"-- TEST NES estimate nes_est_grad --")
    sigma = 0.001
    n_samples_list = [10, 50, 100, 500, 1000, 2000, 5000, 10000]

    grads= []
    errors = []
    for n_samples in n_samples_list : 
        grad = nes_est_grad(net, y_true, x, sigma, n_samples)
        
        value = grad.norm().item()
        err = (grad - grad_exact).norm().item()
        
        grads.append(value)
        errors.append(err)
        
        print(f"[{n_samples}] est_norm={value:.4f}, error={err:.4f}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(n_samples_list, grads, marker='o')
    axes[0].set_xlabel("n_samples")
    axes[0].set_ylabel("Gradient norm (L2)")
    axes[0].set_title("Gradient estimate vs. number of NES samples")
    axes[0].grid(True)

    axes[1].plot(n_samples_list, errors, marker='o')
    axes[1].set_xlabel("n_samples")
    axes[1].set_ylabel("‖grad_nes - grad_exact‖ (L2)")
    axes[1].set_title("Convergence of NES estimate")
    axes[1].grid(True)

    plt.tight_layout()
    plt.savefig(save_path + "grad_estimate_subplots.png")
    plt.show()"""

    