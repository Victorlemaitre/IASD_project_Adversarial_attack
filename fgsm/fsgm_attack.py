import os, sys
sys.path.append("../")
from test_project import load_project, test_natural, get_validation_loader
import torch
import torchvision
import torch.nn.functional as F
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torchvision.utils as vutils

import torch
import torch.nn.functional as F

def fgsm_attack(model, images, labels, epsilon, criterion=None):
    """
    Generates adversarial examples using the Fast Gradient Sign Method (FGSM):

        x_adv = x + ε · sign(∇_x L(model(x), y))

    Args:
        model (nn.Module): Network to attack.
        images (Tensor): Input batch.
        labels (Tensor): Ground-truth labels.
        epsilon (float): Perturbation magnitude.
        criterion (callable): Loss function.

    Returns:
        Tensor: Adversarial images clipped to [0, 1].
    """

    if criterion is None:
        criterion = torch.nn.NLLLoss()

    # Cloning
    images = images.clone().detach().requires_grad_(True)

    # Forward
    outputs = model(images)
    loss = criterion(outputs, labels)

    # Backward 
    model.zero_grad()
    loss.backward()

    # Perturbation
    grad_sign = images.grad.detach().sign()
    adv_images = images + epsilon * grad_sign

    # Clamping
    adv_images = torch.clamp(adv_images, 0, 1).detach()

    return adv_images

def test_fgsm(net, test_loader, num_samples, epsilon, criterion, save_path='assets/', max_examples=1):

    correct = 0
    total = 0
    saved = 0

    for i, (images, labels) in enumerate(test_loader):
        images = images.to(device)
        labels = labels.to(device)

        total = 0
        correct = 0

        for _ in range(num_samples):

            # FGSM Attack
            perturbed = fgsm_attack(net, images, labels, epsilon, criterion)

            # Forward on perturbation
            outputs_adv = net(perturbed)
            _, predicted = torch.max(outputs_adv.data, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # Savings
            if saved < max_examples:
                clean_img = images[0]
                adv_img   = perturbed[0]

                with torch.no_grad():
                    outputs_clean = net(images)
                clean_pred = outputs_clean.max(1)[1][0].item()
                adv_pred   = predicted[0].item()

                save_dir = os.path.join(save_path, f"eps_{epsilon}")
                save_image_pair(clean_img, adv_img, clean_pred, adv_pred, idx=saved, save_dir=save_dir)
                saved += 1

    return 100 * correct / total


def save_image_pair(clean_img, adv_img, clean_pred, adv_pred, idx, save_dir):
    os.makedirs(save_dir, exist_ok=True)

    clean_path = os.path.join(save_dir, f"{idx:03d}_clean_pred_{clean_pred}.png")
    adv_path   = os.path.join(save_dir, f"{idx:03d}_adv_pred_{adv_pred}.png")

    vutils.save_image(clean_img.detach().cpu(), clean_path)
    vutils.save_image(adv_img.detach().cpu(),   adv_path)


if __name__ == "__main__":

    # Parameter setups
    pretrained_model = "../data/lenet_mnist_model.pth"
    project_dir = "../"
    batch_size=64
    num_samples=100
    saved_examples_eps=4
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    save_path = 'assets/'

    os.makedirs(save_path, exist_ok=True)
    torch.manual_seed(42)
    criterion = torch.nn.NLLLoss()
    
    # Model Loading
    project_module = load_project(project_dir)
    net = project_module.Net().to(device)
    net.load_for_testing(project_dir=project_dir)

    # CIFAR Loading
    transform = transforms.Compose([transforms.ToTensor()])
    cifar = torchvision.datasets.CIFAR10(project_dir + 'data/', download=True, transform=transform)
    valid_loader = get_validation_loader(cifar, batch_size=batch_size)

    # Natural accuracy
    acc_nat = test_natural(net, valid_loader, num_samples=num_samples)
    print(f"Clean accuracy: {acc_nat:.2f}%",)

    # Attacked accuracies
    epsilons = [0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.1]
    accuracies = []
    for eps in epsilons:
        attacked_acc = test_fgsm(net, valid_loader, num_samples=num_samples, epsilon=eps, criterion=criterion, save_path=save_path, max_examples=saved_examples_eps)
        print(f"Epsilon={eps:.2f} -> Attacked accuracy = {attacked_acc:.2f}%")
        accuracies.append(attacked_acc)

    # Plots
    plt.figure(figsize=(6,5))
    plt.plot(epsilons, accuracies, marker="*")
    plt.title("Attacked accuracy vs epsilon on CIFAR-10")
    plt.xlabel("epsilon")
    plt.ylabel("accuracy")
    plt.grid()
    plot_path = os.path.join(save_path, "fgsm_accuracy_curve.png")
    plt.savefig(plot_path, dpi=200)
    plt.show()
