import os, sys
import time
sys.path.append("../")
from test_project import load_project, test_natural, get_validation_loader
from model_class import Net, BigNet
import argparse
import torch
import torchvision
import torch.nn.functional as F
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torchvision.utils as vutils
import re

def pgd_attack(model, images, labels, epsilon=0.3, delta=2/255, iters=30, criterion=None):
    """
    Perform PGD attack (for the inf norm) on a batch of images.
    """
    images = images.clone().detach()
    ori_images = images.data
    if criterion is None:
        criterion = torch.nn.NLLLoss()
    for i in range(iters):
        images.requires_grad = True
        outputs = model(images)

        model.zero_grad()
        cost = criterion(outputs, labels)
        cost.backward()

        adv_images = images + delta*images.grad.sign()
        clamped_attack = torch.clamp(adv_images - ori_images, min=-epsilon, max=epsilon)
        images = torch.clamp(ori_images + clamped_attack, min=0, max=1).detach_()

    return images

def test_pgd(net, test_loader, num_samples, epsilon, delta, iters, criterion, save_path='assets/', max_examples=1):
    correct = 0
    total = 0
    saved = 0

    for i, (images, labels) in enumerate(test_loader):
        images = images.to(device)
        labels = labels.to(device)

        for _ in range(num_samples):

            # PGD Attack
            perturbed = pgd_attack(net, images, labels, epsilon, delta, iters, criterion)

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

                save_dir = os.path.join(save_path, f"eps_{epsilon}_delta_{delta}_iters_{iters}")
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

    # Attacked accuracies
    epsilons = [0.01, 0.02, 0.03, 0.04, 0.05, 0.1]
    deltas = [0.01, 0.02]
    iterss = [40]
    for delta in deltas:
        for iters in iterss:
            accuracies = []
            for epsilon in epsilons:
                attacked_acc = test_pgd(net, valid_loader, num_samples=num_samples, epsilon=epsilon, delta=delta, iters=iters, criterion=criterion, save_path=save_path, max_examples=saved_examples_eps)
                print(f"Epsilon={epsilon:.2f}, delta={delta:.2f}, iters={iters} -> Attacked accuracy = {attacked_acc:.2f}%")
                accuracies.append(attacked_acc)
            plt.figure(figsize=(6,5))
            plt.plot(epsilons, accuracies, marker="*")
            plt.title(f"Attacked accuracy vs epsilon on CIFAR-10 (delta={delta}, iters={iters})")
            plt.xlabel("epsilon")
            plt.ylabel("accuracy")
            plt.grid()
            plot_path = os.path.join(save_path, f"pgd_accuracy_curve_delta_{delta}_iters_{iters}.png")
            plt.savefig(plot_path, dpi=200)
            # plt.show()
