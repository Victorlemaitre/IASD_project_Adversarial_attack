#!/usr/bin/env python3 
import os
import re
import argparse
import torch
import torchvision
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
import torchvision.transforms as transforms

from pgd.pgd_attack import pgd_attack
import importlib
from model_class import Net


device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
print(device)
valid_size = 1024 
batch_size = 64 

def adversarial_training(net, train_loader, valid_loader, pth_filename, num_epochs, num_samples, epsilon, delta, iters, adv_to_nat_ratio):
    '''Basic training function (from pytorch doc.)'''
    print("Starting training")
    
    criterion = nn.NLLLoss()
    optimizer = optim.AdamW(net.parameters(), lr=1e-3, weight_decay=0.01)
    
    for epoch in range(num_epochs):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data[0].to(device), data[1].to(device)

            batch_size = inputs.size(0)
            t = epoch/num_epochs
            iters_t = int((1-t)*10 + t*40)
            epsilon_t = (1-t)*0.01 + t*0.1
            rand_idx = torch.randperm(batch_size, device=inputs.device)[:int(batch_size*adv_to_nat_ratio)]
            inputs[rand_idx] = pgd_attack(
                    net, 
                    inputs[rand_idx], 
                    labels[rand_idx], 
                    epsilon_t, 
                    delta, 
                    iters_t, 
                    criterion
                    )
            

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 500 == 499:    # print every 500 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 500))
                running_loss = 0.0
        if epoch%10 == 0:
            # in-training test with the same parameters for PGD as those used for adversarial training 
            nat_acc, adv_acc = test_natural_and_pgd(net, valid_loader, num_samples, epsilon, delta, iters, criterion)
            print(f"Natural accuracy  = {nat_acc:.2f}%")
            print(f"Attacked accuracy = {adv_acc:.2f}%")
            net.save(pth_filename[:-4]+f"_checkpoint{epoch}.pth")
        

    net.save(pth_filename)
    print('Model saved in {}'.format(pth_filename))

def test_natural_and_pgd(net, test_loader, num_samples, epsilon, delta, iters, criterion):
    
    #net.eval()
    net.train()

    total_nat, total_adv, correct_nat, correct_adv = 0, 0, 0, 0
    for i, (images, labels) in enumerate(test_loader):
        images = images.to(device)
        labels = labels.to(device)
        # Natural test
        with torch.no_grad():
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            correct_nat += (predicted == labels).sum().item()
            total_nat += labels.size(0)


        # PGD Attack
        for _ in range(num_samples):

            perturbed = pgd_attack(net, images, labels, epsilon, delta, iters, criterion)

            # Forward on perturbation
            outputs_adv = net(perturbed)
            _, predicted = torch.max(outputs_adv.data, 1)
            total_adv += labels.size(0)

            
            correct_adv += (predicted == labels).sum().item()


    return 100*correct_nat/total_nat, 100*correct_adv/total_adv

def test_natural(net, test_loader):
    '''Basic testing function.'''

    correct = 0
    total = 0
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for i,data in enumerate(test_loader, 0):
            images, labels = data[0].to(device), data[1].to(device)
            # calculate outputs by running images through the network
            outputs = net(images)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return 100 * correct / total

def get_train_loader(dataset, valid_size=1024, batch_size=32):
    '''Split dataset into [train:valid] and return a DataLoader for the training part.'''

    indices = list(range(len(dataset)))
    train_sampler = torch.utils.data.SubsetRandomSampler(indices[valid_size:])
    train = torch.utils.data.DataLoader(dataset, sampler=train_sampler, batch_size=batch_size)

    return train

def get_validation_loader(dataset, valid_size=1024, batch_size=32):
    '''Split dataset into [train:valid] and return a DataLoader for the validation part.'''

    indices = list(range(len(dataset)))
    valid_sampler = torch.utils.data.SubsetRandomSampler(indices[:valid_size])
    valid = torch.utils.data.DataLoader(dataset, sampler=valid_sampler, batch_size=batch_size)

    return valid

def main():

    #### Parse command line arguments 
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-file", default=Net.model_file,
                        help="Name of the file used to load or to sore the model weights."\
                        "If the file exists, the weights will be load from it."\
                        "If the file doesn't exists, or if --force-train is set, training will be performed, "\
                        "and the model weights will be stored in this file."\
                        "Warning: "+Net.model_file+" will be used for testing (see load_for_testing()).")
    parser.add_argument('-f', '--force-train', action="store_true",
                        help="Force training even if model file already exists"\
                             "Warning: previous model file will be erased!).")
    parser.add_argument('-rt', '--resume_training', action="store_true",
                        help="Resumes training or not") 
    parser.add_argument('-e', '--num-epochs', type=int, default=10,
                        help="Set the number of epochs during training")
    parser.add_argument('-eps', '--epsilon', type=float, default=8/225,
                        help="epsilon parameter used during training by the PGD")
    parser.add_argument('-d', '--delta', type=float, default=2/225,
                        help="delta parameter used during training by the PGD")
    parser.add_argument('-i', '--iters', type=int, default=10,
                        help="number of iteration used during training by the PGD")
    parser.add_argument('-ratio', '--adv_to_nat_ratio', type=float, default=0.5,
                        help="ratio clean to adversarial input used during training")
    
    parser.add_argument('-mc', '--model-class', type=str, default='Net')
    args = parser.parse_args()

    #### Create model and move it to whatever device is available (gpu/cpu)
   
    models = importlib.import_module("model_class")
    net_class = getattr(models, args.model_class)
    net = net_class().to(device)

    num_samples = 1 # higher than 1 for randomized networks

    # during final testing.
    cifar = torchvision.datasets.CIFAR10('./data/', download=True, transform=transforms.ToTensor()) 
    valid_loader = get_validation_loader(cifar, valid_size)

    if args.resume_training:
        net.load(args.model_file, device)

    #### Model training (if necessary)
    if not os.path.exists(args.model_file) or args.force_train or args.resume_training:
        print("Training model")
        print(args.model_file)
       

        train_transform = transforms.Compose([transforms.ToTensor()]) 
        cifar = torchvision.datasets.CIFAR10('./data/', download=True, transform=train_transform)
        train_loader = get_train_loader(cifar, valid_size, batch_size=batch_size)
        net.train()
        adversarial_training(
                net,
                train_loader,
                valid_loader,
                args.model_file,
                args.num_epochs,
                num_samples,
                args.epsilon,
                args.delta,
                args.iters,
                args.adv_to_nat_ratio
                )
        print("Model save to '{}'.".format(args.model_file))

    #### Model testing
    print("Testing with model from '{}'. ".format(args.model_file))

    # Note: You should not change the transform applied to the
    # validation dataset since, it will be the only transform used
    # during final testing.
    

    net.load(args.model_file,device)
    net.eval()
    acc = test_natural(net, valid_loader)
    print("Model natural accuracy (valid): {}".format(acc))

    

    if args.model_file != Net.model_file:
        print("Warning: '{0}' is not the default model file, "\
              "it will not be the one used for testing your project. "\
              "If this is your best model, "\
              "you should rename/link '{0}' to '{1}'.".format(args.model_file, Net.model_file))

if __name__ == "__main__":
    main()

