#!/usr/bin/env python3 
import os
import argparse
import torch
import torchvision
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
import torchvision.transforms as transforms



use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

valid_size = 1024 
batch_size = 32 

# '''Basic neural network architecture (from pytorch doc).'''
# class Net(nn.Module):

#     model_file="models/default_model.pth"
#     '''This file will be loaded to test your model. Use --model-file to load/store a different model.'''

#     def __init__(self):
#         super().__init__()
#         self.conv1 = nn.Conv2d(3, 6, 5)
#         self.pool = nn.MaxPool2d(2, 2)
#         self.conv2 = nn.Conv2d(6, 16, 5)
#         self.fc1 = nn.Linear(16 * 5 * 5, 120)
#         self.fc2 = nn.Linear(120, 84)
#         self.fc3 = nn.Linear(84, 10)
        
#     def forward(self, x):
#         x = self.pool(F.relu(self.conv1(x)))
#         x = self.pool(F.relu(self.conv2(x)))
#         x = torch.flatten(x, 1)
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         x = self.fc3(x)
#         x = F.log_softmax(x, dim=1)
#         return x

#     def save(self, model_file):
#         '''Helper function, use it to save the model weights after training.'''
#         torch.save(self.state_dict(), model_file)

#     def load(self, model_file):
#         self.load_state_dict(torch.load(model_file, map_location=torch.device(device)))

        
#     def load_for_testing(self, project_dir='./'):
#         '''This function will be called automatically before testing your
#            project, and will load the model weights from the file
#            specify in Net.model_file.
           
#            You must not change the prototype of this function. You may
#            add extra code in its body if you feel it is necessary, but
#            beware that paths of files used in this function should be
#            refered relative to the root of your project directory.
#         '''        
#         self.load(os.path.join(project_dir, Net.model_file))

class RSE_NoiseLayer(nn.Module):
    def __init__(self, sigma):
        super().__init__()
        self.sigma = sigma

    def forward(self, x):
        noise = torch.randn_like(x) * self.sigma
        return x + noise

class WideBlock_RSE(nn.Module):
    def __init__(self, in_planes, planes, dropout_rate, stride, sigma_inner):
        super().__init__()
        self.noise1 = RSE_NoiseLayer(sigma_inner)
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, padding=1, bias=False)

        self.noise2 = RSE_NoiseLayer(sigma_inner)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)

        self.dropout = nn.Dropout(dropout_rate)
        self.act = nn.GELU()

        self.shortcut = nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False) \
                        if stride != 1 or in_planes != planes else nn.Identity()

    def forward(self, x):
        out = self.noise1(x)
        out = self.bn1(out)
        out = self.act(out)
        out = self.conv1(out)

        out = self.dropout(out)

        out = self.noise2(out)
        out = self.bn2(out)
        out = self.act(out)
        out = self.conv2(out)

        return self.shortcut(x) + out

class WideResNet_RSE(nn.Module):
    model_file="models/default_model.pth"
    def __init__(self, depth=13, widen_factor=10, dropout_rate=0.3,
                 sigma_init=0.2, sigma_inner=0.1, num_classes=10):
        super().__init__()

        self.sigma_init = sigma_init
        self.sigma_inner = sigma_inner

        self.init_noise = RSE_NoiseLayer(sigma_init)

        self.in_planes = 16
        k = widen_factor
        n = (depth - 1) // 6

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)

        self.layer1 = self._make_layer(16*k, n, dropout_rate, stride=1)
        self.layer2 = self._make_layer(32*k, n, dropout_rate, stride=2)
        self.layer3 = self._make_layer(64*k, n, dropout_rate, stride=2)

        self.bn1 = nn.BatchNorm2d(64*k)
        self.act = nn.GELU()
        self.fc = nn.Linear(64*k, num_classes)

    def _make_layer(self, planes, num_blocks, dropout_rate, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for s in strides:
            layers.append(WideBlock_RSE(self.in_planes, planes, dropout_rate, s, self.sigma_inner))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.init_noise(x)
        out = self.conv1(x)

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)

        out = self.bn1(out)
        out = self.act(out)

        out = F.avg_pool2d(out, out.size(3))
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return F.log_softmax(out, dim=1)
    
    def save(self, model_file):
        torch.save(self.state_dict(), model_file)

    def load(self, model_file):
        self.load_state_dict(torch.load(model_file, map_location=torch.device(device)))

    def load_for_testing(self, project_dir='./'):
        self.load(os.path.join(project_dir, WideResNet_RSE.model_file))

class Net(WideResNet_RSE):
    model_file="models/default_model.pth"
    '''This file will be loaded to test your model. Use --model-file to load/store a different model.'''
    def __init__(self):
        super().__init__(depth=13, widen_factor=10, dropout_rate=0.3,
                 sigma_init=0.2, sigma_inner=0.1, num_classes=10)



def train_model(net, train_loader, pth_filename, num_epochs):
    '''Basic training function (from pytorch doc.)'''
    print("Starting training")
    criterion = nn.NLLLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    for epoch in range(num_epochs):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data[0].to(device), data[1].to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 500 == 499:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0

    net.save(pth_filename)
    print('Model saved in {}'.format(pth_filename))

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
    parser.add_argument('-e', '--num-epochs', type=int, default=10,
                        help="Set the number of epochs during training")
    args = parser.parse_args()

    #### Create model and move it to whatever device is available (gpu/cpu)
    net = Net()
    net.to(device)

    #### Model training (if necessary)
    if not os.path.exists(args.model_file) or args.force_train:
        print("Training model")
        print(args.model_file)

        train_transform = transforms.Compose([transforms.ToTensor()]) 
        cifar = torchvision.datasets.CIFAR10('./data/', download=True, transform=train_transform)
        train_loader = get_train_loader(cifar, valid_size, batch_size=batch_size)
        train_model(net, train_loader, args.model_file, args.num_epochs)
        print("Model save to '{}'.".format(args.model_file))

    #### Model testing
    print("Testing with model from '{}'. ".format(args.model_file))

    # Note: You should not change the transform applied to the
    # validation dataset since, it will be the only transform used
    # during final testing.
    cifar = torchvision.datasets.CIFAR10('./data/', download=True, transform=transforms.ToTensor()) 
    valid_loader = get_validation_loader(cifar, valid_size)

    net.load(args.model_file)

    acc = test_natural(net, valid_loader)
    print("Model natural accuracy (valid): {}".format(acc))

    if args.model_file != Net.model_file:
        print("Warning: '{0}' is not the default model file, "\
              "it will not be the one used for testing your project. "\
              "If this is your best model, "\
              "you should rename/link '{0}' to '{1}'.".format(args.model_file, Net.model_file))

if __name__ == "__main__":
    main()

