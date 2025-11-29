#!/usr/bin/env python3 
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np




class WideBlock(nn.Module):
    def __init__(self, in_planes, planes, dropout_rate, stride=1):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size = 3, padding = 1, bias = False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size = 3 , stride = stride, padding = 1, bias = False)
        self.dropout = nn.Dropout(dropout_rate)
        self.act = nn.GELU()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Conv2d(in_planes, planes, kernel_size = 1, stride = stride, bias = False)
        else :
            self.shortcut = nn.Sequential()


    def forward(self, x):
        out = self.bn1(x) 
        out = self.act(out)
        out = self.conv1(out)
        out = self.dropout(out)
        out = self.bn2(out)
        out = self.act(out)
        out = self.conv2(out)
        return self.shortcut(x) + out

class WideResNet(nn.Module):
    model_file="models/default_model.pth"
    def __init__(self, depth=13, widen_factor=10, dropout_rate = 0.3):
        super().__init__()
        self.in_planes = 16
        k = widen_factor
        n = (depth-1)//6 # depth is the total count of convolutional layers
        self.conv1 = nn.Conv2d(3, 16, kernel_size = 3, stride = 1, padding = 1, bias = False)
        
        self.layer1 = self._make_layer(16*k, n, dropout_rate, stride=1)
        self.layer2 = self._make_layer(32*k, n, dropout_rate, stride=2)
        
        self.layer3 = self._make_layer(64*k, n, dropout_rate, stride=2)

        self.bn1 = nn.BatchNorm2d(64*k)
        self.linear = nn.Linear(64*k, 10)
        self.act = nn.GELU()

    def _make_layer(self, planes, num_blocks, dropout_rate, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(WideBlock(self.in_planes, planes, dropout_rate, stride))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        
        out = self.bn1(out)
        out = self.act(out)
        
        out = F.avg_pool2d(out, 8)
        
        out = torch.flatten(out,1)
        out = self.linear(out)
        out = F.log_softmax(out, dim=1)
        return out

    def save(self, model_file):
        '''Helper function, use it to save the model weights after training.'''
        torch.save(self.state_dict(), model_file)

    def load(self, model_file,device):
        self.load_state_dict(torch.load(model_file, map_location=torch.device(device)), strict = False)

        
    def load_for_testing(self, project_dir='./'):
        '''This function will be called automatically before testing your
           project, and will load the model weights from the file
           specify in Net.model_file.
           
           You must not change the prototype of this function. You may
           add extra code in its body if you feel it is necessary, but
           beware that paths of files used in this function should be
           refered relative to the root of your project directory.
        '''        
        self.load(os.path.join(project_dir, WideResNet.model_file))

class WideResNetNoisy(WideResNet):
    def __init__(self, noise_std=0.1):
        super().__init__()
        self.noise_std = noise_std
    def forward(self, x):
        noise = torch.randn_like(x)*self.noise_std
        x = x + noise
        return super().forward(x)
    





'''Basic neural network architecture (from pytorch doc).'''
class Net(nn.Module):

    model_file="models/default_model.pth"
    '''This file will be loaded to test your model. Use --model-file to load/store a different model.'''

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = F.log_softmax(x, dim=1)
        return x

    def save(self, model_file):
        '''Helper function, use it to save the model weights after training.'''
        torch.save(self.state_dict(), model_file)

    def load(self, model_file,device):
        self.load_state_dict(torch.load(model_file, map_location=torch.device(device)))

        
    def load_for_testing(self, project_dir='./'):
        '''This function will be called automatically before testing your
           project, and will load the model weights from the file
           specify in Net.model_file.
           
           You must not change the prototype of this function. You may
           add extra code in its body if you feel it is necessary, but
           beware that paths of files used in this function should be
           refered relative to the root of your project directory.
        '''        
        self.load(os.path.join(project_dir, Net.model_file))

class BigNetPCA(nn.Module):

    model_file="models/default_model.pth"
    '''This file will be loaded to test your model. Use --model-file to load/store a different model.'''

    def __init__(self):
        super().__init__()
        self.sim_PCA = nn.Sequential(
                nn.Linear(3072, 625),
                nn.Linear(625, 3072)
                )
        self.conv1 = nn.Conv2d(3, 12, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(12, 32, 5)
        self.fc1 = nn.Linear(32 * 5 * 5, 248)
        self.fc2 = nn.Linear(248, 160)
        self.fc3 = nn.Linear(160, 84)
        self.fc4 = nn.Linear(84,10)
        
    def forward(self, x):
        x = x.view(-1, 3072)
        x = self.sim_PCA(x)
        x = x.view(-1, 3, 32, 32)
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.gelu(self.fc1(x))
        x = F.gelu(self.fc2(x))
        x = F.gelu(self.fc3(x))
        x = self.fc4(x)
        x = F.log_softmax(x, dim=1)
        return x

    def save(self, model_file):
        '''Helper function, use it to save the model weights after training.'''
        torch.save(self.state_dict(), model_file)

    def load(self, model_file,device):
        self.load_state_dict(torch.load(model_file, map_location=torch.device(device)))

        
    def load_for_testing(self, project_dir='./'):
        '''This function will be called automatically before testing your
           project, and will load the model weights from the file
           specify in Net.model_file.
           
           You must not change the prototype of this function. You may
           add extra code in its body if you feel it is necessary, but
           beware that paths of files used in this function should be
           refered relative to the root of your project directory.
        '''        
        self.load(os.path.join(project_dir, BigNet.model_file))


class BigNet(nn.Module):

    model_file="models/default_model.pth"
    '''This file will be loaded to test your model. Use --model-file to load/store a different model.'''

    def __init__(self):
        super().__init__()
    
        self.conv1 = nn.Conv2d(3, 12, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(12, 32, 5)
        self.fc1 = nn.Linear(32 * 5 * 5, 248)
        self.fc2 = nn.Linear(248, 160)
        self.fc3 = nn.Linear(160, 84)
        self.fc4 = nn.Linear(84,10)
        
    def forward(self, x):
        
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.gelu(self.fc1(x))
        x = F.gelu(self.fc2(x))
        x = F.gelu(self.fc3(x))
        x = self.fc4(x)
        x = F.log_softmax(x, dim=1)
        return x

    def save(self, model_file):
        '''Helper function, use it to save the model weights after training.'''
        torch.save(self.state_dict(), model_file)

    def load(self, model_file,device):
        self.load_state_dict(torch.load(model_file, map_location=torch.device(device)))

        
    def load_for_testing(self, project_dir='./'):
        '''This function will be called automatically before testing your
           project, and will load the model weights from the file
           specify in Net.model_file.
           
           You must not change the prototype of this function. You may
           add extra code in its body if you feel it is necessary, but
           beware that paths of files used in this function should be
           refered relative to the root of your project directory.
        '''        
        self.load(os.path.join(project_dir, BigNet.model_file))



class BiggerNet(nn.Module):

    model_file="models/default_model.pth"
    '''This file will be loaded to test your model. Use --model-file to load/store a different model.'''

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 5)
        self.fc1 = nn.Linear(64 * 5 * 5, 248)
        self.fc2 = nn.Linear(248, 160)
        self.fc3 = nn.Linear(160, 100)
        self.fc4 = nn.Linear(100, 84)
        self.fc5 = nn.Linear(84,10)
        self.dropout = nn.Dropout(0.3)
    def forward(self, x):
        x = self.pool(F.gelu(self.conv1(x)))
        x = self.dropout(x)
        x = self.pool(F.gelu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.gelu(self.fc1(x))
        x = F.gelu(self.fc2(x))
        x = F.gelu(self.fc3(x))
        x = F.gelu(self.fc4(x))
        x = self.fc5(x)
        x = F.log_softmax(x, dim=1)
        return x

    def save(self, model_file):
        '''Helper function, use it to save the model weights after training.'''
        torch.save(self.state_dict(), model_file)

    def load(self, model_file,device):
        self.load_state_dict(torch.load(model_file, map_location=torch.device(device)))

        
    def load_for_testing(self, project_dir='./'):
        '''This function will be called automatically before testing your
           project, and will load the model weights from the file
           specify in Net.model_file.
           
           You must not change the prototype of this function. You may
           add extra code in its body if you feel it is necessary, but
           beware that paths of files used in this function should be
           refered relative to the root of your project directory.
        '''        
        self.load(os.path.join(project_dir, BigNet.model_file))



