#!/usr/bin/env python3 
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import random

# reproducibility
random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)

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

##################################
### Random Self-Ensemble (RSE) ###
##################################

class RSE_NoiseLayer(nn.Module):
    def __init__(self, sigma):
        super().__init__()
        self.sigma = sigma

    def forward(self, x):
        noise = torch.randn_like(x) * self.sigma
        return x + noise

class BigNet_RSE(nn.Module):
    model_file="models/default_model.pth"
    def __init__(self, sigma_init=0.2, sigma_inner=0.1): # values from the paper
        super().__init__()
        # noise layers
        self.init_noise  = RSE_NoiseLayer(sigma_init)
        self.inner_noise1 = RSE_NoiseLayer(sigma_inner)
        # original BigNet layers
        self.conv1 = nn.Conv2d(3, 12, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(12, 32, 5)
        self.fc1 = nn.Linear(32 * 5 * 5, 248)
        self.fc2 = nn.Linear(248, 160)
        self.fc3 = nn.Linear(160, 84)
        self.fc4 = nn.Linear(84,10)
        
    def forward(self, x):
        # init-noise before first conv
        x = self.init_noise(x)
        x = self.pool(F.relu(self.conv1(x)))
        # inner-noise before second conv
        x = self.inner_noise1(x)
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
        self.load(os.path.join(project_dir, BigNet_RSE.model_file))

# When testing, use multiple forward passes and average the outputs to get the final prediction, something like:
# def predict_ensemble_for_one(self, x, n=10):
#     """Forward-pass multiple times and return averaged predictions."""
#     self.eval()
#     preds = []
#     with torch.no_grad():
#         for _ in range(n):
#             logits = self.forward(x)
#             preds.append(logits)
#     return torch.mean(torch.stack(preds), dim=0)

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

    def load(self, model_file,device):
        self.load_state_dict(torch.load(model_file, map_location=torch.device(device)), strict = False)

    def load_for_testing(self, project_dir='./'):
        self.load(os.path.join(project_dir, WideResNet_RSE.model_file))




########################################
### Parametric Noise Injection (PNI) ###
########################################

# Train it with adversarial training (PGD) else alpha will go to zero according to the paper

class PNI(nn.Module):
    def __init__(self, weight_tensor):
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(0.25)) # learnable scaling factor for the noise
        self.register_buffer("sigma", weight_tensor.std().detach())

    def forward(self, w):
        noise = torch.randn_like(w) * self.sigma
        return w + self.alpha * noise
    
class PNI_Conv2d(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pni = PNI(self.weight)

    def forward(self, x):
        w_tilde = self.pni(self.weight)
        return nn.functional.conv2d(
            x, w_tilde, self.bias, self.stride,
            self.padding, self.dilation, self.groups
        )

class PNI_Linear(nn.Linear):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pni = PNI(self.weight)

    def forward(self, x):
        w_tilde = self.pni(self.weight)
        return nn.functional.linear(x, w_tilde, self.bias)

class BigNet_PNI(nn.Module):
    model_file="models/default_model.pth"
    
    def __init__(self):
        super().__init__()
        self.conv1 = PNI_Conv2d(3, 12, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = PNI_Conv2d(12, 32, 5)
        self.fc1 = PNI_Linear(32 * 5 * 5, 248)
        self.fc2 = PNI_Linear(248, 160)
        self.fc3 = PNI_Linear(160, 84)
        self.fc4 = PNI_Linear(84,10)
    
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
        torch.save(self.state_dict(), model_file)
    
    def load(self, model_file,device):
        self.load_state_dict(torch.load(model_file, map_location=torch.device(device)))

    def load_for_testing(self, project_dir='./'):
        self.load(os.path.join(project_dir, BigNet_PNI.model_file))

#################################
### Random Resize and Padding ###
#################################

class RandomResizePadding(nn.Module):
    def __init__(self, min_size=32, max_size=40, final_size=32):
        super().__init__()
        self.min_size = min_size
        self.max_size = max_size
        self.final_size = final_size

    def forward(self, x):
        # B, C, H, W = x.shape
        # random resize
        new_size = random.randint(self.min_size, self.max_size)
        x = F.interpolate(x, size=(new_size, new_size), mode='bilinear', align_corners=False)

        # random padding
        pad_total = self.max_size - new_size
        pad_left = random.randint(0, pad_total)
        pad_right = pad_total - pad_left
        pad_top = random.randint(0, pad_total)
        pad_bottom = pad_total - pad_top

        x = F.pad(x, (pad_left, pad_right, pad_top, pad_bottom))

        # need to resize back to original size because BigNet expects 32x32 input
        if self.final_size is not None:
            x = F.interpolate(x, size=(self.final_size, self.final_size), mode='bilinear')

        return x

class SmallFullConv(nn.Module): # to be used with RandomResizePadding at testing time

    model_file = "models/default_model.pth"

    def __init__(self):
        super().__init__()

        # 32×32 → 28×28
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, padding=0)
        # 28×28 → 14×14
        self.pool1 = nn.MaxPool2d(2, 2)

        # 14×14 → 10×10
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, padding=0)
        # 10×10 → 5×5
        self.pool2 = nn.MaxPool2d(2, 2)

        # 5×5 → 5×5 (1×1 conv)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)

        # 1×1 classification head (remplace les FC)
        self.classifier = nn.Conv2d(64, 10, kernel_size=1)

        self.gap = nn.AdaptiveAvgPool2d((1, 1))


    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)

        x = F.relu(self.conv2(x))
        x = self.pool2(x)

        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))

        x = self.classifier(x)     # B × 10 × 5 × 5
        x = self.gap(x)            # B × 10 × 1 × 1
        x = x.view(x.size(0), 10)  # Flatten class scores

        return F.log_softmax(x, dim=1)


    def save(self, model_file):
        torch.save(self.state_dict(), model_file)

    def load(self, model_file, device):
        self.load_state_dict(torch.load(model_file, map_location=torch.device(device)))

    def load_for_testing(self, project_dir="./"):
        self.load(os.path.join(project_dir, SmallFullConv.model_file))



# Use the randomization block only at inference time
# Train BigNet normally, then when testing, prepend the RandomResizePadding block
# Something like model = nn.Sequential(RandomResizePadding(), BigNet())
# NB: apparently, averaging multiple forward passes (with different random resizes/paddings) improves robustness