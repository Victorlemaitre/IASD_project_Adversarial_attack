
This repo is the third project of the datalab course at master IASD. 

The goal was to train a classifier on CIFAR-10 images and make it robust to white-box adversarial attacks like FGSM and PGD. 

A short list of what we did (sections with a * are my own contribution) : 
- Implemented FSGM and PGD 
- Trained a classifier using curriculum adversarial training *
- Tested more complex architectures like Vision Transformers and Wide ResNet *
- Dabbled in data augmentation and synthetic data generation using conditional Gans *
- Added noise in the form of noise layer or a learnable noise parameter
- Tried black box gradient estimation attacks


You can find a much more detailed account of our work in our final [report](report.pdf). 


Just for fun. Here are the best synthetic images we got using a conditional Gan inspired from the BigGan architecture. 

<p align="center">
  <img src="./Gan/EMA_epoch_200.png" alt="CIFAR-10 images">
</p>

Sadly even with our best efforts (including exponential moving average and discriminator driven latent sampling !) we did not get satisfying results :(
