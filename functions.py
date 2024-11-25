import sys
sys.path.append('../')

# libraries
import os
import numpy as np
import numpy.linalg as LA
import torch
import torch.nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.datasets as datasets
from torchvision import datasets, transforms
from torch.autograd import Variable

from resnet import resnet
from models.resnet_width import *

# mlptraining hyperparameters
IN_DIM = 28 * 28
OUT_DIM = 10
LR = 10 ** -2

# CUDA support
if torch.cuda.is_available():
    device = torch.device('cuda')
    FLAG = True
else:
    device = torch.device('cpu')

class MLPSmall(torch.nn.Module):
    """ Fully connected feed-forward neural network with one hidden layer. """
    def __init__(self, x_dim, y_dim):
        super().__init__()
        self.linear_1 = torch.nn.Linear(x_dim, 32)
        self.linear_2 = torch.nn.Linear(32, y_dim)

    def forward(self, x):
        h = F.relu(self.linear_1(x))
        return F.softmax(self.linear_2(h), dim=1)


class Flatten(object):
    """ Transforms a PIL image to a flat numpy array. """
    def __call__(self, sample):
        return np.array(sample, dtype=np.float32).flatten()    


def get_params(model_orig,  model_perb, direction, alpha):
    for m_orig, m_perb, d in zip(model_orig.parameters(), model_perb.parameters(), direction):
        m_perb.data = m_orig.data + alpha * d
    return model_perb


def getData(name='cifar10', train_bs=128, test_bs=1000, shuffle_not=True, train_index=None, normalize=True, normalize_svhn=False):    

    if name == 'mnist':

        train_loader = torch.utils.data.DataLoader(
            datasets.MNIST('../data', train=True, download=True,
                           transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize((0.1307,), (0.3081,))
                           ])),
            batch_size=train_bs, shuffle=True)
        test_loader = torch.utils.data.DataLoader(
            datasets.MNIST('../data', train=False, transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize((0.1307,), (0.3081,))
                           ])),
            batch_size=test_bs, shuffle=False)

    if name == 'cifar10':
        if normalize:
            transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

            transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        else:
            transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            #transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

            transform_test = transforms.Compose([
            transforms.ToTensor(),
            #transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        trainset = datasets.CIFAR10(root='../data', train=True, download=True, transform=transform_train)
        train_loader = torch.utils.data.DataLoader(trainset, batch_size=train_bs, shuffle=True)

        testset = datasets.CIFAR10(root='../data', train=False, download=False, transform=transform_test)
        test_loader = torch.utils.data.DataLoader(testset, batch_size=test_bs, shuffle=False)
    if name == 'cifar10_without_dataaugmentation':
        transform_train = transforms.Compose([
        # transforms.RandomCrop(32, padding=4),
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

        transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

        trainset = datasets.CIFAR10(root='../data', train=True, download=True, transform=transform_train)
        train_loader = torch.utils.data.DataLoader(trainset, batch_size=train_bs, shuffle=True)

        testset = datasets.CIFAR10(root='../data', train=False, download=False, transform=transform_test)
        test_loader = torch.utils.data.DataLoader(testset, batch_size=test_bs, shuffle=False)

    if name == 'cifar100':
        
        transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

        transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

        trainset = datasets.CIFAR100(root='../data', train=True, download=True, transform=transform_train)
        train_loader = torch.utils.data.DataLoader(trainset, batch_size=train_bs, shuffle=True)

        testset = datasets.CIFAR100(root='../data', train=False, download=False, transform=transform_test)
        test_loader = torch.utils.data.DataLoader(testset, batch_size=test_bs, shuffle=False)
    
    if name == 'svhn':
        
        if normalize_svhn:
            train_loader = torch.utils.data.DataLoader(
    datasets.SVHN('../data', split='train', download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                   ])),
    batch_size=train_bs, shuffle=True)
            
            test_loader = torch.utils.data.DataLoader(
    datasets.SVHN('../data', split='test', download=True,transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                   ])),
    batch_size=test_bs, shuffle=False)
            
        else:
            train_loader = torch.utils.data.DataLoader(
    datasets.SVHN('../data', split='train', download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor()
                   ])),
    batch_size=train_bs, shuffle=True)
            
            test_loader = torch.utils.data.DataLoader(
    datasets.SVHN('../data', split='test', download=True,transform=transforms.Compose([
                       transforms.ToTensor()
                   ])),
    batch_size=test_bs, shuffle=False)

    return train_loader, test_loader


def get_train_loader(dataset, trian_bs=128, test_bs=1000):
    if dataset == "mnist":
        # download MNIST and setup data loaders
        mnist_train = datasets.MNIST(root='../data', train=True, download=True, transform=Flatten())
        train_loader = torch.utils.data.DataLoader(mnist_train, batch_size=trian_bs, shuffle=False)
    if dataset == "cifar10":
        train_loader, test_loader = getData()
    return train_loader


# def getModel(model="mlp"):
#     if model == "mlp":
#         model = MLPSmall(IN_DIM, OUT_DIM)
#         optimizer = optim.Adam(model.parameters(), lr=LR)
#         model.load_state_dict(torch.load('../models/mnist_mlp.pt'))
#     if model == "resnet20":
#         # get the model 
#         model = ptcv_get_model("resnet20_cifar10", pretrained=True)
#     # set the model to eval mode
#     model.eval()
#     return model


def get_modified_resnet_model(model_depth, model_residual, model_batch_norm, seed=0):
    model = resnet(num_classes=10, depth=model_depth, residual_not=model_residual, batch_norm_not=model_batch_norm)
    model.load_state_dict(torch.load('saved_models/resnet' + str(model_depth) + '_batch_norm' + str(model_batch_norm) + '_residual' + str(model_residual) + '_seed' + str(seed) + '_net.pkl', map_location=torch.device('cpu')), strict=False)
    model = torch.nn.DataParallel(model)
    model.eval()
    return model

def get_resnet18_model(model_depth,subset,bs,width,seed,model_type='best'):
    model = ResNet18(width=width, num_classes=10).to(device)
    model_path = f'saved_models/ResNet{model_depth}_checkpoints/different_knobs_subset_{subset}/bs_{bs}/normal/ResNet{model_depth}_w{width}/'
    if model_type == 'best':
        model_name = f'net_exp_{seed}_{model_type}.pkl'
    elif model_type == 'early_stopped_model':
        model_name = f'net_exp_{seed}_{model_type}.pkl'
    else:
        model_name = f'net_exp_{seed}.pkl'
    model = torch.nn.DataParallel(model)
    state_dict = torch.load(model_path + model_name, map_location=torch.device(device))
    model.load_state_dict(state_dict)
    model.eval()
    return model

def get_pinn_model(system,u0_str,nu,beta,rho,N_f,layers,L,source,seed,lr):
    model_path = f"saved_models/PINN_checkpoints/PINN_{system}/lr_{lr}/beta_{beta}/"
    model_name = f"pretrained_{system}_u0{u0_str}_nu{nu}_beta{beta}_rho{rho}_Nf{N_f}_{layers}_L{L}_source{source}_seed{seed}.pt"
    model = torch.load(model_path+model_name, map_location=device)
    return model
