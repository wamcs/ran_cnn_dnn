import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch
import os
from data.Datasets import Dataset

train_path = '/home/data/data/ImageNet_ilsvrc2012_2014/'
test_path = './test/'

# [-1,1]
transform = transforms.Compose([
    # transforms.Resize((224,224)),
    transforms.Scale((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    # transforms.Normalize(mean = [ 0.5, 0.5, 0.5],
    #                      std = [ 0.5, 0.5, 0.5 ]),
])


def get_train_data(batch=1,path=train_path):
    traindir = os.path.join(path, 'train')
    train = Dataset(traindir, transform=transform)
    train_loader = torch.utils.data.DataLoader(train, batch_size=batch, shuffle=True, num_workers=1)
    return train_loader


def get_test_data(batch=1,path=test_path):
    valdir = os.path.join(path, 'val')
    val = Dataset(valdir, transform=transform)
    val_loader = torch.utils.data.DataLoader(val, batch_size=batch, shuffle=True, num_workers=1)
    return val_loader
