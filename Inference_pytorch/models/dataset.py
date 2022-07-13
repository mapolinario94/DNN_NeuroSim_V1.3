import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
import os

def get_cifar10(batch_size, data_root='/tmp/public_dataset/pytorch', train=True, val=True, **kwargs):
    data_root = os.path.expanduser(os.path.join(data_root, 'cifar10-data'))
    num_workers = kwargs.setdefault('num_workers', 1)
    kwargs.pop('input_size', None)
    print("Building CIFAR-10 data loader with {} workers".format(num_workers))
    ds = []
    if train:
        train_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10(
                root=data_root, train=True, download=True,
                transform=transforms.Compose([
                    transforms.Pad(4),
                    transforms.RandomCrop(32),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ])),
            batch_size=batch_size, shuffle=True, **kwargs)

        ds.append(train_loader)
    if val:
        test_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10(
                root=data_root, train=False, download=True,
                transform=transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ])),
            batch_size=batch_size, shuffle=False, **kwargs)
        ds.append(test_loader)
    ds = ds[0] if len(ds) == 1 else ds
    return ds

def get_cifar100(batch_size, data_root='/tmp/public_dataset/pytorch', train=True, val=True, **kwargs):
    data_root = os.path.expanduser(os.path.join(data_root, 'cifar100-data'))
    num_workers = kwargs.setdefault('num_workers', 1)
    kwargs.pop('input_size', None)
    print("Building CIFAR-100 data loader with {} workers".format(num_workers))
    ds = []
    if train:
        train_loader = torch.utils.data.DataLoader(
            datasets.CIFAR100(
                root=data_root, train=True, download=True,
                transform=transforms.Compose([
                    transforms.Pad(4),
                    transforms.RandomCrop(32),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ])),
            batch_size=batch_size, shuffle=True, **kwargs)
        ds.append(train_loader)

    if val:
        test_loader = torch.utils.data.DataLoader(
            datasets.CIFAR100(
                root=data_root, train=False, download=True,
                transform=transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ])),
            batch_size=batch_size, shuffle=False, **kwargs)
        ds.append(test_loader)
    ds = ds[0] if len(ds) == 1 else ds
    return ds

def get_imagenet(batch_size, data_root='/home/shimeng/Documents/Data', train=True, val=True, **kwargs):
    # data_root = data_root
    num_workers = kwargs.setdefault('num_workers', 1)
    print("Building ImageNet data loader with {} workers".format(num_workers))
    
    ds = []
    if train:
        transform=transforms.Compose([
            # transforms.Pad(4),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        train_path = os.path.join(data_root, 'train')
        imagenet_traindata = datasets.ImageFolder(train_path, transform=transform)
        train_loader = torch.utils.data.DataLoader(
            imagenet_traindata,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0)
        ds.append(train_loader)
    if val:
        transform=transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        val_path = os.path.join(data_root, 'val')
        imagenet_testdata = datasets.ImageFolder(val_path, transform=transform)
        test_loader = torch.utils.data.DataLoader(
            imagenet_testdata,
            batch_size=batch_size, 
            shuffle=False, 
            **kwargs)
        ds.append(test_loader)
    ds = ds[0] if len(ds) == 1 else ds
    return ds

def custom_dataset(args, batch_size, data_root='/tmp/public_dataset/pytorch', train=True, val=True, **kwargs):
    train_dataset = CustomDataset(args.model)
    num_workers = kwargs.setdefault('num_workers', 1)
    kwargs.pop('input_size', None)
    ds = []
    if train:
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, **kwargs)
        ds.append(train_loader)
    if val:
        test_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, **kwargs)
        ds.append(test_loader)

    ds = ds[0] if len(ds) == 1 else ds
    return ds


class CustomDataset(Dataset):
    def __init__(self, model):
        self.data = "Custom"
        self.model = model
        self.cfg_list_in_ch = {
            'l1': (4, 64),
            'l2': (64, 128),
            'l3': (128, 256),
            'l4': (256, 512),
            'l5': (512, 512),
            'l6': (512, 512),
            'l7': (512, 512),
            'l8': (512, 512),
            'l9': (512, 128),
            'l10': (416, 64),
            'l11': (224, 4),
            'l12': (512, 32),
            'l13': (416, 32),
            'l14': (224, 32),
            'l15': (100, 32),
            'l16': (32, 32),
            'l17': (4, 32),
            'l18': (4, 512),
            'SFN': (4, 1)
        }
        self.cfg_list = {
            'l1': (256, 256),
            'l2':  ( 128, 128),
            'l3':  ( 64, 64),
            'l4':  ( 32, 32),
            'l5':  ( 16, 16),
            'l6':  ( 16, 16),
            'l7':  ( 16, 16),
            'l8':  ( 16, 16),
            'l9':  ( 35, 35),
            'l10': ( 67, 67),
            'l11': ( 131, 131),
            'l12': ( 35, 35),
            'l13': ( 67, 67),
            'l14': ( 131, 131),
            'l15': ( 259, 259),
            'l16': (256,256),
            'l17': (256,256),
            'l18': (256, 256),
            'SFN': (256, 256),
        }
    def __len__(self):
        return 10

    def __getitem__(self, idx):
        ch = self.cfg_list_in_ch[self.model]
        in_ch = ch[0]
        out_ch = ch[1]

        size = self.cfg_list[self.model]
        torch.random.manual_seed(1234)
        img = torch.ones([in_ch, size[0], size[1]])
        label = torch.ones(1)

        return img, label