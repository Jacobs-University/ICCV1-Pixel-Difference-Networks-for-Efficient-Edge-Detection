from torch.utils import data
import torchvision.transforms as transforms
import os
from pathlib import Path
from PIL import Image
import numpy as np




class Dataloader_BSDS500(data.Dataset):
    """
    Dataloader BSDS500
    """
    def __init__(self, root='data/HED-BSDS', split='train', transform=False, threshold=0.3, ablation=False):
        self.root = root
        self.split = split
        self.threshold = threshold * 256
        print('Threshold for ground truth: %f on BSDS' % self.threshold)
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            normalize])
        if self.split == 'train':
            if ablation:
                self.filelist = os.path.join(self.root, 'train200_pair.lst')
            else:
                self.filelist = os.path.join(self.root, 'train_pair.lst')
        elif self.split == 'test':
            if ablation:
                self.filelist = os.path.join(self.root, 'val.lst')
            else:
                self.filelist = os.path.join(self.root, 'test.lst')
        else:
            raise ValueError("Invalid split type!")
        with open(self.filelist, 'r') as f:
            self.filelist = f.readlines()

    def __len__(self):
        return len(self.filelist)
    
    def __getitem__(self, index):
        if self.split == "train":
            img_file, lb_file = self.filelist[index].split()
            img_file = img_file.strip()
            lb_file = lb_file.strip()
            lb = np.array(Image.open(os.path.join(self.root, lb_file)), dtype=np.float32)
            if lb.ndim == 3:
                lb = np.squeeze(lb[:, :, 0])
            assert lb.ndim == 2
            threshold = self.threshold
            lb = lb[np.newaxis, :, :]
            lb[lb == 0] = 0
            lb[np.logical_and(lb>0, lb<threshold)] = 2
            lb[lb >= threshold] = 1
            
        else:
            img_file = self.filelist[index].rstrip()

        with open(os.path.join(self.root, img_file), 'rb') as f:
            img = Image.open(f)
            img = img.convert('RGB')
        img = self.transform(img)

        if self.split == "train":
            return img, lb
        else:
            img_name = Path(img_file).stem
            return img, img_name


class BSDS_VOCLoader(data.Dataset):
    """
    Dataloader BSDS500
    """
    def __init__(self, root='data/HED-BSDS_PASCAL', split='train', transform=False, threshold=0.3, ablation=False):
        self.root = root
        self.split = split
        self.threshold = threshold * 256
        print('Threshold for ground truth: %f on BSDS_VOC' % self.threshold)
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            normalize])
        if self.split == 'train':
            if ablation:
                self.filelist = os.path.join(self.root, 'bsds_pascal_train200_pair.lst')
            else:
                self.filelist = os.path.join(self.root, 'bsds_pascal_train_pair.lst')
        elif self.split == 'test':
            if ablation:
                self.filelist = os.path.join(self.root, 'val.lst')
            else:
                self.filelist = os.path.join(self.root, 'test.lst')
        else:
            raise ValueError("Invalid split type!")
        with open(self.filelist, 'r') as f:
            self.filelist = f.readlines()

    def __len__(self):
        return len(self.filelist)
    
    def __getitem__(self, index):
        if self.split == "train":
            img_file, lb_file = self.filelist[index].split()
            img_file = img_file.strip()
            lb_file = lb_file.strip()
            lb = np.array(Image.open(os.path.join(self.root, lb_file)), dtype=np.float32)
            if lb.ndim == 3:
                lb = np.squeeze(lb[:, :, 0])
            assert lb.ndim == 2
            threshold = self.threshold
            lb = lb[np.newaxis, :, :]
            lb[lb == 0] = 0
            lb[np.logical_and(lb>0, lb<threshold)] = 2
            lb[lb >= threshold] = 1
            
        else:
            img_file = self.filelist[index].rstrip()

        with open(os.path.join(self.root, img_file), 'rb') as f:
            img = Image.open(f)
            img = img.convert('RGB')
        img = self.transform(img)

        if self.split == "train":
            return img, lb
        else:
            img_name = Path(img_file).stem
            return img, img_name


class Custom_Loader(data.Dataset):
    """
    Custom Dataloader
    """
    def __init__(self, root='data/'):
        self.root = root
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            normalize])
        self.imgList = fold_files(os.path.join(root))

    def __len__(self):
        return len(self.imgList)
    
    def __getitem__(self, index):

        with open(os.path.join(self.root, self.imgList[index]), 'rb') as f:
            img = Image.open(f)
            img = img.convert('RGB')
        img = self.transform(img)

        filename = Path(self.imgList[index]).stem

        return img, filename



def fold_files(foldname):
    """All files in the fold should have the same extern"""
    allfiles = os.listdir(foldname)
    if len(allfiles) < 1:
        raise ValueError('No images in the data folder')
        return None
    else:
        return allfiles