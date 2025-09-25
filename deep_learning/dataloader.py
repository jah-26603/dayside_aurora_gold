import os
import torch.nn as nn
import torchvision.transforms.v2 as v2
import torchvision.transforms.v2.functional as TF
from torch.utils.data import Dataset
import numpy as np
import torch

import matplotlib.pyplot as plt


class SegmentationDataset(Dataset):
    def __init__(self, root_dir, transform=None, aurora = True, north = False):
        self.root_dir = root_dir
        self.transform = transform
        self.aurora = aurora
        self.north = north
        self.image_label_pairs = self._load_image_label_pairs()
        
    def _load_image_label_pairs(self):
        pairs = []
        if self.aurora is True: 
            for subdir in os.listdir(self.root_dir):
                lbh_dir = os.path.join(self.root_dir, subdir, 'graphics', 'LBH')
                raw_north_dir = os.path.join(lbh_dir, 'raw_north')
                results_dir = os.path.join(lbh_dir, 'results')
    
                if not os.path.isdir(raw_north_dir) or not os.path.isdir(results_dir):
                    continue
    
                # Match files by unique identifier (last 9 to 4 characters of filename)
                raw_north_files = {f"{subdir}_{f[-9:-4]}": os.path.join(raw_north_dir, f)
                                   for f in os.listdir(raw_north_dir) if f.endswith('.npy')}
                results_files = {f"{subdir}_{f[-9:-4]}": os.path.join(results_dir, f)
                                 for f in os.listdir(results_dir) if f.endswith('.npy')}
    
                common_keys = set(raw_north_files.keys()) & set(results_files.keys())
    
                for key in common_keys:
                    pairs.append((raw_north_files[key], results_files[key]))
    
            return pairs
        else:
            if self.north:
                raw_north_dir = os.path.join(self.root_dir, 'sza')
                results_dir = os.path.join(self.root_dir, 'south')
                raw_north_files = {f[6:-4]: os.path.join(raw_north_dir, f)
                                   for f in os.listdir(raw_north_dir) if f.endswith('.npy')}
                results_files = {f[5:-4]: os.path.join(results_dir, f)
                                 for f in os.listdir(results_dir) if f.endswith('.npy')}
    
                common_keys = set(raw_north_files.keys()) & set(results_files.keys())
        
                for key in common_keys:
                    pairs.append((raw_north_files[key], results_files[key]))
            
            else:
                raw_north_dir = os.path.join(self.root_dir, 'sza')
                results_dir = os.path.join(self.root_dir, 'south')
        
                raw_north_files = {f[4:-4]: os.path.join(raw_north_dir, f)
                                   for f in os.listdir(raw_north_dir) if f.endswith('.npy')}
                results_files = {f[6:-4]: os.path.join(results_dir, f)
                                 for f in os.listdir(results_dir) if f.endswith('.npy')}

                common_keys = set(raw_north_files.keys()) & set(results_files.keys())
        
                for key in common_keys:
                    pairs.append((raw_north_files[key], results_files[key]))
        return pairs

    def __len__(self):
        return len(self.image_label_pairs)

    def __getitem__(self, idx):
        image_path, label_path = self.image_label_pairs[idx]
        
        image = np.load(image_path).astype(np.float32)

        label = np.load(label_path).astype(np.float32)

        if not self.aurora:
            image[:,:,0][np.isnan(image[:,:,0])] = 0
            image[:,:,2][np.isnan(image[:,:,2])] = 0
            image = torch.tensor(np.stack([np.cos(image[:,:,0] *np.pi /180),#sza
                                           np.sin(image[:,:,0] *np.pi /180), 
                                           np.cos(image[:,:,1] * 2*np.pi/366),#doy
                                           np.sin(image[:,:,1]* 2*np.pi/366),
                                           np.cos(image[:,:,2] * np.pi/180),
                                           np.sin(image[:,:,2]* np.pi/180),
                                           image[:,:,3],image[:,:,4]], 
                                          axis=0), dtype=torch.float32)
            
        else:
            image = torch.from_numpy(image).unsqueeze(0)  # Add channel dim if grayscale
        label = torch.from_numpy(label).unsqueeze(0)

        if self.transform:
            image, label = self.transform(image, label)

        label = label.squeeze(0)  # Remove the singleton dimension
        return image, label







class JointCompose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, label):
        for t in self.transforms:
            image, label = t(image, label)
        return image, label
class JointResize:
    def __init__(self, size):
        self.size = size

    def __call__(self, image, label):
        image = TF.resize(image, self.size)
        label = TF.resize(label, self.size, interpolation=TF.InterpolationMode.NEAREST)
        return image, label

class JointRandomHorizontalFlip:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, image, label):
        if torch.rand(1) < self.p:
            image = TF.hflip(image)
            label = TF.hflip(label)
        return image, label

class JointRandomVerticalFlip:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, image, label):
        if torch.rand(1) < self.p:
            image = TF.vflip(image)
            label = TF.vflip(label)
        return image, label

class JointNormalize:
    def __init__(self, mean, std, aurora = True):
        self.aurora = aurora
        self.mean = mean
        self.std = std

    def __call__(self, image, label):

        if self.aurora:
            label = (label > 0).float() #binarizes the labels
            for c in range(image.shape[0]):
                image[c] = image[c]/torch.max(image[c])
                mean = image[c].mean()
                std = image[c].std()
                # image[c] = (image[c] - self.mean[c]) / (self.std[c] + 1e-8)
                image[c] = (image[c] - mean) / (std + 1e-6)
                
        else:
            image[-2:] = TF.normalize(image[-2:], mean=self.mean[-1], std=self.std[-1])
            label = TF.normalize(label, mean=self.mean[:-1], std=self.std[:-1])
            
            
        return image, label
    
class JointRandomChannelShuffle:
    def __call__(self, img, label):
        # Shuffle channels in img only, not label
        perm = torch.randperm(img.shape[0])
        img = img[perm]
        if np.random.rand() < .1:
            img[0].zero_()
            
        # perm2 = torch.randperm(img.shape[0])
        # img = img[perm2]
        
        # Compute inverse permutation
        inv_perm = torch.argsort(perm)
        
        # Restore original order
        img = img[inv_perm]
        return img, label    


from torchvision.transforms import functional as F


class JointPad:
    def __init__(self, div_by=16):
        self.div_by = div_by
        
    def __call__(self, image, label):
        # Calculate padding to make dimensions divisible by div_by
        h, w = image.shape[-2:]
        pad_h = (((h // self.div_by) + 1) * self.div_by - h) % self.div_by
        pad_w = (((w // self.div_by) + 1) * self.div_by - w) % self.div_by
        
        # Apply padding
        padding = (0, 0, pad_w, pad_h)  # left, top, right, bottom
        image = F.pad(image, padding, 0)
        label = F.pad(label, padding, 0)
        
        return image, label
    
    
class JointRandomAffine:
    def __init__(self, degrees, translate=None, scale=None, shear=None):
        self.degrees = degrees
        self.translate = translate
        self.scale = scale
        self.shear = shear
        
    def __call__(self, image, mask):
        # Get the parameters for the affine transformation
        angle, translations, scale, shear = v2.RandomAffine.get_params(
            self.degrees, self.translate, self.scale, self.shear, tuple(image.size()[1:]))
        
        # Create the affine transform instance
        affine_transform = v2.functional.affine
        
        # # Apply the same transformation to both image and mask
        # transformed_image = affine_transform(image, angle, translations, scale, shear, fill=0)
        # transformed_mask = affine_transform(mask, angle, translations, scale, shear, fill=0)
        if mask.dim() == 2:
            mask = mask.unsqueeze(0)  # [H, W] -> [1, H, W]
        
        transformed_image = affine_transform(image, angle, translations, scale, shear, fill=0)
        transformed_mask = affine_transform(mask, angle, translations, scale, shear, fill=0)
        
        if transformed_mask.shape[0] == 1:
            transformed_mask = transformed_mask.squeeze(0)  # back to [H, W]
        
        return transformed_image, transformed_mask
