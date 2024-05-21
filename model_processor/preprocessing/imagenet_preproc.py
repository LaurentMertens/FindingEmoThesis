"""
Load and preprocess an image for use with ImageNet network.

.. codeauthor:: Laurent Mertens <laurent.mertens@kuleuven.be>
"""
import os

import PIL
import torch
from torchvision import transforms


class ImageNetPreProcess(object):
    FULL = "full"  # Full ImageNet processing: resize, to tensor, centercrop, normalize
    RANDOM = "random"  # Resize, to tensor,  random crop, random horizontal flip, normalize
    SMALL = "small"  # To tensor + normalize
    NORMALIZE = "normalize"  # Normalize using ImageNet values

    """
    Load and preprocess image for use with ImageNet network.
    """
    def __init__(self, chain_type="full"):
        """

        :param chain_tyoe: which processing chain to use; see ImageNetPreProcess constant descriptions for more info
        """
        print(f"Initializing ImageNetPreProcess with chain_type={chain_type}...")
        if chain_type == self.FULL:
            self.preprocess = transforms.Compose([
                transforms.Resize(256),
                transforms.ToTensor(),
                transforms.CenterCrop(224),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        elif chain_type == self.RANDOM:
            self.preprocess = transforms.Compose([
                transforms.Resize(256),
                transforms.ToTensor(),
                transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip(p=0.3),
                # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        elif chain_type == self.SMALL:
            self.preprocess = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        elif chain_type == self.NORMALIZE:
            self.preprocess = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        else:
            raise ValueError(f"Parameter 'chain_type' has invalid value: {chain_type}")

    def __call__(self, img: str or PIL.Image) -> torch.Tensor:
        """
        Resize image.

        :param img: Input image
        :return: the resized image, as torch.Tensor.
        """
        if isinstance(img, str):
            with open(img, 'rb') as f:
                img = PIL.Image.open(f).convert('RGB')
        img = self.preprocess(img)

        return img
