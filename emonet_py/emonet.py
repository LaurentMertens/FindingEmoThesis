"""
PyTorch version of https://sites.google.com/colorado.edu/emonet/

By default, the network has 20 outputs, corresponding to the following emotions/states:
Adoration, Aesthetic Appreciation, Amusement, Anxiety, Awe, Boredom, Confusion, Craving,
Disgust, Empathetic Pain, Entrancement, Excitement, Fear, Horror, Interest, Joy, Romance,
Sadness, Sexual Desire, Surprise

.. codeauthor:: Laurent Mertens <laurent.mertens@kuleuven.be>
"""
import os

import PIL
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt

from alexnet_big import AlexNetBigEmoNet


class EmoNet:
    EMOTIONS = [
        'Adoration',
        'Aesthetic Appreciation',
        'Amusement',
        'Anxiety',
        'Awe',
        'Boredom',
        'Confusion',
        'Craving',
        'Disgust',
        'Empathetic Pain',
        'Entrancement',
        'Excitement',
        'Fear',
        'Horror',
        'Interest',
        'Joy',
        'Romance',
        'Sadness',
        'Sexual Desire',
        'Surprise'
    ]

    def __init__(self, b_eval=True):
        self.emonet = self.get_emonet(b_eval=b_eval)

    @staticmethod
    def get_emonet(b_eval=True):
        """
        Static method that returns an instance of 'AlexNetBig' initialized with the MatLab parameters.

        :param b_eval: set model to 'eval' mode.
        :return:
        """
        emonet = AlexNetBigEmoNet()
        # Get absolute path of current directory; this way the emonet_py.pth file can be opened, even
        # when using this package as an import in another project.
        abs_dir = os.path.dirname(os.path.abspath(__file__))
        emonet.load_state_dict(torch.load(os.path.join(abs_dir, '../data', 'emonet.pth')))
        if b_eval:
            emonet.eval()

        return emonet

    @staticmethod
    def prettyprint(preds: torch.Tensor, b_pc=True):
        """
        Prettyprint model predictions, writing out the emotions corresponding to each prediction value.

        :param preds: torch.Tensor object of shape (N, 20), with N the number of samples.
        :param b_pc: write out percentage values instead of probability scores.
        :return:
        """
        for sample in range(preds.shape[0]):
            if sample > 0:
                print("\n" + "="*60)
            print(f"Results for sample {sample}:")
            for emo_idx in range(20):
                if b_pc:
                    print(f"\t{EmoNet.EMOTIONS[emo_idx]:22s}: {100*preds[sample, emo_idx]:5.1f}%")
                else:
                    print(f"\t{EmoNet.EMOTIONS[emo_idx]:22s}: {preds[sample, emo_idx]:.4f}")


class EmoNetPreProcess(object):
    """
    Preprocess an image using the same preprocessing chain used by the MatLab model.
    """
    def __init__(self):
        # Load mean image pixel values used for image normalization in original MatLab implementation.
        mean = []
        # Get absolute path of current directory; this way the img_mean.txt file can be opened, even
        # when using this package as an import in another project.
        abs_dir = os.path.dirname(os.path.abspath(__file__))
        with open(os.path.join(abs_dir, '../data', 'img_mean.txt'), 'r') as fin:
            for line in fin:
                line = line.strip()
                parts = line.split(',')
                row = []
                for i in range(3):
                    row.append([float(parts[i]) for i in range(i*227, (i+1)*227)])
                mean.append(row)
        # Yet again, beware! Ordering of dimensions differs between PyTorch and MatLab.
        self.mean = torch.tensor(mean, dtype=torch.float).permute((0, 2, 1))

    def __call__(self, img: str or PIL.Image) -> torch.Tensor:
        """
        Load and preprocess an image:\
         -Resize the image to 227x227 if needed.\
         -Subtract the original mean pixel values from the (resized) image.

        :param img: path to the image file to be loaded, or PIL.Image.
        :return: PyTorch tensor containing the preprocessed image.
        """
        if isinstance(img, str):
            with open(img, 'rb') as f:
                img = Image.open(f).convert('RGB')
        # Note that the original implementation does not seem to use padding for images with
        # an aspect ratio != 1.
        if img.width != 227 or img.height != 227:
            img = img.resize((227, 227), resample=PIL.Image.BICUBIC)


        # Convert PIL Image to PyTorch Tensor.
        img_t = torch.tensor(np.asarray(img), dtype=torch.float)
        # Normalize with original mean pixel values.
        img_t -= self.mean
        # Permute dimensions: PIL Image puts the channel at the end, PyTorch models
        # expect it at the front.
        img_t = img_t.permute((2, 0, 1))

        return img_t, img
