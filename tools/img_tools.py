"""
Methods relating to images

.. codeauthor:: Laurent Mertens <laurent.mertens@kuleuven.be>
"""
import PIL
import matplotlib.pyplot as plt
import torch
from PIL import Image


class ImageTools:
    @staticmethod
    def print_grayscale_img(t: torch.Tensor):
        """
        Print grayscale image to screen using MatPlotLib.

        :param t: 2D tensor representing the grayscale image; values are expected to be 0 <= x <= 255.n
        :return:
        """
        plt.figure()
        plt.imshow(t.cpu().numpy(), cmap='gray')
        plt.show()

    @staticmethod
    def open_image(img_path):
        # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
        with open(img_path, 'rb') as f:
            img = Image.open(f)
            img = img.convert('RGB')

        return img
