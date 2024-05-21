"""

.. codeauthor:: Laurent Mertens <laurent.mertens@kuleuven.be>
"""
import PIL
import torch
import torchvision


class ImgResize(object):
    """
    Resize image to pre-defined size and convert to torch.Tensor.

    The image will be resized such that it completely fits within the pre-defined size and will be further padded
    such as to fully obtain the pre-defined size.

    E.g., if the target dimensions are 800x600, and the image to be resized is 1000x600, the image will first be resized to
    800x480, and then its height will be further padded so that it becomes 800x600.
    """
    def __init__(self, width: int, height: int):
        if width <= 0:
            raise ValueError('Target width needs to be >0.')
        if height <= 0:
            raise ValueError('Target height needs to be >0.')
        self.width = width
        self.height = height
        self.target_ratio = self.width/self.height
        self.to_tensor = torchvision.transforms.ToTensor()

    def __call__(self, img: PIL.Image) -> torch.Tensor:
        """
        Resize image.

        :param img: Input image
        :return: the resized image, as torch.Tensor.
        """
        img_ratio = img.width/img.height
        # Resize image such that width is self.width, then pad height
        if img_ratio > self.target_ratio:
            # print("case 1")
            factor = self.width/img.width
            img = img.resize((self.width, int(factor*img.height)), resample=PIL.Image.BICUBIC)
            # padding=(left,top,right,bottom)
            pad_top = int((self.height - img.height)/2)
            pad_bottom = self.height - img.height - pad_top
            padder = torchvision.transforms.Pad(padding=(0, pad_top, 0, pad_bottom))
            img = padder(self.to_tensor(img))
        # Resize image such that height is 600, then pad width
        elif img_ratio < self.target_ratio:
            # print("case 2")
            factor = self.height/img.height
            img = img.resize((int(factor*img.width), self.height), resample=PIL.Image.BICUBIC)
            # padding=(left,top,right,bottom)
            pad_left = int((self.width - img.width)/2)
            pad_right = self.width - img.width - pad_left
            padder = torchvision.transforms.Pad(padding=(pad_left, 0, pad_right, 0))
            img = padder(self.to_tensor(img))
        # Both are equal; no padding needed
        else:
            # print("case 3")
            # Ratios are equal, so checking one side to check for dimension equality suffices.
            if self.width != img.width:
                img = img.resize((self.width, self.height), resample=PIL.Image.BICUBIC)
            img = self.to_tensor(img)

        return img
