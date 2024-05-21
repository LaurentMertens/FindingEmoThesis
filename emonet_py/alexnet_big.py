"""
Implementation of the "full" 1st version of AlexNet described in the paper "ImageNet Classification
with Deep Convolutional Neural Networks", by Krizhevsky et al.[1], as opposed to the slightly smaller
2nd version of AlexNet that comes with torchvision.models, and is described in "One weird trick for
parallelizing convolutional neural networks", by Krizhevsky[2].

This is also the AlexNet version used by the original EmoNet MatLab implementation.

[1] https://papers.nips.cc/paper/2012/hash/c399862d3b9d6b76c8436e924a68c45b-Abstract.html
[2] https://arxiv.org/abs/1404.5997
"""
import torch.nn.functional
from torch import nn
from torch.nn import functional as F


class AlexNetBig(nn.Module):
    """
    This is the original MatLab output when printing out the layers of the AlexNet/EmoNet network.
    23×1 Layer array with layers:

     1   'input'         Image Input                   227×227×3 images with 'zerocenter' normalization
     2   'conv1'         Convolution                   96 11×11×3 convolutions with stride [4  4] and padding [0  0  0  0]
     3   'relu1'         ReLU                          ReLU
     4   'norm1'         Cross Channel Normalization   cross channel normalization with 5 channels per element
     5   'pool1'         Max Pooling                   3×3 max pooling with stride [2  2] and padding [0  0  0  0]
     6   'conv2'         Convolution                   256 5×5×48 convolutions with stride [1  1] and padding [2  2  2  2]
     7   'relu2'         ReLU                          ReLU
     8   'norm2'         Cross Channel Normalization   cross channel normalization with 5 channels per element
     9   'pool2'         Max Pooling                   3×3 max pooling with stride [2  2] and padding [0  0  0  0]
    10   'conv3'         Convolution                   384 3×3×256 convolutions with stride [1  1] and padding [1  1  1  1]
    11   'relu3'         ReLU                          ReLU
    12   'conv4'         Convolution                   384 3×3×192 convolutions with stride [1  1] and padding [1  1  1  1]
    13   'relu4'         ReLU                          ReLU
    14   'conv5'         Convolution                   256 3×3×192 convolutions with stride [1  1] and padding [1  1  1  1]
    15   'relu5'         ReLU                          ReLU
    16   'pool5'         Max Pooling                   3×3 max pooling with stride [2  2] and padding [0  0  0  0]
    17   'fc6'           Fully Connected               4096 fully connected layer
    18   'relu6'         ReLU                          ReLU
    19   'fc7'           Fully Connected               4096 fully connected layer
    20   'relu7'         ReLU                          ReLU
    21   'fc'            Fully Connected               20 fully connected layer
    22   'softmax'       Softmax                       softmax
    23   'classoutput'   Classification Output         crossentropyex with 'Adoration' and 19 other classes
    """
    def __init__(self, nb_outputs=20):
        super().__init__()
        self.lr_norm = nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=1.0)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2)

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=96, kernel_size=11, stride=4, padding=0)
        self.conv2 = nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5, stride=1, padding=2, groups=2)
        self.conv3 = nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, stride=1, padding=1, groups=2)
        self.conv5 = nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, stride=1, padding=1, groups=2)
        self.fc1 = nn.Linear(in_features=9216, out_features=4096)
        self.fc2 = nn.Linear(in_features=4096, out_features=4096)
        self.fc3 = nn.Linear(in_features=4096, out_features=nb_outputs)

    def forward(self, x):
        # Block 1
        x = self.maxpool(self.lr_norm(F.relu(self.conv1(x))))
        # Block 2
        x = self.maxpool(self.lr_norm(F.relu(self.conv2(x))))
        # Block 3
        x = F.relu(self.conv3(x))
        # Block 4
        x = F.relu(self.conv4(x))
        # Block 5
        x = self.maxpool(F.relu(self.conv5(x)))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        if not self.training:
            return torch.softmax(x, dim=1)

        return x


class AlexNetBigEmoNet(AlexNetBig):
    """
    Identical to AlexNetBig, but with a slightly altered 'forward' method with an added
    permutation between the last convolution layer and first linear layer, as this
    permutation is needed to align the ordering of the flattened output of the convolution
    layer with the original ordering in the MatLab model.

    Also, the number of outputs is fixed to 20.

    """
    def __init__(self):
        super().__init__(nb_outputs=20)

    def forward(self, x):
        # Block 1
        x = self.maxpool(self.lr_norm(F.relu(self.conv1(x))))
        # Block 2
        x = self.maxpool(self.lr_norm(F.relu(self.conv2(x))))
        # Block 3
        x = F.relu(self.conv3(x))
        # Block 4
        x = F.relu(self.conv4(x))
        # Block 5
        x = self.maxpool(F.relu(self.conv5(x)))
        x = x.permute((0, 1, 3, 2)).reshape(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        if not self.training:
            return torch.softmax(x, dim=1)

        return x
