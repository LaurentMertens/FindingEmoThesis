"""
Convert weights from EmoNet (https://sites.google.com/colorado.edu/emonet/) MatLab implementation to
Pytorch "state_dict" that can be easily loaded afterwards.

As basis for this conversion, we use (bzip2'ed) text files generated using the MatLab 'writematrix' command
to write the individual convolution and linear layers' bias and weight parameters.
Care should be taken to load this data in the correct way! The ordering of the dimensions in MatLab is
different from the one used in PyTorch!
.. codeauthor:: Laurent Mertens <laurent.mertens@kuleuven.be>
"""
import bz2
import os

import torch

from alexnet_big import AlexNetBigEmoNet


def load_lin_bias(f):
    """
    Load bias parameters for linear layer.

    :param f: path to the bz2 file containing the bias parameters.
    :return:
    """
    data = []
    with bz2.open(f, 'rb') as fin:
        for line in fin:
            line = line.decode('utf8')
            b = float(line.strip())
            data.append(b)

    bias = torch.as_tensor(data, dtype=torch.float32)

    return bias


def load_lin_weights(f):
    """
    Load weight parameters for linear layer.

    :param f: path to the bz2 file containing the weight parameters.
    :return:
    """
    weights = []
    with bz2.open(f, 'rb') as fin:
        for line in fin:
            line = line.decode('utf8')
            data = [float(w) for w in line.strip().split(',')]
            weights.append(data)

    weights = torch.as_tensor(weights, dtype=torch.float32)

    return weights


def load_conv_bias(f):
    """
    Load bias parameters for convolution layer.

    :param f: path to the bz2 file containing the bias parameters.
    :return:
    """
    with bz2.open(f, 'rb') as fin:
        for line in fin:
            line = line.decode('utf8')
            data = [float(w) for w in line.strip().split(',')]
    bias = torch.as_tensor(data, dtype=torch.float32)

    return bias


def load_conv_weights(f, orig_shape: tuple, b_inverse=False):
    """
    Load weight parameters for linear layer.

    :param f: path to the bz2 file containing the weight parameters.
    :param orig_shape: tuple describing the shape of the weight matrix in MatLab.
    :param b_inverse: for the first convolution layer, this should be set to 'False', and 'True' for all the others.\
    Assuming a filtersize of 'n' by 'n' pixels, the first convolution layer (with 3 RGB input channels) follows the\
    following formatting per line in the text file (ch. = channel):\
    \
    |[n floats][n floats][n floats]|[n floats][n floats][n floats]|...\
    |[ch.1    ][ch.2    ][ch.3    ]|[ch.1    ][ch.2    ][ch.3    ]|...\
    |[filter 1                    ]|[filter 2                    ]|...\
    So the first n floats belong to channel 1 of filter 1, the second set of n floats belong to channel 2 of filter 1, etc.
    \
    The other convolution layers follow the following formatting:\
    \
    |[n floats][n floats][n floats]...[n floats]|...\
    |[filter 1][filter 2][filter 3]...[filter x]|...\
    |[channel 1                                ]|...\
    So the first n floats belong to filter 1 of channel 1, the second set of n floats belong to filter 2 of channel 1, etc.

    :return:
    """
    weights = torch.zeros(orig_shape, dtype=torch.float32)

    at_row = -1
    with bz2.open(f, 'rb') as fin:
        for line in fin:
            line = line.decode('utf8')
            at_row += 1
            sub_weights = [float(w) for w in line.strip().split(',')]
            nb_elems = len(sub_weights)
            assert nb_elems == orig_shape[1]*orig_shape[2]*orig_shape[3]

            if b_inverse:
                for at_channel in range(orig_shape[2]):
                    pos_start_channel = at_channel*(orig_shape[1]*orig_shape[3])
                    for at_filter in range(orig_shape[3]):
                        pos_start = pos_start_channel + at_filter*orig_shape[1]
                        pos_end = pos_start + orig_shape[1]
                        weights[at_row, :, at_channel, at_filter] = torch.as_tensor(sub_weights[pos_start:pos_end])
            else:
                for at_filter in range(orig_shape[3]):
                    pos_start_filter = at_filter*(orig_shape[1]*orig_shape[2])
                    for at_channel in range(orig_shape[2]):
                        pos_start = pos_start_filter + at_channel*orig_shape[1]
                        pos_end = pos_start + orig_shape[1]
                        weights[at_row, :, at_channel, at_filter] = torch.as_tensor(sub_weights[pos_start:pos_end])

    return weights


if __name__ == '__main__':
    in_dir = os.path.join('..', 'data')

    # These are the shapes of the weight matrices in MatLab.
    orig_w_shapes = {'conv1': (11, 11, 3, 96),
                     'conv2': (5, 5, 256, 48),
                     'conv3': (3, 3, 384, 256),
                     'conv4': (3, 3, 384, 192),
                     'conv5': (3, 3, 256, 192)}

    # We create a new AlexNet instance, and will gradually fill in the correct layer bias and weight parameters.
    emonet = AlexNetBigEmoNet()
    # Process the convolution layers.
    for i in range(1, 6):
        print(f"Conv{i}")
        conv_b = load_conv_bias(os.path.join(in_dir, f'conv{i}.bias.txt.bz2'))
        getattr(emonet, f'conv{i}').bias = torch.nn.Parameter(conv_b)
        # The weights for the first convolution layer need to be loaded differently than the others.
        if i == 1:
            conv_w = load_conv_weights(os.path.join(in_dir, f'conv{i}.weights.txt.bz2'), orig_shape=orig_w_shapes[f'conv{i}'],
                                       b_inverse=False)
            # The original MatLab matrix needs to be permuted to PyTorch dimension ordering!
            getattr(emonet, f'conv{i}').weight = torch.nn.Parameter(conv_w.permute((3, 2, 0, 1)))
        else:
            conv_w = load_conv_weights(os.path.join(in_dir, f'conv{i}.weights.txt.bz2'), orig_shape=orig_w_shapes[f'conv{i}'],
                                       b_inverse=True)
            # The original MatLab matrix needs to be permuted to PyTorch dimension ordering!
            getattr(emonet, f'conv{i}').weight = torch.nn.Parameter(conv_w.permute((2, 3, 0, 1)))

    # Process the linear layers.
    for i in range(1, 4):
        print(f"Linear{i}")
        fc_b = load_lin_bias(os.path.join(in_dir, f'fc{i}.bias.txt.bz2'))
        fc_w = load_lin_weights(os.path.join(in_dir, f'fc{i}.weights.txt.bz2'))
        getattr(emonet, f'fc{i}').bias = torch.nn.Parameter(fc_b)
        getattr(emonet, f'fc{i}').weight = torch.nn.Parameter(fc_w)

    # Check the network doesn't crash, i.e., all dimensionalities are fine.
    # Of course, a stronger check to see that the output of this model for a given image
    # is equal to the output of the original MatLab implementation still needs to be done
    # separately. For that, we created 'test_integrity.py'.
    emonet.eval()
    x = torch.rand((1, 3, 227, 227))
    emonet(x)

    # Save the loaded weights as a model state_dict, so they can easily be reloaded afterwards.
    torch.save(emonet.state_dict(), os.path.join(in_dir, 'emonet_py.pth'))
