"""
Load an ImageNet network, using the pre-trained weights for all layers but the last, and using custom trained
weights for the last layer.

.. codeauthor:: Laurent Mertens <laurent.mertens@kuleuven.be>
"""
import os

import dill
import torch
import torchvision

from tools.trainer_tools import TrainerTools


class LoadBufferedImageNet:
    @staticmethod
    def load(model, weights_file: str, preproc_file: str=None, device=torch.device('cpu')):
        """

        :param model: UNINITIALIZED ImageNet model, e.g., torchvision.models.vgg16
        :param weights_file: weights file containing the weights to be used for the final layer
        :param preproc_file: file containing the preprocessing chain to be used with the model
        :param device:
        :return: (model,preprocessor) tuple; model = instance with initialized weights/parameters,\
         preprocessor = image preprocessor used for buffering the model features
        """
        # Get model class
        model_name = model.__name__

        # Initialize model with default weights
        model = model(weights=torchvision.models.get_model_weights(model_name).DEFAULT).to(device)

        saved_weights = torch.load(weights_file, map_location=device)
        if model_name.startswith('vgg') or model_name.startswith('alex'):
            # Get number of outputs
            try:
                nb_outs = saved_weights['classifier.1.bias'].shape[0]
            except KeyError:
                nb_outs = saved_weights['module.classifier.1.bias'].shape[0]
            # Replace last layer
            TrainerTools.change_last_layer(model, nb_outputs=nb_outs, device=device)
            # Set weights on last layer
            try:
                model.classifier[6].weight = torch.nn.Parameter(saved_weights['classifier.1.weight'].to(device))
                model.classifier[6].bias = torch.nn.Parameter(saved_weights['classifier.1.bias'].to(device))
            except KeyError:
                model.classifier[6].weight = torch.nn.Parameter(saved_weights['module.classifier.1.weight'].to(device))
                model.classifier[6].bias = torch.nn.Parameter(saved_weights['module.classifier.1.bias'].to(device))
        elif model_name.startswith('resnet'):
            # Get number of outputs
            prefix = ''  # When the model has been trained using DataParallel, the dict keys should be prefixed by 'module.'
            try:
                nb_outs = saved_weights['linear.bias'].shape[0]
            except KeyError:
                nb_outs = saved_weights['module.linear.bias'].shape[0]
                prefix = 'module.'
            # Replace last layer
            TrainerTools.change_last_layer(model, nb_outputs=nb_outs, device=device)
            # Set weights on last layer
            model.fc.weight = torch.nn.Parameter(saved_weights[prefix + 'linear.weight'].to(device))
            model.fc.bias = torch.nn.Parameter(saved_weights[prefix + 'linear.bias'].to(device))
        elif model_name in {'googlenet', 'inception_v3'}:
            # Get number of outputs
            nb_outs = saved_weights['classifier.1.bias'].shape[0]
            # Replace last layer
            TrainerTools.change_last_layer(model, nb_outputs=nb_outs, device=device)
            # Set weights on last layer
            model.fc.weight = torch.nn.Parameter(saved_weights['classifier.1.weight'].to(device))
            model.fc.bias = torch.nn.Parameter(saved_weights['classifier.1.bias'].to(device))
        #     return ImageNetClassifierInceptionV3
        elif model_name.startswith('densenet'):
            # Get number of outputs
            nb_outs = saved_weights['linear.bias'].shape[0]
            # Replace last layer
            TrainerTools.change_last_layer(model, nb_outputs=nb_outs, device=device)
            # Set weights on last layer
            model.classifier.weight = torch.nn.Parameter(saved_weights['linear.weight'].to(device))
            model.classifier.bias = torch.nn.Parameter(saved_weights['linear.bias'].to(device))
        # elif model_name.startswith('squeezenet'):
        #     return ImageNetClassifierSqueezeNet
        else:
            raise ValueError(f"Don't know what to do with model of type '{model_name}'.")

        pp = None
        if preproc_file is not None:
            pp = dill.load(open(preproc_file, 'rb'))

        return model, pp


if __name__ == '__main__':
    saved_weights = os.path.join(os.path.expanduser('~'),
                                 'Work', 'Projects', 'NeuroANN', 'Data', 'Models',
                                 'BestModels', '20231222',
                                 'emo8_vgg16_lr=0.001_loss=UnbalancedCrossEntropyLoss_sc=test_cuda.pth')

    # saved_weights = os.path.join(Config.DIR_MODELS, 'Emo8', 'emo8_resnet50^_lr=0.001_loss=CrossEntropyLoss_sc=test_cuda.pth')
    # saved_weights = os.path.join(Config.DIR_MODELS, 'Emo8', 'emo8_densenet121^_lr=0.001_loss=CrossEntropyLoss_sc=test_cuda.pth')
    # saved_weights = os.path.join(Config.DIR_MODELS, 'Emo8', 'emo8_inception_v3^_lr=0.001_loss=CrossEntropyLoss_sc=test_cuda.pth')
    model, pp = LoadBufferedImageNet.load(torchvision.models.vgg16, weights_file=saved_weights, device=torch.device('cuda'))

    model(torch.rand((1, 3, 800, 600)).to(torch.device('cuda')))
