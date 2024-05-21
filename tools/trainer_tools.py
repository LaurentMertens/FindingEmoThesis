"""
Class containing tools that are potentially useful for (initializing the) training (of) networks.

.. codeauthor:: Laurent Mertens <laurent.mertens@kuleuven.be>
"""
from collections import Counter, defaultdict

import numpy as np
import torch
import torchvision
from sklearn.metrics import average_precision_score

from tools.logger import Logger


class Depth:
    """
    Depth options when freezing classifier layer for VGG or AlexNet.
    """
    DEPTH_1 = 1
    DEPTH_2 = 2
    DEPTH_3 = 3


class TrainerTools:
    @staticmethod
    def change_last_layer(model, nb_outputs, device: torch.device, logger: Logger = None):
        """
        Use this method to change the last layer of a pretrained ImageNet network with a linear layer with nb_outputs nodes.
        Changes are applied in place.

        :param model: the model to alter.
        :param nb_outputs: the number of nodes in the new layer.
        :param device: the device the layer should live on.
        :param logger: Logger instance to be used to print messages; if None, revert to standard print().
        :return:
        """
        if logger is None:
            logger = Logger(logfile=None)

        logger.print(f"Replacing last layer of model {model.__class__.__name__}...")
        # Change the number of output classes, i.e., the last network layer
        if model.__class__.__name__ == 'SqueezeNet':
            num_features = model.classifier[1].in_channels
            features = list(model.classifier)[:-3]  # Remove last 3 layers
            features.extend([torch.nn.Conv2d(num_features, nb_outputs, kernel_size=(1, 1))])  # Add
            features.extend([torch.nn.ReLU(inplace=True)])  # Add
            features.extend([torch.nn.AdaptiveAvgPool2d(output_size=(1, 1))])  # Add
            model.classifier = torch.nn.Sequential(*features).to(device)  # Replace the model classifier
        elif model.__class__.__name__ in {'ResNet', 'GoogLeNet', 'ShuffleNetV2'}:
            num_features = model.fc.in_features
            model.fc = torch.nn.Linear(num_features, nb_outputs).to(device)  # Resnet18
            # elif net_config.model_name == 'resnet50' or net_config.model_name == 'resnext50_32x4d' \
            #         or net_config.model_name == 'wide_resnet50_2' \
            #         or net_config.model_name == 'wide_resnet50_2':
            #     model.fc = torch.nn.Linear(2048, net_config.nb_outputs).to(device) # ResNet50
        elif model.__class__.__name__ == 'Inception3':
            num_features = model.fc.in_features
            model.fc = torch.nn.Linear(num_features, nb_outputs).to(device)
            aux_features = model.AuxLogits.fc.in_features
            model.AuxLogits.fc = torch.nn.Linear(aux_features, nb_outputs).to(device)
            # elif net_config.model_name == 'googlenet' or net_config.model_name == 'shufflenet_v2_x1_0':
            #     model.fc = torch.nn.Linear(1024, net_config.nb_outputs).to(device) # GoogLeNet
        elif model.__class__.__name__ in {'MobileNetV2', 'EfficientNet', 'MNASNet'}:
            num_features = model.classifier[1].in_features
            model.classifier[1] = torch.nn.Linear(num_features, nb_outputs).to(device)  # MobileNet V2
        elif model.__class__.__name__ == 'MobileNetV3':
            num_features = model.classifier[3].in_features
            model.classifier[3] = torch.nn.Linear(num_features, nb_outputs).to(
                device)  # MobileNet V3 Large
        elif model.__class__.__name__ == 'DenseNet':
            num_features = model.classifier.in_features
            model.classifier = torch.nn.Linear(num_features, nb_outputs).to(device)  # DenseNet-121
            # elif net_config.model_name == 'densenet161':
            #     model.classifier = torch.nn.Linear(2208, net_config.nb_outputs).to(device) # DenseNet-161
        elif model.__class__.__name__ in {'AlexNet', 'VGG'}:
            num_features = model.classifier[6].in_features
            model.classifier[6] = torch.nn.Linear(num_features, nb_outputs).to(
                device)
        else:
            msg = f"Don't know what to do with model of class: {model.__class__.__name__}."
            logger.print('ValueError: ' + msg)
            raise ValueError(msg)

    @staticmethod
    def freeze_layers(model, logger: Logger = None, b_freeze_classifier=False, depth: int = Depth.DEPTH_3):
        """
        Use this method to freeze the layers of pretrained networks.

        :param model: the model whose layers need to be frozen.
        :param logger: Logger instance to be used to print messages; if None, revert to standard print().
        :param b_freeze_classifier: by default, for SqueezeNet, AlexNet and VGG, only the Conv-layers are frozen, not the linear layers\
        that constitute the "classifier" following the "features" (i.e., conv) part. If this boolean is set to true, also the\
        classifier layers will be frozen.
        :param depth: for VGG and Alexnet, number of linear layers to freeze in case b_freeze_classifier = True. Min = 1, max = 3.
        :return:
        """
        if logger is None:
            logger = Logger(logfile=None)

        # AlexNet, VGG, SqueezeNet: only disable CNN part, not the classifier
        if model.__module__ in {'torchvision.models.squeezenet', 'torchvision.models.vgg', 'torchvision.models.alexnet'}:
            logger.print('Freeze convolution layers...')
            for param in model.features.parameters():
                param.requires_grad = False
            if b_freeze_classifier:
                logger.print('Also freezing classifier layers...')
                if model.__module__ == 'torchvision.models.vgg':
                    logger.print(f'\tVGG: Freezing {depth} layers...')
                    if depth > 0:
                        TrainerTools._freeze_layer(model.classifier[0])
                    if depth > 1:
                        TrainerTools._freeze_layer(model.classifier[3])
                    if depth > 2:
                        TrainerTools._freeze_layer(model.classifier[6])
                elif model.__module__ == 'torchvision.models.alexnet':
                    logger.print(f'\tAlexNet: Freezing {depth} layers...')
                    if depth > 0:
                        TrainerTools._freeze_layer(model.classifier[1])
                    if depth > 1:
                        TrainerTools._freeze_layer(model.classifier[4])
                    if depth > 2:
                        TrainerTools._freeze_layer(model.classifier[6])
                else:
                    for param in model.classifier.parameters():
                        param.requires_grad = False
        else:
            logger.print('Freeze all layers...')
            for param in model.parameters():
                param.requires_grad = False

    @staticmethod
    def _freeze_layer(layer):
        for param in layer.parameters():
            param.requires_grad = False

    @staticmethod
    def compute_class_scores(nb_classes: int, counts_per_label: Counter, preds_per_label: Counter, correct_per_label: Counter):
        """

        :param nb_classes:
        :param counts_per_label:
        :param preds_per_label:
        :param correct_per_label:
        :return: (accuracy, weighted accuracy, macro f1, weighted f1)
        """
        nb_elems = sum(counts_per_label.values())
        prec_per_label = {i: correct_per_label[i]/preds_per_label[i] if preds_per_label[i] > 0 else
            (0 if counts_per_label[i] > 0 else 1) for i in range(nb_classes)}
        rec_per_label = {i: correct_per_label[i]/counts_per_label[i] if counts_per_label[i] > 0 else 1 for i in range(nb_classes)}
        f1_per_label = {i: 2*prec_per_label[i]*rec_per_label[i]/(prec_per_label[i]+rec_per_label[i]) if
                        prec_per_label[i]+rec_per_label[i] > 0 else 0 for i in range(nb_classes)}
        acc = sum(correct_per_label.values())/nb_elems
        weighted_acc = sum([prec_per_label[i]*counts_per_label[i] for i in range(nb_classes)])/nb_elems
        macro_f1 = sum(f1_per_label.values())/nb_classes
        weighted_f1 = sum([f1_per_label[i]*counts_per_label[i] for i in range(nb_classes)])/nb_elems

        return prec_per_label, rec_per_label, f1_per_label, acc, weighted_acc, macro_f1, weighted_f1

    @staticmethod
    def compute_multilabel_scores(predictions, targets):
        """

        :param predictions:
        :param targets:
        :return: each variable is a numpy array of shape (nb_classes,), with the following metrics:\
         tp, fp, tn, fn, prec, rec, f1, ap
        """
        nb_classes = len(targets[0])
        tp, fp = np.zeros((nb_classes,)), np.zeros((nb_classes,))
        tn, fn = np.zeros((nb_classes,)), np.zeros((nb_classes,))

        for t in zip(predictions, targets):
            pred = set(np.where(t[0] > 0.5)[0])
            ref = set(np.nonzero(t[1])[0])
            _tp = ref.intersection(pred)
            _fp = pred.difference(ref)
            _fn = ref.difference(pred)
            _tn = set(range(nb_classes)).difference(_tp.union(_fp).union(_fn))

            for e in _tp:
                tp[e] += 1
            for e in _fp:
                fp[e] += 1
            for e in _tn:
                tn[e] += 1
            for e in _fn:
                fn[e] += 1

            # Sanity check
            if False:  # Has been checked to work
                sanity_check = Counter(_tp)
                sanity_check.update(_fp)
                sanity_check.update(_tn)
                sanity_check.update(_fn)
                for e in range(26):
                    if not sanity_check[e] == 1:
                        raise ValueError("Check your code, dude.")

            # Compute P, R, F1
            prec, rec, f1 = np.zeros((nb_classes,)), np.zeros((nb_classes,)), np.zeros((nb_classes,))
            for c in range(nb_classes):
                if (tp[c] + fn[c]) > 0:
                    rec[c] = tp[c]/(tp[c] + fn[c])
                if (tp[c] + fp[c]) > 0:
                    prec[c] = tp[c]/(tp[c] + fp[c])
                if prec[c] > 0 and rec[c] > 0:
                    f1[c] = (2*prec[c]*rec[c])/(prec[c] + rec[c])

        ap = np.zeros((nb_classes,))
        arr_preds = np.asarray(predictions)
        arr_targets = np.asarray(targets)
        for c in range(nb_classes):
            ap[c] = average_precision_score(arr_targets[:,c], arr_preds[:,c])

        return tp, fp, tn, fn, prec, rec, f1, ap

    @staticmethod
    def compute_reg_scores(preds: [], targets: []):
        """
        Collect predictions per target (even though we are looking at regression problems, the target values are
        fixed in our case), and compute avg/std for predictions per target.

        Apply reverse normalize transform along the way.

        :param preds:
        :param targets:
        :return: (avg_per_target, std_per_target)
        """
        per_target = defaultdict(list)
        for (p, t) in zip(preds, targets):
            per_target[t].append(p)
        avg_per_target = {t2: np.average(p2) for t2, p2 in per_target.items()}
        std_per_target = {t2: np.std(p2) for t2, p2 in per_target.items()}

        return avg_per_target, std_per_target

    @staticmethod
    def inverse_reg(val, avg, std):
        return (val*std)+avg


if __name__ == '__main__':
    m = torchvision.models.mobilenet_v3_large()
    print(m.__class__)
    print(m.__class__.__name__)