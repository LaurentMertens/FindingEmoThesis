"""
Network to be used with pre-computed ImageNet netwerk features.

These networks represent the 'classifier' part of the ImageNet networks, i.e., the last layer(s).

.. codeauthor:: Laurent Mertens <laurent.mertens@kuleuven.be>
"""
import torch
import torchvision.models.vgg

from networks.nn.clip.clip_models import CLIPRegressor1, CLIPClassifier1
from networks.nn.dinov2.dinov2_models import DINOv2Classifier1, DINOv2Regressor1
from networks.nn.emonet.emonet_models import EmoNetRegressor1, EmoNetClassifier1
from networks.nn.regularization.targeted_dropout import TargetedLinDropout


def get_classifier_for_imagenet(model_name: str):
    print(f"Fetching model for model_name: {model_name}")
    if model_name.endswith('_apbw'):
        model_name = model_name[:-5]

    if model_name.endswith('-emo8'):
        return ClassifierEmo8
    elif model_name.startswith('vgg') or model_name.startswith('alexnet') or model_name.startswith('places365-feats_alexnet'):
        return ImageNetRegressorVGG if model_name.endswith('_reg') else ImageNetClassifierVGG
    elif (model_name.startswith('resnet18') or model_name.startswith('resnet34') or
          model_name.startswith('places365-feats_resnet18')):
        return ImageNetRegressorResNetVar1 if model_name.endswith('_reg') else ImageNetClassifierResNetVar1
    elif (model_name.startswith('resnet50') or model_name.startswith('resnet101') or
          model_name.startswith('places365-feats_resnet50')):
        return ImageNetRegressorResNetVar2 if model_name.endswith('_reg') else ImageNetClassifierResNetVar2
    elif model_name.startswith('googlenet'):
        return ImageNetRegressorGoogLeNet if model_name.endswith('_reg') else ImageNetClassifierGoogLeNet
    elif model_name.startswith('inception_v3'):
        return ImageNetRegressorInceptionV3 if model_name.endswith('_reg') else ImageNetClassifierInceptionV3
    elif model_name.startswith('densenet121'):
        return ImageNetRegressorDenseNet121 if model_name.endswith('_reg') else ImageNetClassifierDenseNet121
    elif model_name.startswith('densenet161') or model_name.startswith('places365-feats_densenet161'):
        return ImageNetRegressorDenseNet161 if model_name.endswith('_reg') else ImageNetClassifierDenseNet161
    elif model_name.startswith('squeezenet'):
        return None if model_name.endswith('_reg') else ImageNetClassifierSqueezeNet
    elif model_name.startswith('DINOv2'):
        return DINOv2Regressor1 if model_name.endswith('_reg') else DINOv2Classifier1
    elif model_name.startswith('CLIP'):
        return CLIPRegressor1 if model_name.endswith('_reg') else CLIPClassifier1
    elif model_name.startswith('emonet'):
        return EmoNetRegressor1 if model_name.endswith('_reg') else EmoNetClassifier1
    else:
        raise ValueError(f"Don't know what to do with model of type '{model_name}'.")


class ImageNetNbConvolutionFeats:
    """
    Class to contain the number of convolutional features that each ImageNetClassifier type produces.
    E.g., in case of AlexNet, this is the size of the flattened output of the last conv layer.
    """
    ALEXNET = 9216
    VGG = 25088
    RESNET_VAR1 = 512
    RESNET_VAR2 = 2048
    GOOGLENET = 1024
    INCEPTION_V3 = 2048
    DENSENET121 = 1024
    DENSENET161 = 2208

    @classmethod
    def get_nb_infeats_for_model(cls, model_name: str):
        if model_name.startswith('vgg') or model_name.startswith('alexnet'):
            return cls.VGG
        elif model_name in {'resnet18', 'resnet34'}:
            return cls.RESNET_VAR1
        elif model_name in {'resnet50', 'resnet101', 'resnet50_sin'}:
            return cls.RESNET_VAR2
        elif model_name == 'googlenet':
            return cls.GOOGLENET
        elif model_name == 'inception_v3':
            return cls.INCEPTION_V3
        elif model_name == 'densenet121':
            return cls.DENSENET121
        elif model_name == 'densenet161':
            return cls.DENSENET161
        else:
            raise ValueError(f"Don't know what to do with model name: {model_name}")


class ImageNetClassifierNbInFeats:
    """
    Class to contain the number of input features that each ImageNetClassifier type requires.
    """
    VGG = 4096
    RESNET_VAR1 = 512
    RESNET_VAR2 = 2048
    GOOGLENET = 1024
    INCEPTION_V3 = 2048
    DENSENET121 = 1024
    DENSENET161 = 2208

    @classmethod
    def get_nb_infeats_for_model(cls, model_name: str):
        if model_name.startswith('vgg') or model_name.startswith('alexnet') or model_name.startswith('places365-feats_alexnet'):
            return cls.VGG
        elif model_name in {'resnet18', 'resnet18_cc', 'resnet34', 'places365-feats_resnet18'}:
            return cls.RESNET_VAR1
        elif model_name in {'resnet50', 'resnet101', 'resnet50_sin'}:
            return cls.RESNET_VAR2
        elif model_name == 'googlenet':
            return cls.GOOGLENET
        elif model_name == 'inception_v3':
            return cls.INCEPTION_V3
        elif model_name == 'densenet121':
            return cls.DENSENET121
        elif model_name == 'densenet161':
            return cls.DENSENET161
        else:
            raise ValueError(f"Don't know what to do with model name: {model_name}")


# ############################## VGG ##############################
class ImageNetClassifierVGG(torch.nn.Module):
    def __init__(self, nb_outputs: int, dropout=0.5, b_full=False, b_classifier_l1=False):
        super().__init__()
        if b_full:
            self.classifier = torch.nn.Sequential(
                torch.nn.Linear(in_features=25088, out_features=4096, bias=True),
                torch.nn.ReLU(inplace=True),
                torch.nn.Dropout(p=dropout, inplace=False),
                torch.nn.Linear(in_features=4096, out_features=4096, bias=True),
                torch.nn.ReLU(inplace=True),
                # torch.nn.Dropout(p=dropout, inplace=False),
                torch.nn.Linear(in_features=4096, out_features=nb_outputs, bias=True),
                torch.nn.Sigmoid()
                # torch.nn.Hardtanh(min_val=0.0, max_val=1.0)
            )

            # Load pretrained and copy weights for classifier layer 2
            pretrained_model = torchvision.models.vgg.vgg16()
            self.classifier[0].weight = pretrained_model.classifier[0].weight
            self.classifier[3].weight = pretrained_model.classifier[3].weight
            del pretrained_model
        elif b_classifier_l1:
            self.classifier = torch.nn.Sequential(
                torch.nn.Dropout(p=dropout, inplace=False),
                torch.nn.Linear(in_features=4096, out_features=4096, bias=True),
                torch.nn.ReLU(inplace=True),
                torch.nn.Dropout(p=dropout, inplace=False),
                torch.nn.Linear(in_features=4096, out_features=nb_outputs, bias=True)
            )

            # Load pretrained and copy weights for classifier layer 2
            pretrained_model = torchvision.models.vgg.vgg16()
            self.classifier[1].weight = pretrained_model.classifier[3].weight
            del pretrained_model
        else:
            self.classifier = torch.nn.Sequential(
                # TargetedLinDropout(top_n=512, bottom_n=512),
                torch.nn.Dropout(p=dropout, inplace=False),
                torch.nn.Linear(in_features=4096, out_features=nb_outputs, bias=True)
            )

    def forward(self, x):
        return self.classifier(x)


class ImageNetRegressorVGG(torch.nn.Module):
    def __init__(self, dropout=0.5, b_classifier_l1=False):
        super().__init__()
        if b_classifier_l1:
            print(f"USING ImageNetRegressorVGGReg WITH b_classifier_l1={b_classifier_l1}")
            self.regressor = torch.nn.Sequential(
                torch.nn.Dropout(p=dropout, inplace=False),
                torch.nn.Linear(in_features=4096, out_features=4096, bias=True),
                torch.nn.Dropout(p=dropout, inplace=False),
                torch.nn.Linear(in_features=4096, out_features=1, bias=True),
                torch.nn.Sigmoid()
            )

            # Load pretrained and copy weights for classifier layer 2
            pretrained_model = torchvision.models.vgg.vgg16()
            self.regressor[1].weight = pretrained_model.classifier[3].weight
            del pretrained_model
        else:
            self.regressor = torch.nn.Sequential(
                torch.nn.Dropout(p=dropout, inplace=False),
                torch.nn.Linear(in_features=4096, out_features=1, bias=True),
                torch.nn.Sigmoid()
            )

    def forward(self, x):
        return self.regressor(x)


# ############################## SqueezeNet ##############################
class ImageNetClassifierSqueezeNet(torch.nn.Module):
    def __init__(self, nb_outputs: int, dropout=0.5):
        super().__init__()
        self.classifier = torch.nn.Sequential(
            torch.nn.Dropout(p=dropout, inplace=False),
            torch.nn.Conv2d(512, nb_outputs, kernel_size=(1, 1), stride=(1, 1)),
            torch.nn.ReLU(inplace=True),
            torch.nn.AdaptiveAvgPool2d(output_size=(1, 1))
        )

    def forward(self, x):
        return self.classifier(x)


# ############################## ResNet ##############################
class ImageNetClassifierResNetVar1(torch.nn.Module):
    def __init__(self, nb_outputs: int, dropout=0.5, b_full=False):
        super().__init__()
        if b_full:
            self.classifier = torch.nn.Sequential(
                torch.nn.Dropout(p=dropout, inplace=False),
                torch.nn.Linear(in_features=512, out_features=nb_outputs, bias=True),
                torch.nn.Sigmoid()
            )
        else:
            self.classifier = torch.nn.Sequential(
                torch.nn.Dropout(p=dropout, inplace=False),
                torch.nn.Linear(in_features=512, out_features=nb_outputs, bias=True)
            )

    def forward(self, x):
        return self.classifier(x)


class ImageNetClassifierResNetVar2(torch.nn.Module):
    def __init__(self, nb_outputs: int, dropout=0.5):
        super().__init__()
        self.classifier = torch.nn.Sequential(
            torch.nn.Dropout(p=dropout, inplace=False),
            torch.nn.Linear(in_features=2048, out_features=nb_outputs, bias=True)
        )

    def forward(self, x):
        return self.classifier(x)


class ImageNetRegressorResNetVar1(torch.nn.Module):
    def __init__(self, dropout=0.5):
        super().__init__()
        self.regressor = torch.nn.Sequential(
            torch.nn.Dropout(p=dropout, inplace=False),
            torch.nn.Linear(in_features=512, out_features=1, bias=True),
            torch.nn.Sigmoid()
        )

    def forward(self, x):
        return self.regressor(x)


class ImageNetRegressorResNetVar2(torch.nn.Module):
    def __init__(self, dropout=0.5):
        super().__init__()
        self.regressor = torch.nn.Sequential(
            torch.nn.Dropout(p=dropout, inplace=False),
            torch.nn.Linear(in_features=2048, out_features=1, bias=True),
            torch.nn.Sigmoid()
        )

    def forward(self, x):
        return self.regressor(x)


# ############################## GoogLeNet ##############################
class ImageNetClassifierGoogLeNet(torch.nn.Module):
    def __init__(self, nb_outputs: int, dropout=0.5):
        super().__init__()
        self.classifier = torch.nn.Sequential(
            torch.nn.Dropout(p=dropout, inplace=False),
            torch.nn.Linear(in_features=1024, out_features=nb_outputs, bias=True)
        )

    def forward(self, x):
        return self.classifier(x)


class ImageNetRegressorGoogLeNet(torch.nn.Module):
    def __init__(self, dropout=0.5):
        super().__init__()
        self.regressor = torch.nn.Sequential(
            torch.nn.Dropout(p=dropout, inplace=False),
            torch.nn.Linear(in_features=1024, out_features=1, bias=True),
            torch.nn.Sigmoid()
        )

    def forward(self, x):
        return self.regressor(x)


# ############################## InceptionV3 ##############################
class ImageNetClassifierInceptionV3(torch.nn.Module):
    def __init__(self, nb_outputs: int, dropout=0.5):
        super().__init__()
        self.classifier = torch.nn.Sequential(
            torch.nn.Dropout(p=dropout, inplace=False),
            torch.nn.Linear(in_features=2048, out_features=nb_outputs, bias=True)
        )

    def forward(self, x):
        return self.classifier(x)


class ImageNetRegressorInceptionV3(torch.nn.Module):
    def __init__(self, dropout=0.5):
        super().__init__()
        self.regressor = torch.nn.Sequential(
            torch.nn.Dropout(p=dropout, inplace=False),
            torch.nn.Linear(in_features=2048, out_features=1, bias=True),
            torch.nn.Sigmoid()
        )

    def forward(self, x):
        return self.regressor(x)


# ############################## DenseNet ##############################
class ImageNetClassifierDenseNet121(torch.nn.Module):
    def __init__(self, nb_outputs: int, dropout=0.5):
        super().__init__()
        self.classifier = torch.nn.Sequential(
            torch.nn.Dropout(p=dropout, inplace=False),
            torch.nn.Linear(in_features=1024, out_features=nb_outputs, bias=True)
        )

    def forward(self, x):
        return self.classifier(x)


class ImageNetClassifierDenseNet161(torch.nn.Module):
    def __init__(self, nb_outputs: int, dropout=0.5):
        super().__init__()
        self.classifier = torch.nn.Sequential(
            torch.nn.Dropout(p=dropout, inplace=False),
            torch.nn.Linear(in_features=2208, out_features=nb_outputs, bias=True)
        )

    def forward(self, x):
        return self.classifier(x)


class ImageNetRegressorDenseNet121(torch.nn.Module):
    def __init__(self, dropout=0.5):
        super().__init__()
        self.classifier = torch.nn.Sequential(
            torch.nn.Dropout(p=dropout, inplace=False),
            torch.nn.Linear(in_features=1024, out_features=1, bias=True),
            torch.nn.Sigmoid()
        )

    def forward(self, x):
        return self.classifier(x)


class ImageNetRegressorDenseNet161(torch.nn.Module):
    def __init__(self, dropout=0.5):
        super().__init__()
        self.regressor = torch.nn.Sequential(
            torch.nn.Dropout(p=dropout, inplace=False),
            torch.nn.Linear(in_features=2208, out_features=1, bias=True),
            torch.nn.Sigmoid()
        )

    def forward(self, x):
        return self.regressor(x)


# ############################## ClassifierEmo8 ##############################
class ClassifierEmo8(torch.nn.Module):
    def __init__(self, nb_outputs: int, dropout=None):
        """

        :param nb_outputs:
        :param dropout: not used, just there for compatibility purposes.
        """
        super().__init__()
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(in_features=8, out_features=nb_outputs, bias=True)
        )

    def forward(self, x):
        return self.classifier(x)


if __name__ == '__main__':
    in_vgg = ImageNetClassifierVGG(nb_outputs=2, b_classifier_l1=True)
    t = torch.rand((5, 4096))
    print(in_vgg(t))
