"""
A class to process an image file with a/multiple particular GradCAM algorithm(s).

.. codeauthor:: Laurent Mertens <laurent.mertens@kuleuven.be>
"""
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import torch
from pytorch_grad_cam import GradCAM, GradCAMPlusPlus, ScoreCAM, AblationCAM, EigenCAM, guided_backprop
from pytorch_grad_cam.utils.image import show_cam_on_image

from emonet_py.explanations_liftcam import CAM_Explanation


class Processors:
    # BEWARE! CONSTANT FIELDS ARE DEFINED AT THE BOTTOM, SO AS TO BE ABE TO REFERENCE THESE FOLLOWING METHODS
    # They need to be defined BEFORE referencing them in the statis fields.
    @staticmethod
    def process_type_a(processor, input_tensor, targets, image, image_weight, **kwargs):
        """

        :param processor:
        :param input_tensor:
        :param targets:
        :param image:
        :param image_weight:
        :param kwargs: stuff we can ignore, but this way we can send the same parameters to all process_type methods
        :return:
        """
        grayscale = processor(input_tensor=input_tensor, targets=targets)
        grayscale = grayscale[0, :]
        vis = show_cam_on_image(image, grayscale, use_rgb=True, image_weight=image_weight)

        return grayscale, vis

    @staticmethod
    def process_type_b(processor, input_tensor, class_index, img_size, image, image_weight, **kwargs):
        """

        :param processor:
        :param input_tensor:
        :param class_index:
        :param img_size:
        :param image:
        :param image_weight:
        :param kwargs: stuff we can ignore, but this way we can send the same parameters to all process_type methods
        :return:
        """
        grayscale = processor(input_tensor, int(class_index), img_size)
        grayscale = torch.squeeze(grayscale).cpu().detach().numpy()
        vis = show_cam_on_image(image, grayscale, use_rgb=True, image_weight=image_weight)

        return grayscale, vis

    @staticmethod
    def process_type_c(processor, input_tensor, class_index, targets, image, **kwargs):
        """

        :param processor:
        :param input_tensor:
        :param class_index:
        :param targets:
        :param image:
        :param kwargs: stuff we can ignore, but this way we can send the same parameters to all process_type methods
        :return:
        """
        grayscale = processor(input_tensor=input_tensor, targets=targets)
        grayscale = grayscale[0, :]
        vis = show_cam_on_image(image, grayscale, use_rgb=True, image_weight=0.0)
        # Get guided backpropagation map
        camGuided = guided_backprop.GuidedBackpropReLUModel(processor.model, "cpu")
        grayscale_cam_Guided = camGuided(input_tensor, class_index)
        # Elementwise multiplication
        guidedmap = grayscale_cam_Guided * vis

        return grayscale, (("Guided Backprop", grayscale_cam_Guided), ("Guided Grad-Cam", guidedmap))

    # Define static fields
    GRADCAM = (GradCAM, process_type_a)
    GUIDED = (GradCAM, process_type_c)
    GRADCAMPP = (GradCAMPlusPlus, process_type_a)
    ABLATIONCAM = (AblationCAM, process_type_a)
    SCORECAM = (ScoreCAM, process_type_a)
    EIGENCAM = (EigenCAM, process_type_a)
    LIFTCAM = ((CAM_Explanation, {'method': 'LIFT-CAM'}), process_type_b)
    LRPCAM = ((CAM_Explanation, {'method': 'LRP-CAM'}), process_type_b)
#    LIMECAM = (CAM_Explanation, {'method': 'LIME-CAM'})


class FileProcessor:
    def __init__(self, model, target_layers, methods=Iterable[Processors]):
        self.processors = {}
        for method in methods:
            if isinstance(method[0], tuple):
                # method[0] = (CAM method, parameters), method[1] = process method to be used with CAM method
                proc_name, proc_params = str(method[0][0]), method[0][1]
                proc_params['model'] = model
                processor = method[0][0](**proc_params)
                self.processors[proc_name] = (processor, method[1])
            else:
                # method[0] = CAM method, method[1] = process method to be used with CAM method
                proc_name = str(method[0])
                processor = method[0](model=model, target_layers=target_layers)
                self.processors[proc_name] = (processor, method[1])

    def get_visualizations(self, image: np.ndarray,
                           input_tensor: torch.Tensor,
                           class_index: int,
                           img_size,
                           file_name: str,
                           image_weight: float = 0.5,
                           targets=None):
        """
        Get visualizations for a single image

        :param image: numpy array representation of the image to process
        :param input_tensor:
        :param class_index:
        :param img_size:
        :param file_name:
        :param image_weight:
        :param targets:
        :return: vis, grayscales
        """
        vis = []
        grayscales = []
        # Keep track of CAM output we are generating, then delete afterwards
        # saved_files = set()
        for method, processor_info in self.processors.items():
            processor, processor_method = processor_info
            params = {'processor': processor, 'input_tensor': input_tensor,
                      'targets': targets, 'image': image, 'image_weight': image_weight,
                      'file_name': file_name,
                      'class_index': class_index, 'img_size': img_size}
            grayscale, vis_cam = processor_method(**params)
            grayscales.append(grayscale)
            if len(vis_cam) == 2:  # type_c returns 2, a and b 1 CAM result(s)
                for tpl in vis_cam:
                    vis.append([tpl[0], tpl[1]])
            else:
                vis.append([method, vis_cam])

        return vis, grayscales

    def plot_cam(self, visualization, image, class_label, prob_label, val, aro):
        """
        plot the different localization maps superimposed on image with the most probable class, valence and arousal
        predicted by EmoNet
        """
        ncol = len(visualization)+1
        fig, ax = plt.subplots(1, ncol)
        ax[0].imshow(image, interpolation='none')
        ax[0].axis('off')
        ax[0].set_title("Image")
        for i in range(len(visualization)):
            ax[i+1].imshow(visualization[i][1], interpolation='none')
            ax[i+1].axis('off')
            ax[i+1].set_title(visualization[i][0])
        plt.subplots_adjust(wspace=0.1, hspace=0)
        plt.suptitle(f"Class: {class_label:}\nConfidence: {prob_label:0.2f}% \nValence: {val:.0f} \nArousal: {aro:.0f}")
        plt.show()
