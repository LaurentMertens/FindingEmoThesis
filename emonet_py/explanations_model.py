"""
Given a PyTorch model, find the explanations for this particular model.
Hopefully, one day, I'll find the tie to write something more explicative.

.. codeauthor:: Laurent Mertens <laurent.mertens@kuleuven.be>
"""
from typing import Iterable

import numpy as np
import torch

from emonet_py.visualizations.file_processor import FileProcessor, Processors
from model_processor.preprocessing.img_resize_preproc import ImgResize
from tools.img_tools import ImageTools
from tools.model_tools import ModelTools
from torchvision import transforms


class ExplanationsModel:
    def __init__(self, model, model_pp, target_layers: Iterable, class_labels=Iterable[str], device=torch.device('cpu')):
        """

        :param model: INSTANTIATED PyTorch model
        :param model_pp: model preprocessing chain
        :param target_layers: list containing the target layers for the CAM algorithm
        :param class_labels: the class labels corresponding to the model outputs
        :param device: what device to use
        """
        self.device = device
        self.model = model
        self.model.to(device)
        self.model_pp = model_pp
        self.class_labels = class_labels
        self.file_processor = FileProcessor(model=self.model,
                                            target_layers=target_layers,
                                            methods=[Processors.EIGENCAM])
        self.img_resize = ImgResize(width=800, height=600)

    def get_explanations_for_image(self, img_path, file_name, show_plot=False):
        # Instantiations & definitions
        img_loaded = ImageTools.open_image(img_path)
        img_resized = transforms.functional.to_pil_image(self.img_resize(img_loaded))
        img_tensor = self.model_pp(img_loaded)
        img_size = img_resized.size
        img_tensor = img_tensor.unsqueeze(0)

        # We don't need the gradients
        with torch.no_grad():
            img_tensor = img_tensor.to(self.device)
            pred = self.model(img_tensor)[0]
            if torch.sum(pred) != 1.:
                pred = torch.nn.functional.softmax(pred, dim=0)

        # Convert PIL image to standardized numpy array
        proc_img = np.float32(img_resized)/255
        # Model
        max_prob, max_class, class_index = ModelTools.get_most_probable_class(pred, self.class_labels)

        # Visualization
        vis, cam_output = self.file_processor.get_visualizations(image=proc_img, input_tensor=img_tensor,
                                                                 class_index=class_index, img_size=img_size,
                                                                 file_name=file_name, targets=None)
        # if show_plot:
        #     plot_cam(visualization=vis, image=img_loaded, class_label=max_class, prob_label=max_prob, val=valence, aro=arousal)

        return max_class, max_prob, pred, cam_output
