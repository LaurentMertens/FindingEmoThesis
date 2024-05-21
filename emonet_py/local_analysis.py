"""
A test script to create a pipeline that first uses YoLo v3 + OpenImages to detect faces

.. codeauthor:: Laurent Mertens <laurent.mertens@kuleuven.be>
"""
import os

import brambox as bb
import lightnet as ln
import numpy as np
import torch
from PIL import Image
from lightnet.models import YoloV3
from torchvision import transforms
import matplotlib.pyplot as plt
import cv2

from config import Config
from model_processor.preprocessing.img_resize_preproc import ImgResize
from pytorch_grad_cam.utils.image import show_cam_on_image


class LocalAnalysis:
    def __init__(self, device=torch.device('cpu')):
        self.device = device

        # Load names of OpenImage classes
        class_map = []
        with open(Config.FILE_OPENIMAGES_CLASSES, "r") as fin:
            for line in fin:
                line = line.strip()
                class_map.append(line)

        # Load YoloV3 model + OpenImage weights
        self.model = YoloV3(601)
        self.model.load(os.path.join(os.path.expanduser('~'), 'Work', 'Projects',
                                     'NeuroANN', 'Data', 'PreTrainedModels', 'DarkNet', 'yolov3-openimages.weights'))
        self.model.eval()
        self.model.to(device)

        thresh = 0.005

        # Create post-processing pipeline
        self.post = ln.data.transform.Compose([
            # GetBoxes transformation generates bounding boxes from network output
            ln.data.transform.GetMultiScaleAnchorBoxes(
                conf_thresh=thresh,
                network_stride=self.model.stride,
                anchors=self.model.anchors
            ),

            # Filter transformation to filter the output boxes
            ln.data.transform.NMS(
                iou_thresh=thresh
            ),

            # Miscelaneous transformation that transforms the output boxes to a brambox dataframe
            ln.data.transform.TensorToBrambox(
                class_label_map=class_map,
            )
        ])

        img_resize = ImgResize(width=608, height=608)
        self.transform = transforms.Compose([img_resize])

    def confidence_cutoff(self, df, threshold):
        df.loc[df['confidence'] < threshold, 'importance'] = 0
        return df

    def add_importance(self, df, heatmap):
        importance = []
        for index, row in df.iterrows():
            x_min = int(row["x_top_left"])
            x_max = int(row["x_top_left"] + row["width"])
            y_min = int(row["y_top_left"])
            y_max = int(row["y_top_left"] + row["height"])
            # region inside the bounding box
            bounded_region = heatmap[y_min:y_max, x_min:x_max]
            # define importance as the average activation inside that region
            importance.append(np.mean(bounded_region))
        df["object_importance"] = importance
        return df

    def local_analysis(self, file_path, file_name, show_output=False):
        """
        Perform local analysis on single image.

        :param file_path: full path to the image to process
        :param file_name: filename of the image to process
        :param show_output:
        :param cam_output: numpy array containing the CAM output
        """
        img_path = os.path.join(file_path)

        # load the corresponding grad-cam heatmap
        file_cam = "cam_grad_dataset/cam_grad_"+file_name+".npy"
        if not os.path.exists(file_cam):
            raise FileNotFoundError(f"Could not find CAM output for image {file_name}. "
                                    f"Did you perform the necessary processing beforehand?")
        cam_output = np.load(file_cam)
        os.remove(file_cam)
        print(f"DELETED FILE {file_cam}")

        with torch.no_grad():
            with open(img_path, 'rb') as f:
                img = Image.open(f).convert('RGB')
            img_tensor = self.transform(img)

            # resize heatmap to input image
            original_size_img = img.size
            grayscale_cam = cv2.resize(cam_output, original_size_img)
            grayscale_cam_pil = Image.fromarray(grayscale_cam)
            grayscale_cam_tensor = self.transform(grayscale_cam_pil)
            grayscale_cam_scaled = grayscale_cam_tensor.numpy()[0, :]

            # get output of Yolo
            output_tensor = self.model(img_tensor.unsqueeze(0).to(self.device))

            # post-processing
            output_df = self.post(output_tensor)
            proc_img = img_tensor.cpu().numpy().transpose(1, 2, 0)

            # superimpose image and gradcam heatmap
            cam = show_cam_on_image(proc_img, grayscale_cam_scaled, use_rgb=True)
            pil_img = Image.fromarray(cam)

            # add importance of bounding boxes
            df_complete = self.add_importance(output_df, grayscale_cam_scaled)

            # rename 'class_label' to 'detected_object' for more clarity later
            df_complete_return = df_complete.rename(columns={'class_label': 'detected_object',
                                                             'confidence': 'object_confidence'}).sort_values(by="object_importance", ascending=False)
            df_sorted = df_complete.sort_values(by="object_importance", ascending=False)

            if show_output:
                df_sorted = df_sorted.head(6)
                print(df_sorted)
                bb.util.draw_boxes(pil_img, df_sorted, label=df_sorted.class_label).show()
                plt.show()

        return df_complete_return
