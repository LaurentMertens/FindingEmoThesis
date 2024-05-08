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

from img_resize_preproc import ImgResize
from explanations_emonet import get_visualizations, plot_cam
from pytorch_grad_cam.utils.image import scale_cam_image, scale_accross_batch_and_channels, show_cam_on_image

class LocalAnalysis:
    def __init__(self):
        # Load names of OpenImage classes
        class_map = []
        with open("openimages.names", "r") as fin:
            for line in fin:
                line = line.strip()
                class_map.append(line)

        # Load YoloV3 model + OpenImage weights
        self.model = YoloV3(601)
        self.model.load(os.path.join(os.path.expanduser('~'),
                                     "Work", "Projects", "NeuroANN",
                                     "Data", "PreTrainedModels", "DarkNet", 'yolov3-openimages.weights'))
        self.model.eval()

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
        #plt.imshow(heatmap)
        #plt.show()
        for index, row in df.iterrows():
            x_min = int(row["x_top_left"])
            x_max = int(row["x_top_left"] + row["width"])
            y_min = int(row["y_top_left"])
            y_max = int(row["y_top_left"] + row["height"])
            bounded_region = heatmap[y_min:y_max, x_min:x_max]
                #if bounded_region.size == 0:
            #print(index)
            #print("x_min :", x_min, " x_max :", x_max)
            #print("y_min :", x_min, " y_max :", x_max)
            #print("Importance: ", np.mean(bounded_region))
            importance.append(np.mean(bounded_region))
        df["importance"] = importance
        return df

    def local_analysis(self, file_path, file_name, show_box=False):
        """
        Perform local analysis on single image.
        """
        img_path = os.path.join(file_path)
        grayscale_cam = np.load("cam_grad_dataset/cam_grad_"+file_name+".npy")
        with torch.no_grad():
            with open(img_path, 'rb') as f:
                img = Image.open(f).convert('RGB')
            img_tensor = self.transform(img)
            original_size_img = img.size
            grayscale_cam = cv2.resize(grayscale_cam, original_size_img)
            grayscale_cam_pil = Image.fromarray(grayscale_cam)
            grayscale_cam_tensor = self.transform(grayscale_cam_pil)
            grayscale_cam_scaled = grayscale_cam_tensor.numpy()[0, :]

            output_tensor = self.model(img_tensor.unsqueeze(0))
            output_df = self.post(output_tensor)
            proc_img = img_tensor.cpu().numpy().transpose(1, 2, 0)
            cam = show_cam_on_image(proc_img, grayscale_cam_scaled, use_rgb=True)
            #fig, ax = plt.subplots(1, 2)
            #ax[0].imshow(proc_img)
            #ax[1].imshow(grayscale_cam_scaled)
            #plt.show()
            pil_img = Image.fromarray(cam)
            df_complete = self.add_importance(output_df, grayscale_cam_scaled)
            df_complete = self.confidence_cutoff(df_complete, threshold=0.1)
            df_sorted = df_complete.sort_values(by="importance", ascending=False)
            df_important = df_sorted.head(5)        # check confidence as additional parameter to take into account
            if show_box:
                bb.util.draw_boxes(pil_img, df_important, label=df_important.class_label).show()

        return df_important