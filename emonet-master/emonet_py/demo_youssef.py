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

from img_resize_preproc import ImgResize
from explanations_emonet import get_visualizations, plot_cam



if __name__ == '__main__':
    # Load names of OpenImage classes
    class_map = []
    with open("openimages.names", "r") as fin:
        for line in fin:
            line = line.strip()
            class_map.append(line)

    # Load YoloV3 model + OpenImage weights
    model = YoloV3(601)
    model.load(os.path.join('yolov3-openimages.weights'))
    model.eval()

    thresh = 0.005

    # Create post-processing pipeline
    post = ln.data.transform.Compose([
        # GetBoxes transformation generates bounding boxes from network output
        ln.data.transform.GetMultiScaleAnchorBoxes(
            conf_thresh=thresh,
            network_stride=model.stride,
            anchors=model.anchors
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
    transform = transforms.Compose([img_resize])
    demo_img = os.path.join('..', 'data/images', 'friends_parc.jpg')
    heatmap = np.load("heatmap.npy")
    with torch.no_grad():
        with open(demo_img, 'rb') as f:
            img = Image.open(f).convert('RGB')
        img_tensor = transform(img)
        with torch.no_grad():
            output_tensor = model(img_tensor.unsqueeze(0))
            output_df = post(output_tensor)
            #output_df = output_df[output_df["class_label"] == 'Human face']
        if not output_df.empty:
            pil_img = (255*img_tensor.cpu().numpy().transpose(1, 2, 0)).astype(np.uint8)
            pil_img = Image.fromarray(pil_img)
            bb_img = bb.util.draw_boxes(pil_img, output_df, label=output_df.class_label).show()