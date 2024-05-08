import os
import json

import lightnet.network.loss
import torch
import cv2
import numpy as np
import time
from matplotlib import pyplot as plt
import grad_CAM_Yolov3
from skimage import io
import brambox as bb
import lightnet as ln
import numpy as np
import torch
from PIL import Image
from lightnet.models import YoloV3
from torchvision import transforms
from img_resize_preproc import ImgResize
import demo_youssef
from matplotlib import pyplot as plt
from build_utils import img_utils
from build_utils import torch_utils
from build_utils import utils



if __name__ == '__main__':
    model = YoloV3(601)
    model.load(os.path.join('yolov3-openimages.weights'))
    model.eval()
    demo_img = os.path.join('..', 'data/images', 'friends_parc.jpg')
    img_resize = ImgResize(width=608, height=608)
    transform = transforms.Compose([img_resize])
    with torch.no_grad():
        with open(demo_img, 'rb') as f:
            img = Image.open(f).convert('RGB')
        img_tensor = transform(img)
        # Q: Actually, i'm not sure the image provided to the model needs to be resized/normalized. It seems
        # the model does this itself.
        with torch.no_grad():
            output_tensor = model(img_tensor.unsqueeze(0))
    thresh = 0.005
    target_layer = [model.neck[-1]]
    height, width = 608, 608
    final_shape =[19, 19]
    ori_shape = [height, width]
    img_final = output_tensor
    grad_cam = grad_CAM_Yolov3.GradCAMYolo(model, img_tensor.unsqueeze(0), output_tensor, target_layer, ori_shape, final_shape)
    mask, box, class_id = grad_cam(output_tensor)  # cam mask
