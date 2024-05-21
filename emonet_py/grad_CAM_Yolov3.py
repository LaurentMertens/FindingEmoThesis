import cv2
import numpy as np
import torch
from matplotlib import pyplot as plt
from PIL import Image
import numpy as np
import torch
import ttach as tta
from typing import Callable, List, Tuple, Optional
from pytorch_grad_cam.activations_and_gradients import ActivationsAndGradients
from pytorch_grad_cam.utils.svd_on_activations import get_2d_projection
from pytorch_grad_cam.utils.image import scale_cam_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
import cv2
import random
from matplotlib import pyplot as plt
from build_utils import img_utils
from build_utils import utils
import lightnet as ln
from pytorch_grad_cam.activations_and_gradients import ActivationsAndGradients
from pytorch_grad_cam import GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image, deprocess_image, show_factorization_on_image
from explanations_emonet import adjust_cam


import torchvision
def clip_coords(boxes, shape):
    # Clip bounding xyxy bounding boxes to image shape (height, width)
    if isinstance(boxes, torch.Tensor):  # faster individually
        boxes[:, 0].clamp_(0, shape[1])  # x1
        boxes[:, 1].clamp_(0, shape[0])  # y1
        boxes[:, 2].clamp_(0, shape[1])  # x2
        boxes[:, 3].clamp_(0, shape[0])  # y2
    else:  # np.array (faster grouped)
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, shape[1])  # x1, x2
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, shape[0])  # y1, y2


def scale_coords(img1_shape, coords, img0_shape, ratio_pad=None):
    # Rescale coords (xyxy) from img1_shape to img0_shape
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    coords[:, [0, 2]] -= pad[0]  # x padding
    coords[:, [1, 3]] -= pad[1]  # y padding
    coords[:, :4] /= gain
    clip_coords(coords, img0_shape)
    return coords

class GradCAMYolo(object):
    """
    1: the network does not update gradient, input requires the update
    2: use targeted class's score to do backward propagation
    """

    def __init__(self, net, input_tensor, output_tensor, target_layers, ori_shape, final_shape, reshape_transform=None):
        self.net = net
        self.input_tensor = input_tensor
        self.target_layers = target_layers
        self.ori_shape = ori_shape
        self.final_shape = final_shape
        self.feature = None
        self.gradient = None
        self.output_tensor = output_tensor
        self.net.eval()
        self.activations_and_grads = ActivationsAndGradients(self.net, target_layers, reshape_transform)
        #self.handlers = []
        #self._register_hook()
        #self.feature_extractor = FeatureExtractor(self.net, self.layer_name)


    def _get_features_hook(self, module, input, output):
        self.feature = output
        print("feature shape:{}".format(output.size()))

    
    def _get_grads_hook(self, module, input_grad, output_grad):
        self.gradient = output_grad[0]

    
    def _register_hook(self):
        for i, module in enumerate(self.net.module_list):
            if module == self.layer_name:
                self.handlers.append(module.register_forward_hook(self._get_features_hook))
                #self.handlers.append(module.register_backward_hook(self._get_grads_hook))
            
    def remove_handlers(self):
        for handle in self.handlers:
            handle.remove()
    
    def imageRev(img):
        #im1 = np.array(img)
        im1 = 255 - img;
        #im1 = Image.fromarray(im1)
        return im1

    def __call__(self, output_tensor, index=0):
        output = self.output_tensor
        thresh = 0.005
        nms_filter = ln.data.transform.Compose([
            ln.data.transform.GetMultiScaleAnchorBoxes(
                conf_thresh=thresh,
                network_stride=self.net.stride,
                anchors=self.net.anchors
            ),
            ln.data.transform.NMS(
                iou_thresh=thresh
            )
        ])
        loss = ln.network.loss.MultiScaleRegionLoss(
                num_classes=601,
                anchors=self.net.anchors,
                network_stride=self.net.stride
        )

        output_nms = nms_filter(output)
        # reshape coordinates from final_shape to ori_shape only for 1st predicted box :
        output_nms[:, :4] = scale_coords(self.final_shape, output_nms[:, :4],
                                         self.ori_shape).round()
        # takes objectness scores of the first predicted box :
        scores = output_nms[:, -2]
        # add dim
        scores = scores.unsqueeze(0)
        # takes highest objectness score on the grid
        score = torch.max(scores)
        # index of the grid cell with highest objectness score (from 0 to 19*19-1 = 360)
        idx = scores.argmax().numpy()
        # creates a tensor with 1 row and 19 columns, filled with zeros :
        one_hot_output = torch.FloatTensor(1, scores.size()[-1]).zero_()
        print(one_hot_output.shape )
        # adds 1 at the position of the cell with the highest objectness score
        one_hot_output[0][idx] = 1
        # reset gradients of all parameters to zero
        self.net.zero_grad()
        # computes gradients of the objectness scores (scores has shape [1, 19, 19])
        # with one_hot_output as initial gradient (to help the backward pass)
        #scores.backward(gradient=one_hot_output, retain_graph = True)
        #weights = np.mean(gradients, axis=(1, 2))  # take averages for each gradient
        camGrad = GradCAM(self.net, self.target_layers)
        grayscale_cam_Grad = camGrad(input_tensor=self.input_tensor, targets=one_hot_output)
        proc_img = (255 * self.input_tensor.cpu().numpy().transpose(1, 2, 0)).astype(np.uint8)
        proc_cam_Grad = adjust_cam(grayscale_cam=grayscale_cam_Grad)
        vis_Cam = show_cam_on_image(proc_img, proc_cam_Grad, use_rgb=True)
        plt.show(vis_Cam)

        # create empty numpy array for cam
        cam = np.ones(activations.shape[1:], dtype=np.float32)
        
        # multiply each weight with its conv output and then, sum
        for i, w in enumerate(weights-1):
            cam += w * activations[i, :, :]

        cam = np.maximum(cam, 0)
        cam = (cam - np.min(cam)) / (np.max(cam) - np.min(cam))  # normalize between 0-1
        # comment this line if use colormap
        cam = 255 - cam
        # comment these two lines if use color map
        plt.matshow(cam.squeeze())
        plt.show()
        
        '''
        cam = np.uint8(Image.fromarray(cam).resize((self.ori_shape[1],self.ori_shape[0]), Image.ANTIALIAS))/255
        
        original_image = Image.open('./img/4.png')
        I_array = np.array(original_image)
        original_image = Image.fromarray(I_array.astype("uint8"))
        save_class_activation_images(original_image, cam, 'cam-featuremap')
        '''
        
        ################################## 
        # This is for pixel matplot method
        ##################################
        test_img = cv2.imread('./img/1.png')
        heatmap = cam.astype(np.float32)
        heatmap = cv2.resize(heatmap, (test_img.shape[1], test_img.shape[0]))
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        superimposed_img = heatmap * 0.6 + test_img
        cv2.imwrite('./new_map.jpg', superimposed_img)


        ################################################ 
        # Using these codes here, you can generate CAM map for each object 
        ################################################

        box = output_nms[idx][:4].detach().numpy().astype(np.int32)
        #print(box)
        x1, y1, x2, y2 = box
        ratio_x1 = x1 / test_img.shape[1]
        ratio_x2 = x2 / test_img.shape[0]
        ratio_y1 = y1 / test_img.shape[1]
        ratio_y2 = y2 / test_img.shape[0]

        x1_cam = int(cam.shape[1] * ratio_x1)
        x2_cam = int(cam.shape[0] * ratio_x2)
        y1_cam = int(cam.shape[1] * ratio_y1)
        y2_cam = int(cam.shape[0] * ratio_y2)

        cam = cam[y1_cam:y2_cam, x1_cam:x2_cam]
        cam = cv2.resize(cam, (x2 - x1, y2 - y1))
  

        class_id = output[idx][-1].detach().numpy()
        return cam, box, class_id