import os

import pandas as pd

from emonet import EmoNet, EmoNetPreProcess
from emonet_arousal import EmoNetArousal
from emonet_valence import EmoNetValence
from pytorch_grad_cam import GradCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, EigenCAM, guided_backprop
from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image, deprocess_image, show_factorization_on_image
import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
from torch.nn import functional as F
from PIL import Image
import cv2
from explanations_liftcam import CAM_Explanation
import alexnet_big





def get_most_probable_class(preds: torch.Tensor):
    max_prob = preds[0][0]
    max_class = EmoNet.EMOTIONS[0]
    for sample in range(preds.shape[0]):
        for emo_idx in range(20):
            if preds[sample, emo_idx] > max_prob:
                max_prob = preds[sample, emo_idx]
                max_class = EmoNet.EMOTIONS[emo_idx]
                class_index = emo_idx
    return 100*max_prob, max_class, class_index

def get_visualizations(gradcam: int, gradcampp: int, ablationcam: int, scorecam: int, eigencam:int, liftcam:int, lrpcam:int, limecam:int,
                       guided: int, image: np.ndarray, model: torch.nn.Module, target_layers: list[torch.nn.Module], input_tensor: torch.Tensor,
                       class_index: int, img_size, file_name: str, image_weight: float = 0.5, targets=None):
    vis = []
    if gradcam==1:
        camGrad = GradCAM(model, target_layers)
        grayscale_cam_Grad = camGrad(input_tensor=input_tensor, targets=targets)
        grayscale_cam_Grad = grayscale_cam_Grad[0, :]
        np.save("cam_grad_dataset/cam_grad_"+file_name, grayscale_cam_Grad)
        vis_Cam = show_cam_on_image(image, grayscale_cam_Grad, use_rgb=True, image_weight=image_weight)
        vis.append(["Grad-CAM", vis_Cam])
    if gradcampp==1:
        camPlus = GradCAMPlusPlus(model, target_layers)
        grayscale_cam_Plus = camPlus(input_tensor=input_tensor, targets=targets)
        grayscale_cam_Plus = grayscale_cam_Plus[0, :]
        vis_Plus = show_cam_on_image(image, grayscale_cam_Plus, use_rgb=True, image_weight=image_weight)
        vis.append(["Grad-CAM++", vis_Plus])
    if ablationcam==1:
        camAbla = AblationCAM(model, target_layers)
        grayscale_cam_Abla = camAbla(input_tensor=input_tensor, targets=targets)
        grayscale_cam_Abla = grayscale_cam_Abla[0, :]
        vis_Abla = show_cam_on_image(image, grayscale_cam_Abla, use_rgb=True, image_weight=image_weight)
        vis.append(["Ablation-CAM", vis_Abla])
    if scorecam==1:
        camScore = ScoreCAM(model, target_layers)
        grayscale_cam_Score = camScore(input_tensor=input_tensor, targets=targets)
        grayscale_cam_Score = grayscale_cam_Score[0, :]
        vis_Score = show_cam_on_image(image, grayscale_cam_Score, use_rgb=True, image_weight=image_weight)
        vis.append(["Score-CAM", vis_Score])
    if eigencam==1:
        camEigen = EigenCAM(model, target_layers)
        grayscale_cam_Eigen = camEigen(input_tensor=input_tensor, targets=targets)
        grayscale_cam_Eigen = grayscale_cam_Eigen[0, :]
        vis_Eigen = show_cam_on_image(image, grayscale_cam_Eigen, use_rgb=True, image_weight=image_weight)
        vis.append(["Eigen-CAM", vis_Eigen])
    if liftcam==1:
        input_tensor = input_tensor
        camLift = CAM_Explanation(model, "LIFT-CAM")
        grayscale_cam_Lift = camLift(input_tensor.cpu(), int(class_index), img_size)
        grayscale_cam_Lift = torch.squeeze(grayscale_cam_Lift).cpu().detach().numpy()
        vis_Lift = show_cam_on_image(image, grayscale_cam_Lift, use_rgb=True, image_weight=image_weight)
        vis.append(["Lift-CAM", vis_Lift])
    if lrpcam==1:
        input_tensor = input_tensor
        camLrp = CAM_Explanation(model, "LRP-CAM")
        grayscale_cam_Lrp = camLrp(input_tensor.cpu(), int(class_index), img_size)
        grayscale_cam_Lrp = torch.squeeze(grayscale_cam_Lrp).cpu().detach().numpy()
        vis_Lrp = show_cam_on_image(image, grayscale_cam_Lrp, use_rgb=True, image_weight=image_weight)
        vis.append(["Lrp-CAM", vis_Lrp])
    if limecam==1:
        input_tensor = input_tensor
        camLime = CAM_Explanation(model, "LIME-CAM")
        grayscale_cam_Lime = camLime(input_tensor.cpu(), int(class_index), img_size)
        grayscale_cam_Lime = torch.squeeze(grayscale_cam_Lime).cpu().detach().numpy()
        vis_Lime = show_cam_on_image(image, grayscale_cam_Lime, use_rgb=True, image_weight=image_weight)
        vis.append(["Lime-CAM", vis_Lime])
    if guided==1:
        # Get localization map
        camGrad = GradCAM(model, target_layers)
        grayscale_cam_Grad = camGrad(input_tensor=input_tensor, targets=targets)
        grayscale_cam_Grad = grayscale_cam_Grad[0, :]
        vis_Cam = show_cam_on_image(image, grayscale_cam_Grad, use_rgb=True, image_weight=0.0)
        # Get guided backpropagation map
        camGuided = guided_backprop.GuidedBackpropReLUModel(model, "cpu")
        grayscale_cam_Guided = camGuided(input_tensor, class_index)
        # Elementwise multiplication
        guidedmap = grayscale_cam_Guided*vis_Cam
        vis.append(["Guided Backprop", grayscale_cam_Guided])
        vis.append(["Guided Grad-Cam", guidedmap])

    return vis


def plot_cam(visualization, image, class_label, prob_label, val, aro):
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


class ExplanationsEmonet:
    header=['img_path', 'emonet_adoration_prob', 'emonet_aesthetic_appreciation_prob', 'emonet_amusement_prob',
                 'emonet_anxiety_prob', 'emonet_awe_prob', 'emonet_boredom_prob', 'emonet_confusion_prob',
                 'emonet_craving_prob',
                 'emonet_disgust_prob', 'emonet_empathetic_pain_prob', 'emonet_entrancement_prob',
                 'emonet_excitement_prob',
                 'emonet_fear_prob', 'emonet_horror_prob', 'emonet_interest_prob', 'emonet_joy_prob',
                 'emonet_romance_prob',
                 'emonet_sadness_prob', 'emonet_sexual_desire_prob', 'emonet_surprise_prob', 'emonet_valence',
                 'emonet_arousal',
                 'annotation_user', 'annotation_original_img_path', 'annotation_reject', 'annotation_tag',
                 'annotation_age_group',
                 'annotation_valence', 'annotation_arousal', 'annotation_emotion', 'annotation_deciding_factors',
                 'annotation_ambiguity',
                 'annotation_fmri_candidate', 'annotation_datetime']
    def __init__(self):
        self.emonet = EmoNet(b_eval=True)
        self.emonet_pp = EmoNetPreProcess()

    def create_output_dataframe(self, file_path, emo, aro, val):
        output_emonet = pd.DataFrame(columns=self.header)
        output_emonet.loc[0]



    def explanations_emonet(self, img_path, file_name, show_plot=False):
        # Instantiations & definitions
        img_tensor, img_loaded = self.emonet_pp(img_path)
        img_size = img_loaded.size
        in_tensor = img_tensor.unsqueeze(0)
        emo_aro = EmoNetArousal()
        arousal = emo_aro(in_tensor).item()
        emo_val = EmoNetValence()
        valence = emo_val(in_tensor).item()
        # Images
        proc_img = np.float32(img_loaded)/255
        # Model
        emo_model = self.emonet.emonet
        pred = emo_model(in_tensor)
        max_prob, max_class, class_index = get_most_probable_class(pred)
        activation_maps = [emo_model.conv5]
        # Visualization
        #emonet.prettyprint(pred, b_pc=True)
        vis = get_visualizations(gradcam=1, gradcampp=0, ablationcam=0, scorecam=0, eigencam=0, liftcam=0, lrpcam=0, limecam=0, guided=0,
                                 image=proc_img, model=emo_model, target_layers=activation_maps, input_tensor=in_tensor,
                                 class_index=class_index, img_size=img_size, file_name=file_name, targets=None)
        if show_plot:
            plot_cam(visualization=vis, image=img_loaded, class_label=max_class, prob_label=max_prob, val=valence, aro=arousal)

        return pred, arousal, valence

