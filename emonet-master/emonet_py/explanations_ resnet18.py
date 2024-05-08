import os
from emonet import EmoNet, EmoNetPreProcess
from emonet_arousal import EmoNetArousal
from emonet_valence import EmoNetValence
from pytorch_grad_cam import GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image, deprocess_image, show_factorization_on_image
import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
from torch.nn import functional as F
from PIL import Image
import cv2
from torchvision.models import resnet18, ResNet18_Weights
from torchvision import transforms
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
import json
import urllib.request





def get_most_probable_class(preds: torch.Tensor):
    max_prob = preds[0][0]
    max_class = EmoNet.EMOTIONS[0]
    for sample in range(preds.shape[0]):
        for emo_idx in range(20):
            if preds[sample, emo_idx] > max_prob:
                max_prob = preds[sample, emo_idx]
                max_class = EmoNet.EMOTIONS[emo_idx]
    return 100*max_prob, max_class


def adjust_cam(grayscale_cam: np.ndarray):
    re_cam = np.swapaxes(grayscale_cam, 0, 2)
    return re_cam


def get_visualizations(gradcam: int, gradcampp: int, ablationcam: int, scorecam: int, eigencam:int, image: np.ndarray,
                       model: torch.nn.Module, target_layers: list[torch.nn.Module], input_tensor: torch.Tensor,
                       targets=None):
    vis = []
    if gradcam==1:
        camGrad = GradCAM(model, target_layers)
        grayscale_cam_Grad = camGrad(input_tensor=input_tensor, targets=targets)
        proc_cam_Grad = adjust_cam(grayscale_cam=grayscale_cam_Grad)
        vis_Cam = show_cam_on_image(image, proc_cam_Grad, use_rgb=True)
        vis.append(["Grad-CAM", vis_Cam])
    if gradcampp==1:
        camPlus = GradCAMPlusPlus(model, target_layers)
        grayscale_cam_Plus = camPlus(input_tensor=input_tensor, targets=targets)
        proc_cam_Plus = adjust_cam(grayscale_cam=grayscale_cam_Plus)
        vis_Plus = show_cam_on_image(image, proc_cam_Plus, use_rgb=True)
        vis.append(["Grad-CAM++", vis_Plus])
    if ablationcam==1:
        camAbla = AblationCAM(model, target_layers)
        grayscale_cam_Abla = camAbla(input_tensor=input_tensor, targets=targets)
        proc_cam_Abla = adjust_cam(grayscale_cam=grayscale_cam_Abla)
        vis_Abla = show_cam_on_image(image, proc_cam_Abla, use_rgb=True)
        vis.append(["Ablation-CAM", vis_Abla])
    if scorecam==1:
        camScore = ScoreCAM(model, target_layers)
        grayscale_cam_Score = camScore(input_tensor=input_tensor, targets=targets)
        proc_cam_Score = adjust_cam(grayscale_cam=grayscale_cam_Score)
        vis_Score = show_cam_on_image(image, proc_cam_Score, use_rgb=True)
        vis.append(["Score-CAM", vis_Score])
    if eigencam==1:
        camEigen = EigenCAM(model, target_layers)
        grayscale_cam_Eigen = camEigen(input_tensor=input_tensor, targets=targets)
        proc_cam_Eigen = adjust_cam(grayscale_cam=grayscale_cam_Eigen)
        vis_Eigen = show_cam_on_image(image, proc_cam_Eigen, use_rgb=True)
        vis.append(["Eigen-CAM", vis_Eigen])

    return vis


def plot_cam(visualization, class_label, prob_label, class_index):
    ncol = len(visualization)+1
    fig, ax = plt.subplots(1, ncol)
    ax[0].imshow(img_loaded, interpolation='none')
    ax[0].axis('off')
    ax[0].set_title("Image")
    for i in range(len(visualization)):
        ax[i+1].imshow(visualization[i][1], interpolation='none')
        ax[i+1].axis('off')
        ax[i+1].set_title(visualization[i][0])
    plt.subplots_adjust(wspace=0.1, hspace=0)
    plt.suptitle(f"Class: {class_label:}\nConfidence: {prob_label:0.2f}% \nIndex: {class_index}")
    plt.show()


if __name__ == '__main__':
    # Instantiations & definitions
    img_loaded = os.path.join('..', 'data/images', 'boxer_cat.png')
    with torch.no_grad():
        with open(img_loaded, 'rb') as f:
            img_loaded = Image.open(f).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize the image to 224x224
        transforms.ToTensor(),  # Convert the image to a PyTorch tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize the image
    ])
    resize = transforms.Compose([transforms.Resize((224, 224))])
    img_resized = resize(img_loaded)
    img_tensor = transform(img_loaded)
    # Model
    model = resnet18(pretrained=True)
    output = model(img_tensor.unsqueeze(0))
    activation_maps = [model.layer4[-1]]
    # Images
    #image = torch.Tensor.numpy(img_tensor)
    proc_img = np.float32(deprocess_image(img_resized))/255
    #Check labels
    url = "https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json"
    labels_path, _ = urllib.request.urlretrieve(url, "imagenet-simple-labels.json")
    # Load the class labels
    with open(labels_path) as f:
        class_labels = json.load(f)
    #print("Label for class index 282:", class_labels[282])
    # Visualization
    valence = 8
    arousal = 7
    probabilities = torch.nn.functional.softmax(output[0], dim=0)
    # Choose particular class from label
    #class_index = class_labels.index("pug")
    #prob_value = probabilities[class_index].item()*100

    # Choose nth most probable class as target class
    n = 2
    prob_value, class_index = torch.topk(probabilities, n)
    prob_value = prob_value[-1].item()*100
    class_index = class_index[-1].item()
    print("Predicted class index:", class_index)
    print("Probability:", prob_value)
    print("Class: ", class_labels[class_index])
    target_categories = np.array([class_index])
    targets = [ClassifierOutputTarget(category) for category in target_categories]
    vis = get_visualizations(gradcam=1, gradcampp=0, ablationcam=0, scorecam=0, eigencam=0, image=proc_img,
                             model=model, target_layers=activation_maps, input_tensor=img_tensor.unsqueeze(0), targets=None)
    plot_cam(visualization=vis, class_label=class_labels[class_index], prob_label=prob_value, class_index=class_index)
