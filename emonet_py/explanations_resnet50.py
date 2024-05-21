import os
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
from torchvision.models import resnet50, ResNet50_Weights
from torchvision import transforms
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
import json
import urllib.request
from explanations_liftcam import CAM_Explanation
from captum.attr import GuidedGradCam



def get_most_probable_class(preds: torch.Tensor):
    max_prob = preds[0][0]
    max_class = EmoNet.EMOTIONS[0]
    for sample in range(preds.shape[0]):
        for emo_idx in range(20):
            if preds[sample, emo_idx] > max_prob:
                max_prob = preds[sample, emo_idx]
                max_class = EmoNet.EMOTIONS[emo_idx]
    return 100*max_prob, max_class


def get_visualizations(gradcam: int, gradcampp: int, ablationcam: int, scorecam: int, eigencam:int, liftcam:int, lrpcam:int, limecam:int,
                       guided:int,image: np.ndarray, model: torch.nn.Module, target_layers: list[torch.nn.Module], input_tensor: torch.Tensor,
                       class_index: int, img_size, file_name, targets=None):
    vis = []
    if gradcam==1:
        camGrad = GradCAM(model, target_layers)
        grayscale_cam_Grad = camGrad(input_tensor=input_tensor, targets=targets)
        grayscale_cam_Grad = grayscale_cam_Grad[0, :]
        np.save("cam_grad_"+file_name, grayscale_cam_Grad)
        vis_Cam = show_cam_on_image(image, grayscale_cam_Grad, use_rgb=True)
        vis.append(["Grad-CAM", vis_Cam])
    if gradcampp==1:
        camPlus = GradCAMPlusPlus(model, target_layers)
        grayscale_cam_Plus = camPlus(input_tensor=input_tensor, targets=targets)
        grayscale_cam_Plus = grayscale_cam_Plus[0, :]
        vis_Plus = show_cam_on_image(image, grayscale_cam_Plus, use_rgb=True)
        vis.append(["Grad-CAM++", vis_Plus])
    if ablationcam==1:
        camAbla = AblationCAM(model, target_layers)
        grayscale_cam_Abla = camAbla(input_tensor=input_tensor, targets=targets)
        grayscale_cam_Abla = grayscale_cam_Abla[0, :]
        vis_Abla = show_cam_on_image(image, grayscale_cam_Abla, use_rgb=True)
        vis.append(["Ablation-CAM", vis_Abla])
    if scorecam==1:
        camScore = ScoreCAM(model, target_layers)
        grayscale_cam_Score = camScore(input_tensor=input_tensor, targets=targets)
        grayscale_cam_Score = grayscale_cam_Score[0, :]
        vis_Score = show_cam_on_image(image, grayscale_cam_Score, use_rgb=True)
        vis.append(["Score-CAM", vis_Score])
    if eigencam==1:
        camEigen = EigenCAM(model, target_layers)
        grayscale_cam_Eigen = camEigen(input_tensor=input_tensor, targets=targets)
        grayscale_cam_Eigen = grayscale_cam_Eigen[0, :]
        vis_Eigen = show_cam_on_image(image, grayscale_cam_Eigen, use_rgb=True)
        vis.append(["Eigen-CAM", vis_Eigen])
    if liftcam==1:
        camLift = CAM_Explanation(model, "LIFT-CAM")
        grayscale_cam_Lift = camLift(input_tensor.cpu(), int(class_index), img_size)
        grayscale_cam_Lift = torch.squeeze(grayscale_cam_Lift).cpu().detach().numpy()
        vis_Lift = show_cam_on_image(image, grayscale_cam_Lift, use_rgb=True)
        vis.append(["Lift-CAM", vis_Lift])
    if lrpcam==1:
        camLrp = CAM_Explanation(model, "LRP-CAM")
        grayscale_cam_Lrp = camLrp(input_tensor.cpu(), int(class_index), img_size)
        grayscale_cam_Lrp = torch.squeeze(grayscale_cam_Lrp).cpu().detach().numpy()
        vis_Lrp = show_cam_on_image(image, grayscale_cam_Lrp, use_rgb=True)
        vis.append(["Lrp-CAM", vis_Lrp])
    if limecam==1:
        camLime = CAM_Explanation(model, "LIME-CAM")
        grayscale_cam_Lime = camLime(input_tensor.cpu(), int(class_index), img_size)
        grayscale_cam_Lime = torch.squeeze(grayscale_cam_Lime).cpu().detach().numpy()
        vis_Lime = show_cam_on_image(image, grayscale_cam_Lime, use_rgb=True)
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


def plot_cam(visualization, image, class_label, prob_label, class_index):
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
    plt.suptitle(f"Class: {class_label:}\nConfidence: {prob_label:0.2f}% \nIndex: {class_index}")
    plt.show()


def explanations_resnet50(img_path, file_name):
    # Input processing
    #file_name = 'both'
    #img_path = os.path.join('..', 'data/images', file_name+'.png')
    with torch.no_grad():
        with open(img_path, 'rb') as f:
            img_loaded = Image.open(f).convert('RGB')
    img_size = img_loaded.size
    #resize = transforms.Compose([transforms.Resize((224, 224))])
    #img_loaded = resize(img_loaded)
    proc_img = np.float32(img_loaded)/255
    input_tensor = preprocess_image(img_loaded, mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])

    # Model
    model = resnet50(pretrained=True).eval()
    output = model(input_tensor)
    activation_maps = [model.layer4]
    #image = torch.Tensor.numpy(img_tensor)
    #Check labels
    url = "https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json"
    labels_path, _ = urllib.request.urlretrieve(url, "imagenet-simple-labels.json")
    # Load the class labels
    with open(labels_path) as f:
        class_labels = json.load(f)
    #print("Label for class index 282:", class_labels[282])
    # Visualization
    probabilities = torch.nn.functional.softmax(output[0], dim=0)
    # Choose max prob
    #class_index = torch.argmax(probabilities).item()+1
    #prob_value = torch.max(probabilities).item()*100

    # Choose particular class from label
    #class_index = class_labels.index("pug")
    #prob_value = probabilities[class_index].item()*100

    # Choose nth most probable class as target class
    n = 1
    prob_value, class_index = torch.topk(probabilities, n)
    prob_value = prob_value[-1].item()*100
    class_index = class_index[-1].item()
    class_label = class_labels[class_index]
    print("Predicted class index:", class_index)
    print(f"Probability: {prob_value:0.2f}%")
    print("Class: ", class_label)
    target_categories = np.array([class_index])
    targets = [ClassifierOutputTarget(class_index)]
    vis = get_visualizations(gradcam=1, gradcampp=1, ablationcam=1, scorecam=1, eigencam=0, liftcam=1, lrpcam=0, limecam=0, guided=0,
                             image=proc_img,model=model, target_layers=activation_maps, input_tensor=input_tensor, class_index=class_index,
                             img_size=img_size, file_name=file_name, targets=targets)
    plot_cam(visualization=vis, image=img_loaded, class_label=class_label, prob_label=prob_value, class_index=class_index)
