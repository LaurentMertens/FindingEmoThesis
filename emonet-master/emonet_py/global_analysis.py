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
import signal
import json

from img_resize_preproc import ImgResize
from explanations_emonet import get_visualizations, plot_cam, ExplanationsEmonet
from pytorch_grad_cam.utils.image import scale_cam_image, scale_accross_batch_and_channels, show_cam_on_image
from local_analysis import LocalAnalysis

import os
from PIL import Image
import pandas as pd

def get_dir_image_path(file_path):
    return os.path.basename(os.path.dirname(file_path)) + '/' + os.path.basename(file_path)

def merge_annotations(df_emonet):
    """
    Merge the FindingEmo annotations with the outputs of EmoNet.
    """
    df_annotations = pd.read_csv('annotations_single.ann')
    # modify annotations header to distinguish from EmoNet outputs
    df_annotations = df_annotations.rename(columns={'user': 'ann_user', 'image_path': 'ann_original_image_path',
                                                    'reject': 'ann_reject', 'tag': 'age_group',
                                                    'valence': 'ann_valence', 'arousal': 'ann_arousal',
                                                    'emotion': 'ann_emotion', 'dec_factors': 'ann_dec_factors',
                                                    'ambiguity': 'ann_ambiguity',
                                                    'fmri_candidate': 'ann_fmri_candidate',
                                                    'datetime': 'ann_datetime'})
    # add 'dir_image_path' as path containing only folder name and file name
    df_annotations['dir_image_path'] = df_annotations['ann_original_image_path'].apply(get_dir_image_path)
    # merge both dataframes
    merged_df = pd.merge(df_emonet, df_annotations, on='dir_image_path')
    return merged_df


class GlobalAnalysis:
    """
    GlobalAnalysis class used to apply the EmoNet and Yolov3 pipeline and extract all outputs in .csv format.
    """
    # header for EmoNet outputs
    df_emonet_header = ['dir_image_path', 'emonet_adoration_prob', 'emonet_aesthetic_appreciation_prob',
                        'emonet_amusement_prob',
                        'emonet_anxiety_prob', 'emonet_awe_prob', 'emonet_boredom_prob', 'emonet_confusion_prob',
                        'emonet_craving_prob',
                        'emonet_disgust_prob', 'emonet_empathetic_pain_prob', 'emonet_entrancement_prob',
                        'emonet_excitement_prob',
                        'emonet_fear_prob', 'emonet_horror_prob', 'emonet_interest_prob', 'emonet_joy_prob',
                        'emonet_romance_prob',
                        'emonet_sadness_prob', 'emonet_sexual_desire_prob', 'emonet_surprise_prob', 'emonet_valence',
                        'emonet_arousal']

    def __init__(self):
        self.local_analysis = LocalAnalysis()
        self.expl_emo = ExplanationsEmonet()


    def update_emonet_df(self, file_path, df_emonet):
        """
        Update the EmoNet output dataframe with the analysis of a new image.
        """
        # define paths and names
        image_name = os.path.basename(file_path)
        # define dir_image_path as part of the path containing file and its folder
        dir_image_path = get_dir_image_path(file_path)
        # get outputs of EmoNet
        emotion, arousal, valence = self.expl_emo.explanations_emonet(file_path, image_name)
        # first element of the new row is the image path
        new_row = [dir_image_path]
        # append probabilities for the output tensor emotion
        for sample in range(emotion.shape[0]):
            for emo_idx in range(20):
                new_row.append(emotion[sample, emo_idx].item())
        # add valence and arousal
        new_row += [valence, arousal]
        # add the row to the dataframe
        df_emonet.loc[len(df_emonet)] = new_row
        return df_emonet

    def update_yolo_df(self, file_path, df_yolo):
        """
        Update the yolo output dataframe with the yolo output of a new image.
        """
        # extract image name
        image_name = os.path.basename(file_path)
        # define dir_image_path as part of the path containing file and its folder
        dir_image_path = get_dir_image_path(file_path)
        # get new yolo outputs of image
        new_df_yolo = self.local_analysis.local_analysis(file_path, image_name)
        # insert dir_image_path as first column (one folder + file name only)
        new_df_yolo.insert(0, "dir_image_path", dir_image_path, True)
        return pd.concat([df_yolo, new_df_yolo], ignore_index=True)

    def save_model_outputs(self, directory, total_nb_images, nb_folders_to_process):
        """
        save to .csv two files containing the outputs of EmoNet (and FindingEmo annotations) and Yolov3 respectively
        """
        count_images = 0
        # create EmoNet output dataframe with header
        df_emonet = pd.DataFrame(columns=self.df_emonet_header)
        # create Yolo output dataframe
        df_yolo = pd.DataFrame()
        # loop over folders containing the images
        for folder_name in os.listdir(directory)[:nb_folders_to_process]:
            # raising exceptions for corrupted files
            try:
                folder_path = os.path.join(directory, folder_name)
                for image_name in os.listdir(folder_path):
                    if image_name.endswith(('.jpg', '.jpeg', '.png')):
                        # raising exceptions for corrupted files
                        try:
                            file_path = os.path.join(folder_path, image_name)
                            print("Processing:", file_path)
                            # update emonet dataframe with outputs from new image
                            df_emonet = self.update_emonet_df(file_path, df_emonet)
                            # update yolo dataframe with outputs from new image
                            df_yolo = self.update_yolo_df(file_path, df_yolo)
                            count_images += 1
                            print(f"Progression: {(count_images / total_nb_images) * 100:0.2f}%")
                        except Exception as e:
                            print(f'Error processing: {folder_path + image_name} : {e}')
            except Exception as e:
                print(f'Error processing: {directory + folder_name} : {e}')
        # merge modified annotations with emonet dataframe
        df_emonet_ann = merge_annotations(df_emonet)
        # save emonet_ann outputs dataframe as .csv
        df_emonet_ann.to_csv('emonet_ann_outputs')
        # safe yolo outputs dataframe as .csv
        df_yolo.to_csv('yolo_outputs')

        return count_images

    @staticmethod
    def get_number_of_images(directory, nb_folders_to_process):
        nb_images = 0
        for folder_name in os.listdir(directory)[:nb_folders_to_process]:
            try:
                folder_path = os.path.join(directory, folder_name)
                for image_name in os.listdir(folder_path):
                    if image_name.endswith(('.jpg', '.jpeg', '.png')):
                        nb_images += 1
            except Exception as e:
                print(f"Error processing {directory + folder_name}: {e}")
                continue
        return nb_images


if __name__ == '__main__':
    ga = GlobalAnalysis()
    nb_folders_to_process = 10
    directory_path = os.path.join('findingemo_dataset')
    total_number_images = ga.get_number_of_images(directory_path, nb_folders_to_process)
    print("Total number of images = ", total_number_images)
    nb_img_processed = ga.save_model_outputs(directory_path, total_number_images, nb_folders_to_process)
    print("Total number of images processed: ", nb_img_processed)
