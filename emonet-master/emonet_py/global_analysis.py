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


class GlobalAnalysis:
    def __init__(self):
        self.local_analysis = LocalAnalysis()
        self.expl_emo = ExplanationsEmonet()

    @staticmethod
    def save_dictionary(dictionary):
        with open("global_analysis_dictionary.json", "w") as file:
            json.dump(dictionary, file)
        print(" 'global_analysis_dictionary' saved successfully.")

    def get_emotions_object_dict(self, emotions_objects_dict, file_path, file_name):
        file_name = os.path.splitext(file_name)[0]
        detected_emotion = self.expl_emo.explanations_emonet(file_path, file_name, show_plot=False)
        detected_objects = self.local_analysis.local_analysis(file_path, file_name, show_box=False)
        for index, row in detected_objects.iterrows():
            if detected_emotion in emotions_objects_dict:
                if row['class_label'] in emotions_objects_dict[detected_emotion]:
                    emotions_objects_dict[detected_emotion][row['class_label']] += 1
                else:
                    emotions_objects_dict[detected_emotion][row['class_label']] = 1
            else:
                emotions_objects_dict[detected_emotion] = {row['class_label']: 1}
        return emotions_objects_dict

    @staticmethod
    def get_number_of_images(directory, nb_folders_to_process):
        nb_images = 0
        for folder_name in os.listdir(directory)[:nb_folders_to_process]:
            try:
                folder_path = os.path.join(directory, folder_name)
                for file_name in os.listdir(folder_path):
                    if file_name.endswith(('.jpg', '.jpeg', '.png')):
                        nb_images += 1
            except Exception as e:
                print(f"Error processing {directory+folder_name}: {e}")
                continue
        return nb_images

    def global_analysis(self, directory, total_nb_images, nb_images_to_process):
        emotions_objects_dict = {}
        count_images = 0
        for folder_name in os.listdir(directory)[:nb_folders_to_process]:
            try:
                folder_path = os.path.join(directory, folder_name)
                for file_name in os.listdir(folder_path):
                    if file_name.endswith(('.jpg', '.jpeg', '.png')):
                        try:
                            file_path = os.path.join(folder_path, file_name)
                            print("Processing:", file_path)
                            emotions_objects_dict = self.get_emotions_object_dict(emotions_objects_dict, file_path, file_name)
                            count_images += 1
                        except Exception as e:
                            print(f"Error processing {folder_path}: {e}")
                            continue
            except Exception as e:
                print(f"Error processing {directory+folder_name}: {e}")
                continue
            print(f"Progression: {(count_images/total_nb_images)*100:0.2f}%")
        return emotions_objects_dict, count_images

    @staticmethod
    def plot_statistics(data):
        # Create subplots
        fig, axs = plt.subplots(2, 3, figsize=(15, 10))
        axs = axs.ravel()

        # Plotting pie charts for each emotion
        for i, (emotion, obj_freq) in enumerate(data.items()):
            objects = list(obj_freq.keys())
            frequencies = list(obj_freq.values())

            # Plotting
            axs[i].pie(frequencies, labels=objects, autopct='%1.1f%%', startangle=90)
            axs[i].set_title(f'Distribution of Objects for {emotion} Emotion')

        # Adjust layout
        plt.tight_layout()

        # Show plot
        plt.show()

    @staticmethod
    def print_statistics(dico):
        for emotion in dico:
            n = 0
            for obj in dico[emotion]:
                n += dico[emotion][obj]
            print(emotion)
            for obj in dico[emotion]:
                print(f'{obj}: {(dico[emotion][obj]/n)*100:0.1f}%')
            print("________________")


if __name__ == '__main__':
    ga = GlobalAnalysis()

    nb_folders_to_process = 10
    directory_path = os.path.join(os.path.expanduser('~'),
                                     "Work", "Projects", "NeuroANN",
                                     "Data", "AnnImagesProlific", 'Run_2')
    total_number_images = ga.get_number_of_images(directory_path, nb_folders_to_process)
    print("Total number of images = ", total_number_images)
    statistics_dict, n = ga.global_analysis(directory_path, total_number_images, nb_folders_to_process)
    ga.print_statistics(statistics_dict)
    print("Total number of images processed: ", n)
    ga.save_dictionary(statistics_dict)


