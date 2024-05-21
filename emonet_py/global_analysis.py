"""
A test script to create a pipeline that first uses YoLo v3 + OpenImages to detect faces

.. codeauthor:: Laurent Mertens <laurent.mertens@kuleuven.be>
"""

import os

import PIL
import pandas as pd
import torch

from emonet_py.explanations_emonet import ExplanationsEmonet
from explanations_model import ExplanationsModel
from local_analysis import LocalAnalysis
# matplotlib.use('pdf')  # Purely for debugging purposes


# header for EmoNet outputs
df_emonet_header = ['dir_image_path', 'emonet_emotion', 'emonet_emotion_conf', 'emonet_adoration_conf',
                    'emonet_aesthetic_appreciation_conf', 'emonet_amusement_conf', 'emonet_anxiety_conf',
                    'emonet_awe_conf', 'emonet_boredom_conf', 'emonet_confusion_conf',
                    'emonet_craving_conf', 'emonet_disgust_conf', 'emonet_empathetic_pain_conf',
                    'emonet_entrancement_conf', 'emonet_excitement_conf',
                    'emonet_fear_conf', 'emonet_horror_conf', 'emonet_interest_conf', 'emonet_joy_conf',
                    'emonet_romance_conf',
                    'emonet_sadness_conf', 'emonet_sexual_desire_conf', 'emonet_surprise_conf', 'emonet_valence',
                    'emonet_arousal']


def get_dir_image_path(file_path):
    return os.path.basename(os.path.dirname(file_path)) + '/' + os.path.basename(file_path)


class GlobalAnalysis:
    """
    GlobalAnalysis class used to apply the EmoNet and Yolov3 pipeline and extract all outputs in .csv format.
    """

    def __init__(self, device=torch.device('cpu')):
        self.device = device
        self.local_analysis = LocalAnalysis(device=device)
        self.expl_emo = ExplanationsEmonet(device=device)
        # self.expl_emo = ExplanationsModel(device=device, model=model, model_pp=model_pp, target_layers=target_layers)

    def update_emonet_df(self, file_path, df_emonet):
        """
        Update the EmoNet output dataframe with the analysis of a new image.
        """
        # define paths and names
        image_name = os.path.basename(file_path)
        # define dir_image_path as part of the path containing file and its folder
        dir_image_path = get_dir_image_path(file_path)
        # get outputs of EmoNet
        # max_emotion, max_prob, emotion_tensor, arousal, valence, cam_outputs =\
        #     self.expl_emo.get_explanations_for_image(file_path, image_name)
        max_emotion, max_prob, emotion_tensor, arousal, valence, cam_outputs =\
            self.expl_emo.explanations_emonet(file_path, image_name)
        # first elements of the new row are the image path and predicted emotion
        new_row = [dir_image_path, max_emotion, max_prob]
        # append probabilities for the output tensor emotion
        for sample in range(emotion_tensor.shape[0]):
            for emo_idx in range(20):
                new_row.append(emotion_tensor[sample, emo_idx].item())
        # add valence and arousal
        new_row += [valence, arousal]
        # add the row to the dataframe
        df_emonet.loc[len(df_emonet)] = new_row
        return df_emonet, max_emotion, max_prob, cam_outputs

    def update_yolo_df(self, file_path, max_emotion, max_prob, df_yolo, cam_output):
        """
        Update the yolo output dataframe with the yolo output of a new image.
        """
        # extract image name
        image_name = os.path.basename(file_path)
        # define dir_image_path as part of the path containing file and its folder
        dir_image_path = get_dir_image_path(file_path)
        # get new yolo outputs of image
        new_df_yolo = self.local_analysis.local_analysis(file_path, image_name, cam_output=cam_output)
        # insert emotion confidence
        new_df_yolo.insert(0, "emonet_emotion_conf", max_prob, True)
        # insert most probable emotion
        new_df_yolo.insert(0, "emonet_emotion", max_emotion, True)
        # insert dir_image_path as first column (one folder + file name only)
        new_df_yolo.insert(0, "dir_image_path", dir_image_path, True)
        return pd.concat([df_yolo, new_df_yolo], ignore_index=True)

    def process_model(self, img_dir, model_out_file='emonet_outputs.csv', yolo_out_file='yolo_outputs.csv'):
        """
        save to two .csv files containing the outputs of EmoNet (and FindingEmo annotations) and Yolov3 respectively

        :param img_dir: root path containing the images to process
        :return:
        """
        count_images = 0
        # create EmoNet output dataframe with header
        df_emonet = pd.DataFrame(columns=df_emonet_header)
        # create Yolo output dataframe
        df_yolo = pd.DataFrame()

        # Get image paths
        image_paths = ga.get_image_paths(img_dir)
        nb_images = len(image_paths)
        for image_path in image_paths:
            # Let's be sure
            if not image_path.endswith(('.jpg', '.jpeg', '.png')):
                raise ValueError(f"Expected image file of type jpg/jpeg/png, got something else instead.\n[{image_path}]")
            # raising exceptions for corrupted files
            try:
                print("Processing:", image_path)

                # update emonet dataframe with outputs from new image
                df_emonet, max_emotion, max_prob, cam_outputs = self.update_emonet_df(image_path, df_emonet)

                # if len(cam_outputs) > 1:
                #     raise ValueError("Don't know how to handle multiple CAM outputs for now.\n"
                #                      "Please make sure you're only using one CAM method at a time.")

                # update yolo dataframe with outputs from new image
                df_yolo = self.update_yolo_df(image_path, max_emotion, max_prob, df_yolo, cam_outputs[0])

                count_images += 1
                print(f"Progression: {(count_images / nb_images) * 100:0.2f}%")
            except PIL.UnidentifiedImageError as e:
                print(f'Error processing: {image_path}:\n\t{e}')
                continue
            if count_images % 500 == 0:
                print('Saving progress...')
                df_emonet.to_csv(model_out_file)
                df_yolo.to_csv(yolo_out_file)
                print(f'emonet_outputs and df_yolo saved to \'{model_out_file}\' and \'{yolo_out_file}\'')

        # save emonet_ann outputs dataframe as .csv
        df_emonet.to_csv(model_out_file)
        # safe yolo outputs dataframe as .csv
        df_yolo.to_csv(yolo_out_file)
        print(f'emonet_outputs and df_yolo saved to \'{model_out_file}\' and \'{yolo_out_file}\'')

        return count_images

    @classmethod
    def get_number_of_images(cls, directory, nb_elems_to_process=None):
        if nb_elems_to_process is None:
            nb_elems_to_process = len(os.listdir(directory))

        nb_images = 0
        for e in os.listdir(directory)[:nb_elems_to_process]:
            e_path = os.path.join(directory, e)
            if os.path.isdir(e_path):
                nb_images += cls.get_number_of_images(e_path)
            elif e_path.endswith(('.jpg', '.jpeg', '.png')):
                nb_images += 1

        return nb_images

    @classmethod
    def get_image_paths(cls, directory, nb_elems_to_process=None):
        if nb_elems_to_process is None:
            nb_elems_to_process = len(os.listdir(directory))

        image_paths = []
        for e in os.listdir(directory)[:nb_elems_to_process]:
            e_path = os.path.join(directory, e)
            if os.path.isdir(e_path):
                image_paths += cls.get_image_paths(e_path)
            elif e_path.endswith(('.jpg', '.jpeg', '.png')):
                image_paths.append(e_path)

        return image_paths

    @staticmethod
    def get_number_of_folders(directory):
        nb_dirs = 0
        for e in os.listdir(directory):
            if os.path.isdir(os.path.join(directory, e)):
                nb_dirs += 1

        return nb_dirs


if __name__ == '__main__':
    # instantiation
    ga = GlobalAnalysis(device=torch.device('cuda'))

    # path of directory containing all folders, each with images
    directory_path = os.path.join(os.path.expanduser('~'),
                                  'Work', 'Projects', 'NeuroANN', 'Data', 'AnnImagesProlific')

    # get info
    total_number_folders = ga.get_number_of_folders(directory_path)
    print("Total number of folders = ", total_number_folders)

    # set nb of folders to process
    # Prefix with '_' to not conflict with the "nb_..." variable inside the "get_number_of_images" method
    _nb_elems_to_process = None
    # get nb of images to process
    nb_images_to_process = ga.get_number_of_images(directory_path, _nb_elems_to_process)
    print("Total number of images to process = ", nb_images_to_process)

    # Sanity check
    print(f"Total number of image paths: {len(ga.get_image_paths(directory_path))}")
    # save model outputs & get number of images actually processed
    nb_img_processed = ga.process_model(directory_path, model_out_file='emonet_outputs_lrp.csv',
                                        yolo_out_file='yolo_outputs_lrp.csv')
    print("Total number of images processed: ", nb_img_processed)
