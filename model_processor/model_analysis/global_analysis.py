"""

.. codeauthor:: Laurent Mertens <laurent.mertens@kuleuven.be>
"""
import os

import PIL
import pandas as pd
import torch
import torchvision
from torchvision import transforms

from config import Config
from emonet_py.explanations_model import ExplanationsModel
from model_processor.preprocessing.img_resize_preproc import ImgResize
from emonet_py.local_analysis import LocalAnalysis
from model_processor.preprocessing.imagenet_preproc import ImageNetPreProcess
from model_processor.imagenet.load_buffered_imagenet import LoadBufferedImageNet

# For debug purposes:
import matplotlib
matplotlib.use('pdf')


def get_dir_image_path(file_path):
    return os.path.basename(os.path.dirname(file_path)) + '/' + os.path.basename(file_path)


class GlobalAnalysis:
    """
    GlobalAnalysis class used to apply the EmoNet and Yolov3 pipeline and extract all outputs in .csv format.
    """

    def __init__(self, model, model_pp, class_labels, target_layers, device=torch.device('cpu'),
                 out_model_pred_file=None, out_yolo_file=None):
        self.device = device
        self.local_analysis = LocalAnalysis(device=device)
        # self.expl_emo = ExplanationsEmonet(device=device)
        self.expl_emo = ExplanationsModel(model=model, model_pp=model_pp, class_labels=class_labels,
                                          target_layers=target_layers, device=device)
        self.nb_classes = len(class_labels)

        # Header for CSV output file
        self.df_model_header = ['dir_image_path', 'max_emotion', 'max_conf'] + [f'{x}_conf' for x in class_labels]

        self.out_model_pred_file = os.path.join(Config.DIR_OUTPUT, 'model_preds.csv') if\
            out_model_pred_file is None else out_model_pred_file
        self.out_yolo_file = os.path.join(Config.DIR_OUTPUT, 'yolo_output.csv') if\
            out_yolo_file is None else out_yolo_file

    def update_model_df(self, file_path, df_model):
        """
        Update the EmoNet output dataframe with the analysis of a new image.
        """
        # define paths and names
        image_name = os.path.basename(file_path)
        # define dir_image_path as part of the path containing file and its folder
        dir_image_path = get_dir_image_path(file_path)
        # get outputs of EmoNet
        max_emotion, max_prob, model_probs, cam_outputs =\
            self.expl_emo.get_explanations_for_image(file_path, image_name)
        # first elements of the new row are the image path and predicted emotion
        new_row = [dir_image_path, max_emotion, max_prob]
        # append probabilities for the output tensor emotion
        for emo_idx in range(self.nb_classes):
            new_row.append(model_probs[emo_idx].item())
        # add the row to the dataframe
        df_model.loc[len(df_model)] = new_row
        return df_model, max_emotion, max_prob, cam_outputs

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

    def process_model(self, img_dir):
        """
        save to two .csv files containing the outputs of EmoNet (and FindingEmo annotations) and Yolov3 respectively

        :param img_dir: root path containing the images to process
        :return:
        """
        count_images = 0
        # create EmoNet output dataframe with header
        df_model_preds = pd.DataFrame(columns=self.df_model_header)
        # create Yolo output dataframe
        df_yolo = pd.DataFrame()

        # Get image paths
        image_paths = ga.get_image_paths(img_dir)
        nb_images = len(image_paths)

        annotations = pd.read_csv(Config.FILE_ANN3_SINGLE)
        ann_files = set(annotations['image_path'])
        for image_path in image_paths:
            # Check image is annotated image
            rel_path = '/' + '/'.join(image_path.split('/')[-3:])
            if rel_path not in ann_files:
                print(f"Skipping image {rel_path}")
                continue

            # Let's be sure
            if not image_path.endswith(('.jpg', '.jpeg', '.png')):
                raise ValueError(f"Expected image file of type jpg/jpeg/png, got something else instead.\n[{image_path}]")
            # raising exceptions for corrupted files
            try:
                print("Processing:", image_path)

                # update emonet dataframe with outputs from new image
                df_model_preds, max_emotion, max_prob, cam_outputs = self.update_model_df(image_path, df_model_preds)

                if len(cam_outputs) > 1:
                    raise ValueError("Don't know how to handle multiple CAM outputs for now.\n"
                                     "Please make sure you're only using one CAM method at a time.")

                # update yolo dataframe with outputs from new image
                df_yolo = self.update_yolo_df(image_path, max_emotion, max_prob, df_yolo, cam_outputs[0])

                count_images += 1
                print(f"Progression: {(count_images / nb_images) * 100:0.2f}%")
            except PIL.UnidentifiedImageError as e:
                print(f'Error processing: {image_path}:\n\t{e}')
                continue
            if count_images % 200 == 0:
                print('Saving progress...')
                df_model_preds.to_csv(self.out_model_pred_file)
                df_yolo.to_csv(self.out_yolo_file)
                print(f'Model predictions and yolo output saved to [{self.out_model_pred_file}] and '
                      f'[{self.out_yolo_file}]...')

        # save emonet_ann outputs dataframe as .csv
        df_model_preds.to_csv(self.out_model_pred_file)
        # safe yolo outputs dataframe as .csv
        df_yolo.to_csv(self.out_yolo_file)
        print(f'Model predictions and yolo output saved to [{self.out_model_pred_file}] and '
              f'[{self.out_yolo_file}]...')

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
    # Load model and preprocessing
    _saved_weights = os.path.join(os.path.expanduser('~'),
                                 'Work', 'Projects', 'NeuroANN', 'Data', 'Models',
                                 'BestModels', '20231222',
                                 'emo8_vgg16_lr=0.001_loss=UnbalancedCrossEntropyLoss_sc=test_cuda.pth')

    _file_pp = os.path.join(os.path.expanduser('~'),
                           'Work', 'Projects', 'NeuroANN', 'Data', 'BufferedFeatures', 'Buffer_ImageNet',
                           'transform_cuda.dill')

    _model, _ = LoadBufferedImageNet.load(model=torchvision.models.vgg16,
                                            weights_file=_saved_weights)
    _pp = transforms.Compose([ImgResize(width=800, height=600),
                              ImageNetPreProcess(chain_type=ImageNetPreProcess.NORMALIZE)])

    _target_layers = [_model.features[28]]

    _class_labels = ['joy', 'trust', 'fear', 'surprise', 'sadness', 'disgust', 'anger', 'anticipation']

    # instantiation
    ga = GlobalAnalysis(model=_model, model_pp=_pp, class_labels=_class_labels,
                        target_layers=_target_layers, device=torch.device('cuda'))

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
    nb_img_processed = ga.process_model(directory_path)
    print("Total number of images processed: ", nb_img_processed)
