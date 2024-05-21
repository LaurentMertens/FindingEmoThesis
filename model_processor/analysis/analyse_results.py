"""

.. codeauthor:: Laurent Mertens <laurent.mertens@kuleuven.be>
"""
import os

import dill
import numpy as np
import scipy

import pandas as pd

from config import Config


if __name__ == '__main__':
    model_preds = os.path.join(Config.DIR_OUTPUT, 'model_preds.csv')
    yolo_output = os.path.join(Config.DIR_OUTPUT, 'yolo_output.csv')

    csv_preds = pd.read_csv(model_preds, index_col=0)
    csv_yolo = pd.read_csv(yolo_output, index_col=0)

    print(csv_yolo.columns)

    img_paths = set(csv_preds['dir_image_path'].values)
    nb_images = len(img_paths)

    yolo_objects = sorted(csv_yolo['detected_object'].unique())
    nb_yolo_objects = len(yolo_objects)

    emos = sorted(csv_preds.max_emotion.unique())
    nb_emos = len(emos)

    idxs_max_emo = np.zeros((nb_images,))
    corr_mtx = np.zeros((nb_images, nb_yolo_objects))
    max_imgs = -1
    for idx, img_path in enumerate(img_paths):
        if (idx+1)% 100 == 0:
            print(f"\rAt img {idx+1}/{len(img_paths)}...", end='', flush=True)

        img_preds = csv_preds[csv_preds['dir_image_path'] == img_path]
        max_emo = img_preds.iloc[0].max_emotion

        img_yolo = csv_yolo[(csv_yolo.dir_image_path == img_path) & (csv_yolo.object_importance > 0.25)]
        img_yolo_objects = img_yolo['detected_object'].values

        idx_emo = emos.index(max_emo)
        for obj in img_yolo_objects:
            idx_object = yolo_objects.index(obj)
            corr_mtx[idx, idx_object] += 1
        idxs_max_emo[idx] = idx_emo

        if (idx + 1) == max_imgs:
            break
    print()

    dill.dump((corr_mtx, idxs_max_emo), open('tuple.dill', 'wb'))

    print(corr_mtx)
    print(idxs_max_emo)

    #
    # print(conf_mtx)
    # print(scipy.stats.pearsonr(corr_face_joy[0], corr_face_joy[1]))
    # print(scipy.stats.cov(corr_face_joy[0], corr_face_joy[1]))
