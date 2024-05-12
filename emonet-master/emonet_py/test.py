import os.path

import explanations_emonet, global_analysis, local_analysis, emonet
import pandas as pd

header = ['image_path', 'emonet_adoration_prob', 'emonet_aesthetic_appreciation_prob', 'emonet_amusement_prob',
          'emonet_anxiety_prob', 'emonet_awe_prob', 'emonet_boredom_prob', 'emonet_confusion_prob',
          'emonet_craving_prob',
          'emonet_disgust_prob', 'emonet_empathetic_pain_prob', 'emonet_entrancement_prob',
          'emonet_excitement_prob',
          'emonet_fear_prob', 'emonet_horror_prob', 'emonet_interest_prob', 'emonet_joy_prob',
          'emonet_romance_prob',
          'emonet_sadness_prob', 'emonet_sexual_desire_prob', 'emonet_surprise_prob', 'emonet_valence',
          'emonet_arousal']

def import_annotations():
    df_annotations = pd.read_csv('annotations_single.ann')
    df_annotations = df_annotations.rename(columns={'user': 'ann_user', 'image_path': 'ann_original_image_path',
                                                    'reject': 'ann_reject', 'tag': 'age_group',
                                                    'valence': 'ann_valence', 'arousal': 'ann_arousal',
                                                    'emotion': 'ann_emotion', 'dec_factors': 'ann_dec_factors',
                                                    'ambiguity': 'ann_ambiguity', 'fmri_candidate': 'ann_fmri_candidate',
                                                    'datetime': 'ann_datetime'})
    df_annotations['dir_image_path'] = df_annotations['ann_original_image_path'].apply(get_dir_image_path)
    return df_annotations

def get_dir_image_path(file_path):
    return os.path.basename(os.path.dirname(file_path)) + '/' + os.path.basename(file_path)


path = '/Users/youssefdoulfoukar/Desktop/Thesis/PytorchProject/emonet-master/emonet_py'

import_annotations().to_csv('ann_test')

print(get_dir_image_path(path))