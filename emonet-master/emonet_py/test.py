from local_analysis import local_analysis
from explanations_emonet import explanations_emonet
from explanations_resnet50 import explanations_resnet50
from global_analysis import print_statistics
import os

emotions_objects_dict = {}
file_name = 'friends_parc.jpg'
file_path = '../data/images/'+file_name
file_name = os.path.splitext(file_name)[0]
detected_emotion = explanations_emonet(file_path, file_name, show_plot=False)
detected_objects = local_analysis(file_path, file_name, show_box=False)
print(detected_objects)
"""
for index, row in detected_objects.iterrows():
    if detected_emotion in emotions_objects_dict and row['class_label'] in emotions_objects_dict[detected_emotion]:
        emotions_objects_dict[detected_emotion][row['class_label']] += 1
    else:
        emotions_objects_dict[detected_emotion] = {row['class_label']: 1}

print(emotions_objects_dict)
print_statistics(emotions_objects_dict)
"""