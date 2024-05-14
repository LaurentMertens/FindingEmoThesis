import pandas as pd
import emonet
import matplotlib.pyplot as plt
import json
import copy
import seaborn

def save_dictionary(dico, name):
    with open(f"{name}.json", "w") as file:
        json.dump(dico, file)
    print(f'{name} saved successfully.')

class GlobalStatistics:

    def __init__(self, importance_thres, emo_conf_thres, obj_conf_thres, ann_ambiguity_thres, nb_rows_to_process=None):
        self.emonet_ann_ouputs = pd.read_csv('emonet_ann_outputs')
        self.yolo_outputs = pd.read_csv('yolo_outputs')
        self.emo_obj_dict = self.create_emo_obj_dict(importance_thres=importance_thres,
                                                     emo_conf_thres=emo_conf_thres, obj_conf_thres=obj_conf_thres,
                                                     nb_rows_to_process=nb_rows_to_process)
        self.emo_ann_dict = self.create_emo_ann_dict(ann_ambiguity_thres=ann_ambiguity_thres,
                                                     emo_conf_thres=emo_conf_thres, nb_rows_to_process=nb_rows_to_process)

    def create_emo_obj_dict(self, importance_thres, emo_conf_thres, obj_conf_thres, nb_rows_to_process=None):
        """
        Create a dictionary of a count of the detected objects (above a certain threshold of importance)
        per predicted emotion.
        """
        dict = {}
        n = 0
        # if number of rows not mentioned, whole dictionary is processed
        if nb_rows_to_process == None:
            nb_rows_to_process = len(self.yolo_outputs.index)
        for index, row in self.yolo_outputs.iterrows():
            if n < nb_rows_to_process:
                # only consider objects with importance higher than threshold
                if (row['object_importance'] > importance_thres and row['emotion_confidence'] > emo_conf_thres and
                        row['object_confidence'] > obj_conf_thres):
                    if row['emotion'] in dict:
                        if row['detected_object'] in dict[row['emotion']]:
                            dict[row['emotion']][row['detected_object']] += 1
                        else:
                            dict[row['emotion']][row['detected_object']] = 1
                    else:
                        dict[row['emotion']] = {row['detected_object']: 1}
                if n % 500 == 0:
                    print('Saving progress...')
                    save_dictionary(dict, 'objects_emotions_dict')
                    print('dictionary \'objects_emotions_dict\' saved...')
            n += 1

        return dict

    def create_emo_ann_dict(self, ann_ambiguity_thres, emo_conf_thres, nb_rows_to_process=None):
        """
        Create a dictionary of a count of the detected objects (above a certain threshold of importance)
        per predicted emotion.
        """
        dict = {}
        n = 0
        # if number of rows not mentioned, whole dictionary is processed
        if nb_rows_to_process == None:
            nb_rows_to_process = len(self.emonet_ann_ouputs.index)
        for index, row in self.emonet_ann_ouputs.iterrows():
            if n < nb_rows_to_process:
                # only consider objects with importance higher than threshold
                if row['ann_ambiguity'] < ann_ambiguity_thres and row['emotion_confidence'] > emo_conf_thres:
                    if row['ann_emotion'] in dict:
                        if row['detected_object'] in dict[row['ann_emotion']]:
                            dict[row['ann_emotion']][row['emonet_emotion']] += 1
                        else:
                            dict[row['ann_emotion']][row['emonet_emotion']] = 1
                    else:
                        dict[row['ann_emotion']] = {row['emonet_emotion']: 1}
                if n % 500 == 0:
                    print('Saving progress...')
                    save_dictionary(dict, 'objects_emotions_dict')
                    print('dictionary \'objects_emotions_dict\' saved...')
            n += 1

        return dict

    def plot_emo_obj_stats(self, data):
        """
        Plot the distributions of detected objects per emotion category from obj_emo dictionary.
        """
        # Create subplots
        fig, axs = plt.subplots(2, 3, figsize=(15, 10))
        axs = axs.ravel()
        fig.suptitle("relative frequency of objects per emotion")
        # Plotting pie charts for each emotion
        for i, (emotion, obj_count) in enumerate(data.items()):
            objects = list(obj_count.keys())
            count = list(obj_count.values())
            # Plotting
            axs[i].bar(objects, count, width=0.5, bottom=0, align='center')
            axs[i].set_title(f'{emotion}')
            # setting labels vertically
            axs[i].tick_params(axis='x', labelrotation=90)
            # set y scale between 0 and 1
            axs[i].set_ylim(0, 1)

        # Adjust layout
        plt.tight_layout()
        # Show plot
        plt.show()

    def plot_dict_heatmap(self, emo_obj_freq_dict):
        seaborn.heatmap(pd.DataFrame(emo_obj_freq_dict))
        plt.show()

    def get_emo_obj_freq(self, dico):
        """
        Print frequencies of detected objects per emotion category from the obj_emo dictionary.
        """
        emo_obj_freq_dict = copy.deepcopy(dico)
        for emotion in dico:
            n = 0
            for obj in dico[emotion]:
                n += dico[emotion][obj]
            print(emotion)
            for obj in dico[emotion]:
                print(f'{obj}: {(dico[emotion][obj] / n) * 100:0.1f}%')
                emo_obj_freq_dict[emotion][obj] = dico[emotion][obj] / n
            print("________________")
        return emo_obj_freq_dict

    def emonet_ann_analysis(self):
        """
        Analyze correlations between EmoNet predictions and FindingEmo annotations.
        """
        pass




if __name__ == '__main__':
    gs = GlobalStatistics(importance_thres=0.1, emo_conf_thres=0.5, obj_conf_thres=0.5,
                          ann_ambiguity_thres=4, nb_rows_to_process=None)
    emo_obj_dict = gs.emo_obj_dict
    emo_obj_freq_dict = gs.get_emo_obj_freq(emo_obj_dict)
    emo_ann_dict = gs.emo_ann_dict
    emo_ann_freq_dict = gs.get_emo_obj_freq(emo_ann_dict)
    gs.plot_dict_heatmap(emo_obj_freq_dict)
    gs.plot_dict_heatmap(emo_ann_dict)
