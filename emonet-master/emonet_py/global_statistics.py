import pandas as pd
import emonet
import matplotlib.pyplot as plt
import json
import copy
import seaborn as sns
import scipy as sp


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
        self.val_emonet_ann_dict = self.create_val_emonet_ann_dict(ann_ambiguity_thres=ann_ambiguity_thres,
                                                                   emo_conf_thres=emo_conf_thres, nb_rows_to_process=nb_rows_to_process)
        self.aro_emonet_ann_dict = self.create_aro_emonet_ann_dict(ann_ambiguity_thres=ann_ambiguity_thres,
                                                                   emo_conf_thres=emo_conf_thres, nb_rows_to_process=nb_rows_to_process)
        self.emo_dec_fact_ann_dict = self.create_emo_dec_fact_ann_dict(ann_ambiguity_thres=ann_ambiguity_thres,
                                                                       nb_rows_to_process=nb_rows_to_process)

    def create_emo_obj_dict(self, importance_thres, emo_conf_thres, obj_conf_thres, nb_rows_to_process=None):
        """
        Create a dictionary of the number and type of objects (above a certain threshold of importance)
        detected per predicted emotion.
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
                    save_dictionary(dict, 'emo_obj_dict')
                    print('dictionary \'emo_obj_dict\' saved...')
            n += 1

        return dict

    def create_emo_ann_dict(self, ann_ambiguity_thres, emo_conf_thres, nb_rows_to_process=None):
        """
        Create a dictionary of the number and categories of emotions predicted by EmoNet per emotion
        annotated in FindingEmo.
        """
        dict = {}
        n = 0
        # if number of rows not mentioned, whole dictionary is processed
        if nb_rows_to_process == None:
            nb_rows_to_process = len(self.emonet_ann_ouputs.index)
        for index, row in self.emonet_ann_ouputs.iterrows():
            if n < nb_rows_to_process:
                # only consider objects with importance higher than threshold
                if row['ann_ambiguity'] < ann_ambiguity_thres and row['emonet_max_emotion_prob'] > emo_conf_thres:
                    if row['ann_emotion'] in dict:
                        if row['emonet_max_emotion'] in dict[row['ann_emotion']]:
                            dict[row['ann_emotion']][row['emonet_max_emotion']] += 1
                        else:
                            dict[row['ann_emotion']][row['emonet_max_emotion']] = 1
                    else:
                        dict[row['ann_emotion']] = {row['emonet_max_emotion']: 1}
                if n % 500 == 0:
                    print('Saving progress...')
                    save_dictionary(dict, 'emo_ann_dict')
                    print('dictionary \'emo_ann_dict\' saved...')
            n += 1

        return dict

    def create_emo_dec_fact_ann_dict(self, ann_ambiguity_thres, nb_rows_to_process=None):
        """
        Create a dictionary of the annotated deciding factors per annotated emotion in FindingEmo.
        """
        dict = {}
        n = 0
        # if number of rows not mentioned, whole dictionary is processed
        if nb_rows_to_process == None:
            nb_rows_to_process = len(self.emonet_ann_ouputs.index)
        for index, row in self.emonet_ann_ouputs.iterrows():
            if n < nb_rows_to_process:
                # only consider annotated emotions below the ambiguity threshold
                if row['ann_ambiguity'] < ann_ambiguity_thres:
                    if row['ann_emotion'] in dict:
                        for dec_fact in row['ann_dec_factors'].split(','):
                            if dec_fact in dict[row['ann_emotion']]:
                                dict[row['ann_emotion']][dec_fact] += 1
                            else:
                                dict[row['ann_emotion']][dec_fact] = 1
                    else:
                        for dec_fact in row['ann_dec_factors'].split(','):
                            dict[row['ann_emotion']] = {dec_fact: 1}
                if n % 500 == 0:
                    print('Saving progress...')
                    save_dictionary(dict, 'emo_dec_fact_ann_dict')
                    print('dictionary \'emo_dec_fact_ann_dict\' saved...')
            n += 1

        return dict


    def create_aro_emonet_ann_dict(self, ann_ambiguity_thres, emo_conf_thres, nb_rows_to_process=None):
        """
        Create a dictionary of the arousal predicted by EmoNet vs the arousal annotated in FindingEmo
        for each image.
        """
        dict = {}
        n = 0
        # if number of rows not mentioned, whole dictionary is processed
        if nb_rows_to_process == None:
            nb_rows_to_process = len(self.emonet_ann_ouputs.index)
        for index, row in self.emonet_ann_ouputs.iterrows():
            if n < nb_rows_to_process:
                # only consider objects with importance higher than threshold
                if row['ann_ambiguity'] < ann_ambiguity_thres and row['emonet_max_emotion_prob'] > emo_conf_thres:
                    dict[row['emonet_arousal']] = row['ann_arousal']
                if n % 500 == 0:
                    print('Saving progress...')
                    save_dictionary(dict, 'aro_emonet_ann_dict')
                    print('dictionary \'aro_emonet_ann_dict\' saved...')
            n += 1

        return dict

    def create_val_emonet_ann_dict(self, ann_ambiguity_thres, emo_conf_thres, nb_rows_to_process=None):
        """
        Create a dictionary of the valence predicted by EmoNet vs the valence annotated in FindingEmo
        for each image.
        """
        dict = {}
        n = 0
        # if number of rows not mentioned, whole dictionary is processed
        if nb_rows_to_process == None:
            nb_rows_to_process = len(self.emonet_ann_ouputs.index)
        for index, row in self.emonet_ann_ouputs.iterrows():
            if n < nb_rows_to_process:
                # only consider objects with importance higher than threshold
                if row['ann_ambiguity'] < ann_ambiguity_thres and row['emonet_max_emotion_prob'] > emo_conf_thres:
                    dict[row['emonet_valence']] = row['ann_valence']
                if n % 500 == 0:
                    print('Saving progress...')
                    save_dictionary(dict, 'val_emonet_ann_dict')
                    print('dictionary \'val_emonet_ann_dict\' saved...')
            n += 1

        return dict

    def plot_dict_bars(self, dict):
        """
        Plot the distributions of detected objects per emotion category from obj_emo dictionary.
        """
        df = pd.DataFrame(dict).transpose()
        ax = df.plot(kind='bar', legend="False")
        # extract the legend labels
        handles, legend_labels = ax.get_legend_handles_labels()
        # iterate through the zipped containers and legend_labels
        for c, l in zip(ax.containers, legend_labels):
            # customize the labels: only plot values greater than 0 and append the legend label
            labels = [f'{l}: {w * 100:0.0f}%' if (w := v.get_width()) > 0 else '' for v in c]
            # add the bar annotation
            ax.bar_label(c, labels=labels, label_type='center')
        plt.show()

    def plot_dict_heatmap(self, emo_obj_freq_dict):
        sns.heatmap(pd.DataFrame(emo_obj_freq_dict), annot=True, cmap='coolwarm')
        plt.show()

    def plot_dict_scatter(self, dict):
        plt.scatter(dict.keys(), dict.values())
        plt.show()


    def get_emo_obj_freq(self, dico):
        """
        Print and return frequencies of detected objects per emotion category from the obj_emo dictionary.
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



if __name__ == '__main__':
    gs = GlobalStatistics(importance_thres=0.1, emo_conf_thres=0.5, obj_conf_thres=0.5,
                          ann_ambiguity_thres=4, nb_rows_to_process=None)
    # analysis 1 : detected objects (Yolo & GradCam) vs predicted emotions (EmoNet)
    emo_obj_dict = gs.emo_obj_dict
    emo_obj_freq_dict = gs.get_emo_obj_freq(emo_obj_dict)
    gs.plot_dict_bars(emo_obj_freq_dict)

    # analysis 2 : predicted emotion (EmoNet) vs annotated emotion (ANN)
    emo_ann_dict = gs.emo_ann_dict
    emo_ann_freq_dict = gs.get_emo_obj_freq(emo_ann_dict)
    gs.plot_dict_heatmap(emo_ann_freq_dict)

    # analysis 3 : predicted valence (EmoNet) vs annotated valence (ANN)
    val_emonet_ann_dict = gs.val_emonet_ann_dict
    gs.plot_dict_scatter(val_emonet_ann_dict)
    print(sp.stats.pearsonr(list(val_emonet_ann_dict.keys()), list(val_emonet_ann_dict.values())))

    # analysis 4 : predicted arousal (EmoNet) vs annotated arousal (ANN)
    aro_emonet_ann_dict = gs.aro_emonet_ann_dict
    gs.plot_dict_scatter(aro_emonet_ann_dict)
    print(sp.stats.pearsonr(list(aro_emonet_ann_dict.keys()), list(aro_emonet_ann_dict.values())))

    # analysis 5 : annotated emotion (ANN) vs annotated deciding factors (ANN) (to compare with analysis 1)
    emo_dec_fact_ann_dict = gs.emo_dec_fact_ann_dict
    gs.plot_dict_bars(emo_obj_freq_dict)
