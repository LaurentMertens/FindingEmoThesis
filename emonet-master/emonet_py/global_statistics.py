import pandas as pd
import emonet
import matplotlib.pyplot as plt

class GlobalStatistics:

    def __init__(self):
        self.emonet_ann_ouputs = pd.read_csv('emonet_ann_outputs')
        self.yolo_outputs = pd.read_csv('yolo_outputs')


    def create_objects_emotions_dict(self, importance_threshold, nb_rows_to_process=None):
        """
        Create a dictionary of a count of the detected objects (above a certain threshold of importance)
        per predicted emotion.
        """
        dict = {}
        n = 0
        if nb_rows_to_process == None:
            nb_rows_to_process = len(self.yolo_outputs.index)
        for index, row in self.yolo_outputs.iterrows():
            if n < nb_rows_to_process:
                if row['importance'] > importance_threshold:
                    if row['emotion'] in dict:
                        if row['detected_object'] in dict[row['emotion']]:
                            dict[row['emotion']][row['detected_object']] += 1
                        else:
                            dict[row['emotion']][row['detected_object']] = 0
                    else:
                        dict[row['emotion']] = {row['detected_object']: 1}
            n += 1

        return dict

    def plot_obj_emo_stats(self, data):
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

    def print_obj_emo_stats(self, dico):
        for emotion in dico:
            n = 0
            for obj in dico[emotion]:
                n += dico[emotion][obj]
            print(emotion)
            for obj in dico[emotion]:
                print(f'{obj}: {(dico[emotion][obj] / n) * 100:0.1f}%')
            print("________________")

    def emonet_ann_analysis(self):
        """
        Analyze correlations between EmoNet predictions and FindingEmo annotations.
        """
        pass




if __name__ == '__main__':
    gs = GlobalStatistics()
    emo_obj_dict = gs.create_objects_emotions_dict(importance_threshold=0.1, nb_rows_to_process=100)
    print(emo_obj_dict)
    gs.print_obj_emo_stats(emo_obj_dict)
    gs.plot_obj_emo_stats(emo_obj_dict)
