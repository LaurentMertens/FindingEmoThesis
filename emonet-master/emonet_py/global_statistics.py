import pandas as pd
import emonet
import matplotlib.pyplot as plt
import json
import copy
import seaborn as sns
import scipy as sp
import association_metrics as am


def save_dictionary(dico, name):
    with open(f"{name}.json", "w") as file:
        json.dump(dico, file)
    print(f'{name} saved successfully.')

class GlobalStatistics:

    def __init__(self, importance_thres, emo_conf_thres, obj_conf_thres, ann_ambiguity_thres):
        self.emonet_ann_outputs = pd.read_csv('emonet_ann_outputs')
        self.yolo_outputs = pd.read_csv('yolo_outputs')
        self.importance_thres = importance_thres
        self.emo_conf_thres = emo_conf_thres
        self.obj_conf_thres = obj_conf_thres
        self.ann_ambiguity_thres = ann_ambiguity_thres

    def get_emo_obj_df(self):
        df = self.yolo_outputs.drop(self.yolo_outputs[self.yolo_outputs["object_confidence"] > self.obj_conf_thres and
                                                      self.yolo_outputs["object_importance"] > self.importance_thres and
                                                      self.yolo_outputs["emonet_max_emotion_prob"] > self.emo_conf_thres].index)
        count_df = pd.crosstab(df["emotion"], df["detected_object"])

        return count_df

    def get_emo_ann_df(self,):
        df = self.emonet_ann_ouputs.drop(self.emonet_ann_ouputs[
                                             self.emonet_ann_ouputs["ann_ambiguity"] > self.ann_ambiguity_thres and
                                             self.emonet_ann_ouputs["emonet_emotion_conf"] > self.emo_conf_thres].index)
        count_df = pd.crosstab(df["emonet_emotion"], df["ann_emotion"])

        return count_df

    def get_obj_ann_df(self,):
        merged_df = pd.merge(self.emonet_ann_ouputs, self.yolo_outputs, on='dir_image_path')
        df = merged_df.drop(merged_df[merged_df["ann_ambiguity"] > self.ann_ambiguity_thres and
                                      merged_df["emonet_emotion_conf"] > self.emo_conf_thres and
                                      merged_df["object_confidence"] > self.obj_conf_thres and
                                      merged_df["object_importance"] > self.importance_thres]print(.index)
        count_df = pd.crosstab(df["ann_emotion"], df["detected_object"])

        return count_df

    def get_emo_fact_df(self):
        df = self.emonet_ann_ouputs.drop(self.emonet_ann_ouputs[
                                             self.emonet_ann_ouputs["ann_ambiguity"] > self.ann_ambiguity_thres and
                                             self.emonet_ann_ouputs["emonet_emotion_conf"] > self.emo_conf_thres].index)
        count_df = pd.crosstab(df["emonet_emotion"], df["ann_emotion"])
        # STILL NEED TO SPLIT THE DEC_FACTORS
        return count_df


    def get_aro_df(self):
        df = self.emonet_ann_ouputs.drop(self.emonet_ann_ouputs[
                                             self.emonet_ann_ouputs["ann_ambiguity"] > self.ann_ambiguity_thres and
                                             self.emonet_ann_ouputs["emonet_emotion_conf"] > self.emo_conf_thres].index)

        return df[["emonet_arousal","ann_arousal"]]

    def get_val_df(self):
        df = self.emonet_ann_ouputs.drop(self.emonet_ann_ouputs[
                                             self.emonet_ann_ouputs["ann_ambiguity"] > self.ann_ambiguity_thres and
                                             self.emonet_ann_ouputs["emonet_emotion_conf"] > self.emo_conf_thres].index)

        return df[["emonet_valence","ann_valence"]]

    def plot_dict_bars(self, dict):
        """
        Plot the distributions of detected objects per emotion category from obj_emo dictionary.
        """
        df = pd.DataFrame(dict).transpose()
        ax = df.plot(kind='barh', legend=False, cmap='coolwarm')
        #ax.set_ylim([0, 1])
        # extract the legend labels
        handles, legend_labels = ax.get_legend_handles_labels()
        # iterate through the zipped containers and legend_labels
        for c, l in zip(ax.containers, legend_labels):
            # get labels
            labels = [l if (v.get_width()) > 0 else '' for v in c]
            # add the bar annotation
            ax.bar_label(c, labels=labels, label_type='edge', rotation='horizontal', fontsize=5)
        # set colors
        #for i, bar in enumerate(ax.patches):
        #    bar.set_color('b')

    def plot_heatmap(self, emo_obj_freq_dict):
        sns.heatmap(pd.DataFrame(emo_obj_freq_dict), annot=True, cmap='coolwarm')
        plt.show()

    def plot_scatter_size_plot(self, df):
        c = pd.crosstab(df.emotion, df.detected_object).stack().reset_index(name='C')
        c.plot.scatter('emotion', 'detected_object', s=c.C * 10)

    def plot_scatter(self, dict, x_legend, y_legend):
        fig, ax = plt.subplots()
        ax.scatter(dict.keys(), dict.values(), c="steelblue")
        ax.set_xlabel(x_legend), ax.set_ylabel(y_legend)
        plt.tight_layout()
        plt.show()


    def get_dict_freq(self, dico):
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

def cramers_correlation_matrix(df):
    """
    The Association matrix using the Cramer's V method. (only two variables normally so using a matrix is debatable)
    """
    # Convert you str columns to Category columns
    df = df.apply(
        lambda x: x.astype("category") if x.dtype == "O" else x)
    # Initialize a CamresV object using you pandas.DataFrame
    cramersv = am.CramersV(df)
    # will return a pairwise matrix filled with Cramer's V, where columns and index are
    # the categorical variables of the passed pandas.DataFrame
    cramersv.fit()


def remove_obj_cat(emo_obj_freq_dict, category):
    for emo in emo_obj_freq_dict:
        emo_obj_freq_dict[emo].pop(category)
    return emo_obj_freq_dict

if __name__ == '__main__':
    gs = GlobalStatistics(importance_thres=0.1, emo_conf_thres=0.5, obj_conf_thres=0.3,
                          ann_ambiguity_thres=4, nb_rows_to_process=None)
    # analysis 1 : detected objects (Yolo & GradCam) vs predicted emotions (EmoNet)
    emo_obj_df = gs.get_emo_obj_df()
    #cram_corr_emo_obj = sp.stats.contingency.association(emo_obj_df, 'cramer'))
    #chi_square_corr_emo_obj = sp.states.chi2_contingency(emo_obj_df)

    # analysis 2 : predicted emotion (EmoNet) vs annotated emotion (ANN)
    emo_ann_df = gs.get_emo_ann_df()

    # analysis 3 : predicted valence (EmoNet) vs annotated valence (ANN)
    val_emonet_ann_df = gs.get_val_df()
    #sp.stats.pearsonr(val_emonet_ann_df["emonet_valence"], val_emonet_ann_df["ann_valence"])

    # analysis 4 : predicted arousal (EmoNet) vs annotated arousal (ANN)
    aro_emonet_ann_df = gs.get_aro_df()
    #sp.stats.pearsonr(aro_emonet_ann_df["emonet_arousal"], aro_emonet_ann_df["ann_arousal"])

    # analysis 5 : annotated emotion (ANN) vs annotated deciding factors (ANN) (to compare with analysis 1)
    emo_fact_ann_df = gs.get_emo_fact_df()
    #cram_corr_emo_fact = sp.stats.contingency.association(emo_fact_ann_df, 'cramer'))

    # analysis 6 : detected objects (Yolo) vs annotated emotions (ANN)
    obj_ann_df = gs.get_obj_ann_df()
    #cram_corr_emo_fact = sp.stats.contingency.association(obj_ann_df, 'cramer'))
