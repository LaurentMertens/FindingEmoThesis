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

    def __init__(self, obj_importance_thres, emo_conf_thres, obj_conf_thres, ann_ambiguity_thres):
        self.emonet_ann_outputs = pd.read_csv('emonet_ann_outputs')
        self.yolo_outputs = pd.read_csv('yolo_outputs')
        self.obj_importance_thres = obj_importance_thres
        self.emo_conf_thres = emo_conf_thres
        self.obj_conf_thres = obj_conf_thres
        self.ann_ambiguity_thres = ann_ambiguity_thres
        self.proc_outputs = self.proc_df()

    def proc_df(self):
        #self.emonet_ann_outputs = self.emonet_ann_outputs.explode("ann_dec_factors")
        merged_df = pd.merge(self.emonet_ann_outputs, self.yolo_outputs,
                             on=["dir_image_path", "emonet_emotion", "emonet_emotion_conf"])
        df = merged_df[(merged_df["ann_ambiguity"] < self.ann_ambiguity_thres) &
                       (merged_df["emonet_emotion_conf"] > self.emo_conf_thres) &
                       (merged_df["object_confidence"] > self.obj_conf_thres) &
                       (merged_df["object_importance"] > self.obj_importance_thres)]
        df.to_csv("proc_outputs")
        return df

    def get_emo_obj_df(self):
        count_df = pd.crosstab(self.proc_outputs["emonet_emotion"], self.proc_outputs["detected_object"])

        return count_df

    def get_emo_ann_df(self):
        count_df = pd.crosstab(self.proc_outputs["emonet_emotion"], self.proc_outputs["ann_emotion"])

        return count_df

    def get_obj_ann_df(self):
        count_df = pd.crosstab(self.proc_outputs["ann_emotion"], self.proc_outputs["detected_object"])

        return count_df

    def get_emo_fact_df(self):
        count_df = pd.crosstab(self.proc_outputs["emonet_emotion"], self.proc_outputs["ann_dec_factors"])

        return count_df


    def get_aro_df(self):
        df = self.emonet_ann_outputs.drop(self.emonet_ann_outputs[
                                              (self.emonet_ann_outputs["ann_ambiguity"] < self.ann_ambiguity_thres) &
                                              (self.emonet_ann_outputs["emonet_emotion_conf"] > self.emo_conf_thres)].index)

        return df[["emonet_arousal", "ann_arousal"]]

    def get_val_df(self):
        df = self.emonet_ann_outputs.drop(self.emonet_ann_outputs[
                                              (self.emonet_ann_outputs["ann_ambiguity"] < self.ann_ambiguity_thres) &
                                              (self.emonet_ann_outputs["emonet_emotion_conf"] > self.emo_conf_thres)].index)

        return df[["emonet_valence", "ann_valence"]]

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

    def plot_scatter_size_plot(self, col1, col2):
        c = pd.crosstab(gs.proc_outputs[col1], gs.proc_outputs[col2]).stack().reset_index(name='C')
        c.plot.scatter(col1, col2, s=c.C * 10)

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

    def cramers_correlation_matrix(self):
        """
        Plot an association matrix using the Cramer's V method with for 4 nominal features
        """
        # select features only
        df = self.proc_df()[["detected_object", "emonet_emotion", "ann_emotion", "ann_dec_factors"]]
        # Convert you str columns to Category columns
        df = df.apply(
            lambda x: x.astype("category") if x.dtype == "O" else x)
        # Initialize a CamresV object using you pandas.DataFrame
        cramersv = am.CramersV(df)
        # will return a pairwise matrix filled with Cramer's V, where columns and index are
        # the categorical variables of the passed pandas.DataFrame
        print(cramersv.fit())

    def remove_obj_cat(self, obj):
        self.proc_outputs = self.proc_outputs[self.proc_outputs["detected_object" != obj]]

if __name__ == '__main__':
    gs = GlobalStatistics(obj_importance_thres=0.5, emo_conf_thres=0.5, obj_conf_thres=0.5,
                          ann_ambiguity_thres=4)

    # analysis 1 : detected object (Yolo & GradCam) vs predicted emotion (EmoNet)
    emo_obj_df = gs.get_emo_obj_df()
    cram_corr_emo_obj = sp.stats.contingency.association(emo_obj_df, 'cramer')
    chi_square_corr_emo_obj = sp.stats.chi2_contingency(emo_obj_df)

    # analysis 2 : predicted emotion (EmoNet) vs annotated emotion (ANN)
    emo_ann_df = gs.get_emo_ann_df()
    cram_corr_emo_ann = sp.stats.contingency.association(emo_ann_df, 'cramer')
    chi_square_corr_emo_ann = sp.stats.chi2_contingency(emo_ann_df)

    # analysis 3 : deciding factor (ANN) vs annotated emotion (ANN) (to compare with analysis 1)
    emo_fact_ann_df = gs.get_emo_fact_df()
    cram_corr_emo_fact = sp.stats.contingency.association(emo_fact_ann_df, 'cramer')
    chi_square_corr_emo_fact = sp.stats.chi2_contingency(emo_fact_ann_df)

    # analysis 4 : detected objects (Yolo) vs annotated emotions (ANN)
    obj_ann_df = gs.get_obj_ann_df()
    cram_corr_obj_ann = sp.stats.contingency.association(obj_ann_df, 'cramer')
    chi_square_corr_obj_ann = sp.stats.chi2_contingency(obj_ann_df)

    # analysis 5 : predicted valence (EmoNet) vs annotated valence (ANN)
    val_emonet_ann_df = gs.get_val_df()
    pears_val = sp.stats.pearsonr(val_emonet_ann_df["emonet_valence"], val_emonet_ann_df["ann_valence"])

    # analysis 6 : predicted arousal (EmoNet) vs annotated arousal (ANN)
    aro_emonet_ann_df = gs.get_aro_df()
    pears_aro = sp.stats.pearsonr(aro_emonet_ann_df["emonet_arousal"], aro_emonet_ann_df["ann_arousal"])

    # correlation matrix
    gs.cramers_correlation_matrix()
    print("emonet emotion vs detected object", cram_corr_emo_obj)
    print("emonet emotion vs ann emotion", cram_corr_emo_ann)
    print("emonet emotion vs ann factors", cram_corr_emo_fact)
    print("detected objects vs ann emotions", cram_corr_obj_ann)
    print("emonet valence vs ann valence", pears_val)
    print("emonet arousal vs ann arousal", pears_aro)
    print(gs.proc_outputs[["emonet_valence", "ann_valence", "emonet_arousal", "ann_arousal"]].corr(method="pearson"))

    gs.plot_scatter_size_plot("emonet_emotion", "detected_object")
    gs.plot_scatter_size_plot("emonet_emotion", "ann_emotion")
    plt.show()
