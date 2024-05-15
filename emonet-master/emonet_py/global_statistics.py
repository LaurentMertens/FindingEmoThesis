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
        self.total_outputs = self.merge_df()

    def merge_df(self):
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
        emo_obj_df = self.yolo_outputs[(self.yolo_outputs["emonet_emotion_conf"] > self.emo_conf_thres) &
                       (self.yolo_outputs["object_confidence"] > self.obj_conf_thres) &
                       (self.yolo_outputs["object_importance"] > self.obj_importance_thres)]

        return emo_obj_df[["emonet_emotion", "detected_object"]]

    def get_emo_ann_df(self):
        emo_ann_df = self.emonet_ann_outputs[(self.emonet_ann_outputs["ann_ambiguity"] < self.ann_ambiguity_thres) &
                       (self.emonet_ann_outputs["emonet_emotion_conf"] > self.emo_conf_thres)]

        return emo_ann_df[["emonet_emotion", "ann_emotion"]]

    def get_obj_ann_df(self):

        return self.total_outputs[["ann_emotion", "detected_object"]]

    def get_emo_fact_df(self):
        emo_ann_df = self.emonet_ann_outputs[(self.emonet_ann_outputs["ann_ambiguity"] < self.ann_ambiguity_thres) &
                       (self.emonet_ann_outputs["emonet_emotion_conf"] > self.emo_conf_thres)]

        return emo_ann_df[["ann_emotion", "ann_dec_factors"]]


    def get_aro_df(self):
        df = self.emonet_ann_outputs[(self.emonet_ann_outputs["ann_ambiguity"] < self.ann_ambiguity_thres) &
                       (self.emonet_ann_outputs["emonet_emotion_conf"] > self.emo_conf_thres)]

        return df[["emonet_arousal", "ann_arousal"]]

    def get_val_df(self):
        df = self.emonet_ann_outputs.drop(self.emonet_ann_outputs[
                                              (self.emonet_ann_outputs["ann_ambiguity"] < self.ann_ambiguity_thres) &
                                              (self.emonet_ann_outputs["emonet_emotion_conf"] > self.emo_conf_thres)].index)

        return df[["emonet_valence", "ann_valence"]]


    def plot_scatter_size_plot(self, df, col1, col2):
        c = pd.crosstab(df[col1], df[col2]).stack().reset_index(name='C')
        c.plot.scatter(col1, col2, s=c.C * 10, colormap="viridis")
        plt.xticks(rotation=90)
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
        df = self.total_outputs[["detected_object", "emonet_emotion", "ann_emotion", "ann_dec_factors"]]
        # Convert you str columns to Category columns
        df = df.apply(
            lambda x: x.astype("category") if x.dtype == "O" else x)
        # Initialize a CamresV object using you pandas.DataFrame
        cramersv = am.CramersV(df)
        # will return a pairwise matrix filled with Cramer's V, where columns and index are
        # the categorical variables of the passed pandas.DataFrame
        return cramersv.fit()

    def remove_obj_cat(self, obj):
        self.total_outputs = self.total_outputs[self.total_outputs["detected_object" != obj]]
        self.yolo_outputs = self.yolo_outputs[self.yolo_outputs["detected_object" != obj]]


if __name__ == '__main__':
    gs = GlobalStatistics(obj_importance_thres=0.5, emo_conf_thres=0.5, obj_conf_thres=0.5,
                          ann_ambiguity_thres=4)

    # analysis 1 : detected object (Yolo & GradCam) vs predicted emotion (EmoNet)
    emo_obj_df = gs.get_emo_obj_df()
    emo_obj_df_ct = pd.crosstab(emo_obj_df["emonet_emotion"], emo_obj_df["detected_object"])
    cram_corr_emo_obj = sp.stats.contingency.association(emo_obj_df_ct, 'cramer')
    chi_square_corr_emo_obj = sp.stats.chi2_contingency(emo_obj_df_ct)
    gs.plot_scatter_size_plot(emo_obj_df, "emonet_emotion", "detected_object")
    emo_obj_df_freq = pd.crosstab(emo_obj_df["detected_object"], emo_obj_df["emonet_emotion"], normalize='index')
    sns.heatmap(emo_obj_df_freq, annot=True, cmap='coolwarm')
    plt.tight_layout()
    plt.show()

    # analysis 2 : predicted emotion (EmoNet) vs annotated emotion (ANN)
    emo_ann_df = gs.get_emo_ann_df()
    emo_ann_df_ct = pd.crosstab(emo_ann_df["emonet_emotion"], emo_ann_df["ann_emotion"])
    cram_corr_emo_ann = sp.stats.contingency.association(emo_ann_df_ct, 'cramer')
    chi_square_corr_emo_ann = sp.stats.chi2_contingency(emo_ann_df_ct)
    gs.plot_scatter_size_plot(emo_ann_df, "emonet_emotion", "ann_emotion")
    emo_ann_df_freq = pd.crosstab(emo_ann_df["emonet_emotion"], emo_ann_df["ann_emotion"], normalize='index')
    sns.heatmap(emo_ann_df_freq, annot=True, cmap='coolwarm')
    plt.tight_layout()
    plt.show()

    # analysis 3 : detected objects (Yolo) vs annotated emotions (ANN)
    obj_ann_df = gs.get_obj_ann_df()
    obj_ann_df_ct = pd.crosstab(obj_ann_df["ann_emotion"], obj_ann_df["detected_object"])
    cram_corr_obj_ann = sp.stats.contingency.association(obj_ann_df_ct, 'cramer')
    chi_square_corr_obj_ann = sp.stats.chi2_contingency(obj_ann_df_ct)
    gs.plot_scatter_size_plot(obj_ann_df, "ann_emotion", "detected_object")
    obj_ann_df_freq = pd.crosstab(obj_ann_df["detected_object"], obj_ann_df["ann_emotion"], normalize='index')
    sns.heatmap(obj_ann_df_freq, annot=True, cmap='coolwarm')
    plt.tight_layout()
    plt.show()

    # analysis 4 : predicted valence (EmoNet) vs annotated valence (ANN)
    val_emonet_ann_df = gs.get_val_df()
    pears_val = sp.stats.pearsonr(val_emonet_ann_df["emonet_valence"], val_emonet_ann_df["ann_valence"])
    sns.scatterplot(val_emonet_ann_df, x=val_emonet_ann_df["emonet_valence"], y=val_emonet_ann_df["ann_valence"])
    sns.heatmap(val_emonet_ann_df.corr(method="pearson"), annot=True, cmap='coolwarm')
    plt.tight_layout()
    plt.show()

    # analysis 5 : predicted arousal (EmoNet) vs annotated arousal (ANN)
    aro_emonet_ann_df = gs.get_aro_df()
    pears_aro = sp.stats.pearsonr(aro_emonet_ann_df["emonet_arousal"], aro_emonet_ann_df["ann_arousal"])
    sns.scatterplot(aro_emonet_ann_df, x=aro_emonet_ann_df["emonet_arousal"], y=aro_emonet_ann_df["ann_arousal"])
    sns.heatmap(aro_emonet_ann_df.corr(method="pearson"), annot=True, cmap='coolwarm')
    plt.tight_layout()
    plt.show()

    # ( analysis  : deciding factor (ANN) vs annotated emotion (ANN) )
    emo_fact_ann_df = gs.get_emo_fact_df()
    emo_fact_ann_df_ct = pd.crosstab(emo_fact_ann_df["ann_dec_factors"], emo_fact_ann_df["ann_emotion"])
    cram_corr_emo_fact = sp.stats.contingency.association(emo_fact_ann_df_ct, 'cramer')
    chi_square_corr_emo_fact = sp.stats.chi2_contingency(emo_fact_ann_df_ct)

    # correlations
    sns.heatmap(gs.cramers_correlation_matrix(), annot=True, cmap='coolwarm')
    plt.tight_layout()
    plt.show()
    print("emonet emotion vs detected object", cram_corr_emo_obj)
    print("emonet emotion vs ann emotion", cram_corr_emo_ann)
    print("emonet emotion vs ann factors", cram_corr_emo_fact)
    print("detected objects vs ann emotions", cram_corr_obj_ann)
    print("emonet valence vs ann valence", pears_val)
    print("emonet arousal vs ann arousal", pears_aro)
    print(gs.total_outputs[["emonet_valence", "ann_valence", "emonet_arousal", "ann_arousal"]].corr(method="pearson"))

