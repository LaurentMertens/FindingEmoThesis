import pandas as pd
import emonet
import matplotlib.pyplot as plt
import json
import copy
import seaborn as sns
import scipy as sp
import association_metrics as am
import os

def get_dir_image_path(file_path):
    return os.path.basename(os.path.dirname(file_path)) + '/' + os.path.basename(file_path)

def get_annotations_df():
    """
    Merge the FindingEmo annotations with the outputs of EmoNet.
    """
    df_annotations = pd.read_csv('annotations_single.ann')
    # modify annotations header to distinguish from EmoNet outputs
    df_annotations = df_annotations.rename(columns={'user': 'ann_user', 'image_path': 'ann_original_image_path',
                                                    'reject': 'ann_reject', 'age': 'age_group',
                                                    'valence': 'ann_valence', 'arousal': 'ann_arousal',
                                                    'emotion': 'ann_emotion', 'dec_factors': 'ann_dec_factors',
                                                    'ambiguity': 'ann_ambiguity',
                                                    'fmri_candidate': 'ann_fmri_candidate',
                                                    'datetime': 'ann_datetime'})
    # add 'dir_image_path' as path containing only folder name and file name
    df_annotations['dir_image_path'] = df_annotations['ann_original_image_path'].apply(get_dir_image_path)
    return df_annotations

def save_dictionary(dico, name):
    with open(f"{name}.json", "w") as file:
        json.dump(dico, file)
    print(f'{name} saved successfully.')

class GlobalStatistics:

    def __init__(self, obj_importance_thres, emo_conf_thres, obj_conf_thres, ann_ambiguity_thres):
        self.emonet_outputs = pd.read_csv('emonet_outputs')
        self.yolo_outputs = pd.read_csv('yolo_outputs')
        self.ann = get_annotations_df()
        self.obj_importance_thres = obj_importance_thres
        self.emo_conf_thres = emo_conf_thres
        self.obj_conf_thres = obj_conf_thres
        self.ann_ambiguity_thres = ann_ambiguity_thres
        self.emonet_ann_outputs, self.yolo_ann_outputs = self.merge_df()

    def merge_df(self):
        #self.emonet_ann_outputs = self.emonet_ann_outputs.explode("ann_dec_factors")
        emonet_ann_outputs = pd.merge(self.emonet_outputs, self.ann,
                                      on=["dir_image_path"], how='left')
        yolo_ann_outputs = pd.merge(self.yolo_outputs, self.ann,
                                    on=["dir_image_path"], how='left')

        return emonet_ann_outputs, yolo_ann_outputs


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
        obj_ann_df = self.yolo_ann_outputs[(self.yolo_ann_outputs["ann_ambiguity"] < self.ann_ambiguity_thres) &
                                           (self.yolo_ann_outputs["object_confidence"] > self.obj_conf_thres) &
                                           (self.yolo_ann_outputs["object_importance"] > self.obj_importance_thres)]

        return obj_ann_df[["ann_emotion", "detected_object"]]

    def get_emo_fact_df(self):
        emo_ann_df = self.emonet_ann_outputs[(self.emonet_ann_outputs["ann_ambiguity"] < self.ann_ambiguity_thres) &
                                             (self.emonet_ann_outputs["emonet_emotion_conf"] > self.emo_conf_thres)]

        return emo_ann_df[["ann_emotion", "ann_dec_factors"]]


    def get_aro_df(self):
        df = self.emonet_ann_outputs[(self.emonet_ann_outputs["ann_ambiguity"] < self.ann_ambiguity_thres) &
                                     (self.emonet_ann_outputs["emonet_emotion_conf"] > self.emo_conf_thres)]

        return df[["emonet_arousal", "ann_arousal"]]

    def get_val_df(self):
        df = self.emonet_ann_outputs[(self.emonet_ann_outputs["ann_ambiguity"] < self.ann_ambiguity_thres) &
                                     (self.emonet_ann_outputs["emonet_emotion_conf"] > self.emo_conf_thres)]

        return df[["emonet_valence", "ann_valence"]]


    def plot_scatter_size_plot(self, df, col1, col2):
        c = pd.crosstab(df[col1], df[col2]).stack().reset_index(name='C')
        c.plot.scatter(col1, col2, s=c.C, colormap="viridis")
        plt.xticks(rotation=90)
        plt.tight_layout()
        plt.show()


    def cramers_correlation_matrix(self):
        """
        Plot an association matrix using the Cramer's V method
        """
        # select features only
        df = self.yolo_ann_outputs[["detected_object", "emonet_emotion", "ann_emotion", "ann_dec_factors"]]
        # Convert you str columns to Category columns
        df = df.apply(
            lambda x: x.astype("category") if x.dtype == "O" else x)
        # Initialize a CramersV object using pandas.DataFrame
        cramersv = am.CramersV(df)
        # will return a pairwise matrix filled with Cramer's V, where columns and index are
        # the categorical variables of the passed pandas.DataFrame
        return cramersv.fit()

    def remove_obj_cat(self, obj):
        self.yolo_outputs = self.yolo_outputs[self.yolo_outputs["detected_object"] != obj]
        self.yolo_ann_outputs = self.yolo_ann_outputs[self.yolo_ann_outputs["detected_object"] != obj]


    def remove_obj_outliers(self, count):
        """
        Removes the detected objects occurring less than 'count' times
        """
        v = self.yolo_outputs.detected_object.value_counts()
        self.yolo_outputs = self.yolo_outputs[self.yolo_outputs.detected_object.isin(v.index[v.gt(count)])]
        w = self.yolo_ann_outputs.detected_object.value_counts()
        self.yolo_ann_outputs = self.yolo_ann_outputs[self.yolo_ann_outputs.detected_object.isin(w.index[w.gt(count)])]

def analysis_obj_emo(gs):
    # analysis 1 : detected object (Yolo & GradCam) vs predicted emotion (EmoNet)
    obj_emo_df = gs.get_emo_obj_df()
    # get count table
    obj_emo_df_ct = pd.crosstab(obj_emo_df["emonet_emotion"], obj_emo_df["detected_object"])
    # get cramer's v correlation measure for nominal variables (significance test)
    cram_corr_emo_obj = sp.stats.contingency.association(obj_emo_df_ct, 'cramer')
    # get chi2 correlation measure (strength of association)
    chi_square_corr_emo_obj = sp.stats.chi2_contingency(obj_emo_df_ct)
    gs.plot_scatter_size_plot(obj_emo_df, "emonet_emotion", "detected_object")
    print("emonet emotion vs detected object corr: ", cram_corr_emo_obj)

    plt.tight_layout()
    plt.show()



def analysis_emo_ann(gs):
    # analysis 2 : predicted emotion (EmoNet) vs annotated emotion (ANN)
    emo_ann_df = gs.get_emo_ann_df()
    emo_ann_df_ct = pd.crosstab(emo_ann_df["emonet_emotion"], emo_ann_df["ann_emotion"])
    cram_corr_emo_ann = sp.stats.contingency.association(emo_ann_df_ct, 'cramer')
    chi_square_corr_emo_ann = sp.stats.chi2_contingency(emo_ann_df_ct)
    gs.plot_scatter_size_plot(emo_ann_df, "emonet_emotion", "ann_emotion")
    print("emonet emotion vs ann emotion corr: ", cram_corr_emo_ann)

    plt.tight_layout()
    plt.show()



def analysis_obj_ann(gs):
    # analysis 3 : detected objects (Yolo) vs annotated emotions (ANN)
    obj_ann_df = gs.get_obj_ann_df()
    # get the count table
    obj_ann_df_ct = pd.crosstab(obj_ann_df["ann_emotion"], obj_ann_df["detected_object"])
    # get cramer's v correlation measure for nominal variables (significance test)
    cram_corr_obj_ann = sp.stats.contingency.association(obj_ann_df_ct, 'cramer')
    # get chi2 correlation measure (strength of association)
    chi_square_corr_obj_ann = sp.stats.chi2_contingency(obj_ann_df_ct)
    # plotting
    gs.plot_scatter_size_plot(obj_ann_df, "ann_emotion", "detected_object")
    plt.tight_layout()
    plt.show()

    # cramers correlation
    print("detected objects vs ann emotions corr: ", cram_corr_obj_ann)

    plt.tight_layout()
    plt.show()



def analysis_valence(gs):
    # analysis 4 : predicted valence (EmoNet) vs annotated valence (ANN)
    val_emonet_ann_df = gs.get_val_df()
    # get pearson correlation measure
    pears_val = sp.stats.pearsonr(val_emonet_ann_df["emonet_valence"], val_emonet_ann_df["ann_valence"])
    print("emonet valence vs ann valence corr: ", pears_val)
    # plotting
    sns.scatterplot(val_emonet_ann_df, x=val_emonet_ann_df["emonet_valence"], y=val_emonet_ann_df["ann_valence"])
    plt.tight_layout()
    plt.show()

    # plot correlation matrix (one coefficient only here)
    sns.heatmap(val_emonet_ann_df.corr(method="pearson"), annot=True, cmap='coolwarm')
    # valence vs arousal corr
    print(gs.emonet_ann_outputs[["emonet_valence", "ann_valence", "emonet_arousal", "ann_arousal"]].corr(
        method="pearson"))
    plt.tight_layout()
    plt.show()


def analysis_arousal(gs):
    # analysis 5 : predicted arousal (EmoNet) vs annotated arousal (ANN)
    aro_emonet_ann_df = gs.get_aro_df()
    # get pearson correlation measure
    pears_aro = sp.stats.pearsonr(aro_emonet_ann_df["emonet_arousal"], aro_emonet_ann_df["ann_arousal"])
    print("emonet arousal vs ann arousal corr: ", pears_aro)
    # plotting
    sns.scatterplot(aro_emonet_ann_df, x=aro_emonet_ann_df["emonet_arousal"], y=aro_emonet_ann_df["ann_arousal"])
    plt.tight_layout()
    plt.show()

    # plot correlation matrix (one coefficient only here)
    sns.heatmap(aro_emonet_ann_df.corr(method="pearson"), annot=True, cmap='coolwarm')
    plt.tight_layout()
    plt.show()


def analysis_fact_emo(gs):
    # ( analysis  : deciding factor (ANN) vs annotated emotion (ANN) )
    fact_emo_ann_df = gs.get_emo_fact_df()
    fact_emo_ann_df_ct = pd.crosstab(fact_emo_ann_df["ann_dec_factors"], fact_emo_ann_df["ann_emotion"])
    cram_corr_emo_fact = sp.stats.contingency.association(fact_emo_ann_df_ct, 'cramer')
    chi_square_corr_emo_fact = sp.stats.chi2_contingency(fact_emo_ann_df_ct)
    print("emonet emotion vs ann factors", cram_corr_emo_fact)



if __name__ == '__main__':
    gs = GlobalStatistics(obj_importance_thres=0.7, emo_conf_thres=0.4, obj_conf_thres=0,
                          ann_ambiguity_thres=4)
    # filtering
    gs.remove_obj_cat("Person")
    gs.remove_obj_cat("Clothing")
    gs.remove_obj_outliers(10)

    # analyses
    analysis_obj_emo(gs)
    analysis_obj_ann(gs)
    analysis_emo_ann(gs)
    analysis_valence(gs)
    analysis_arousal(gs)

    # correlations
    sns.heatmap(gs.cramers_correlation_matrix(), annot=True, cmap='coolwarm')

    plt.tight_layout()

    plt.show()
