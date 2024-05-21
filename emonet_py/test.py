import os.path
import matplotlib.pyplot as plt
import explanations_emonet, global_analysis, local_analysis, emonet
import pandas as pd
import numpy as np
import csv
import torch

print(os.listdir('test_images'))


def remove_obj_outliers(df, count):
    """
    Removes the detected objects occurring less than 'count' times
    """
    v = df.detected_object.value_counts()
    df = df[df.detected_object.isin(v.index[v.gt(count)])]
    return df

df = pd.DataFrame({"detected_object": ["car", "toy", "face"], "emotion": ["joy", "joy", "sadness"]})
print(df)
df = remove_obj_outliers(df, 1)
print(df)

