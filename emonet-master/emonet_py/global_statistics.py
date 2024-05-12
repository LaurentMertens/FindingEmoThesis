import pandas as pd

class GlobalStatistics:

    def __init__(self):
        self.emonet_ann_ouputs = pd.read_csv('emonet_ann_outputs')
        self.yolo_outputs = pd.read_csv('yolo_outputs')

    def emonet_ann_analysis(self):
        """
        Analyze correlations between EmoNet predictions and FindingEmo annotations.
        """
        pass

    def objects_emotions_correlation_analysis(self):
        """
        Analyze correlations between the objects detected by Yolo and the emotions predicted by EmoNet.
        """
        pass



if __name__ == '__main__':
    gs = GlobalStatistics()
