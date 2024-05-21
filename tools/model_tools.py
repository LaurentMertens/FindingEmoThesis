"""

.. codeauthor:: Laurent Mertens <laurent.mertens@kuleuven.be>
"""
import torch


class ModelTools:
    @staticmethod
    def get_most_probable_class(preds: torch.Tensor, class_labels):
        """

        :param preds: 1D tensor containing the probabilities for each class
        :param class_labels: the corresponding class labels
        :return:
        """
        max_prob = preds[0]
        max_class = class_labels[0]
        class_index = 0
        for emo_idx in range(len(class_labels)):
            if preds[emo_idx] > max_prob:
                max_prob = preds[emo_idx]
                max_class = class_labels[emo_idx]
                class_index = emo_idx

        return max_prob.item(), max_class, class_index
