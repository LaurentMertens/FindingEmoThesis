"""
Some methods to print stuff.

.. codeauthor:: Laurent Mertens <laurent.mertens@kuleuven.be>
"""
import numpy as np
import termcolor


class PrintTools:
    @staticmethod
    def print_class_metrics_mtx(prec: dict, rec: dict, f1: dict):
        """

        :param prec: {k: v} with k = class, v = precision
        :param rec: {k: v} with k = class, v = recall
        :param f1: {k: v} with k = class, v = f1
        :return: string representation of table depicting the metrics
        """
        assert len(prec) == len(rec)
        assert len(prec) == len(f1)

        nb_classes = len(prec)

        # Print
        tbl = ' '*7
        for i in range(nb_classes):
            tbl += f' {i:>02d}    '

        for metric_name, metric in [('prec', prec), ('rec', rec), ('f1', f1)]:
            tbl += f'\n{metric_name:6s}'
            for r in range(nb_classes):
                tbl += f'  {metric[r]:.3f}'

        tbl += '\n'

        return tbl

    @staticmethod
    def print_multilabel_mtx(tp, fp, tn, fn, prec, rec, f1, ap):
        """

        :param tp: 1D array, true positive counts per class
        :param fp: 1D array, false positive counts per class
        :param tn: 1D array, true negative counts per class
        :param fn: 1D array, false negative counts per class
        :param prec: precision per class
        :param rec: recall per class
        :param f1: f1 per class
        :param ap: average precision per class
        :return:
        """
        nb_classes = len(tp)

        # Print
        tbl = ' '*6
        for i in range(nb_classes):
            tbl += f'      {i:>02d}'

        for metric_name, metric in [('tp', tp), ('fp', fp), ('tn', tn), ('fn', fn)]:
            tbl += f'\n{metric_name:6s}'
            for t in range(nb_classes):
                tbl += f'   {metric[t]:>5.0f}'
        tbl += '\n' + ' '*6 + '-'*8*nb_classes
        for metric_name, metric in [('prec', prec), ('rec', rec), ('f1', f1), ('ap', ap)]:
            tbl += f'\n{metric_name:6s}'
            for t in range(nb_classes):
                tbl += f'   {metric[t]:>4.3f}'

        tbl += '\n'

        return tbl

    @staticmethod
    def print_reg_metrics_mtx(avg_per_target: dict, std_per_target: dict):
        """

        :param avg_per_target: {k: v} with k = target, v = average
        :param std_per_target: {k: v} with k = target, v = std
        :return: string representation of table depicting the metrics
        """
        assert len(avg_per_target) == len(std_per_target)

        targets = sorted(avg_per_target.keys())

        # Print
        tbl = ' '*6
        for t in targets:
            tbl += f'   {t:>5.2f}'

        for metric_name, metric in [('avg', avg_per_target), ('std', std_per_target)]:
            tbl += f'\n{metric_name:6s}'
            for t in targets:
                tbl += f'   {metric[t]:5.2f}'

        tbl += '\n'

        return tbl

    @staticmethod
    def print_confusion_mtx(preds: [int], targets: [int], nb_classes: int, class_names=None, color=None):
        """

        :param preds: list containing the predicted class indices
        :param targets: list containing the target class indices
        :param nb_classes: number of classes for this classification problem
        :param color: color of the diagonal elements, default='None' = no specific color
        :param class_names: list containing class names; default='None' = ignore; else the class name corresponding\
        to each row will be printed to its right
        :return: string representation of the confusion matrix
        """
        confusion_mtx = np.zeros((nb_classes, nb_classes))
        for t in zip(preds, targets):
            confusion_mtx[t[0], t[1]] += 1

        # Normalize and turn in to percentages
        confusion_mtx_pc = confusion_mtx.copy()
        for r in range(nb_classes):
            r_sum = sum(confusion_mtx_pc[r, :])
            if r_sum > 0:
                confusion_mtx_pc[r, :] /= 0.01*r_sum

        # Print
        tbl = ''
        for i in range(nb_classes):
            tbl += f'\t{i:>2d}'
        tbl += f'\t\t#'
        tbl += '\n'
        for r in range(nb_classes):
            tbl += f'{r:2d}|'
            for c in range(nb_classes):
                mtx_entry = f'\t{confusion_mtx_pc[r, c]:>3.2f}'
                if color is not None and r == c:
                    mtx_entry = termcolor.colored(mtx_entry, color=color, attrs=['bold'])
                tbl += mtx_entry
            tbl += f'\t\t{int(sum(confusion_mtx[r, :]))}'
            if class_names is not None:
                tbl += f'\t{class_names[r]}'
            tbl += '\n'

        return tbl


if __name__ == '__main__':
    # print(PrintTools.print_confusion_mtx([1,3,4,1,6,1,2,3,2], [6,1,3,1,7,3,4,5,1], nb_classes=8))
    #
    # p = {0: 0.1, 1: 0.5, 2: 0.4}
    # r = {0: 0.245, 1: 0.34, 2: 0.246}
    # f = {0: 0.5261, 1: 0.1345, 2: 0.346136}
    # print(PrintTools.print_class_metrics_mtx(p, r, f))

    apt = {0: 0.21143, 1: 1.75632, 2.5: 7.352}
    spt = {0: 1.35, 1: 35.61, 2.5: 61.34}
    print(PrintTools.print_reg_metrics_mtx(apt, spt))
