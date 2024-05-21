"""
Logger class. Implements a "print" method that prints to the console, as well as writes to a user-specified log file.
If no log file is specified, will just print to console.

.. codeauthor:: Laurent Mertens <laurent.mertens@kuleuven.be>
"""
import os.path
from datetime import datetime

import torch.nn

from tools.print_tools import PrintTools


class Logger:
    """
    Logger class that permits to print output to the console, whilst also writing it away to a log file.
    It also provides methods to write training results and confusion matrices to separate files.
    """
    def __init__(self, logfile=None, b_overwrite=False, b_create_out_dir=False):
        """

        :param logfile: path to file to write to; default = None, in which case text will only be printed to console,
        i.e., behaviour reverts to the default print() method.
        :param b_overwrite: if 'True' and logfile already exists, will overwrite the existing file; else will create a
        new file with an appended index; e.g., say "logfile"="brol.txt", and "brol.txt" already exists, then "brol_2.txt"
        will be used.
        :param b_create_out_dir: create output directory if it does not already exist
        """
        self.logfile = logfile

        # Store this in a boolean, so as not to always have to preform the check
        self.b_log = (logfile is not None)

        if self.b_log:
            dir_file = os.path.dirname(logfile)
            if not os.path.exists(dir_file):
                if b_create_out_dir:
                    os.makedirs(dir_file)
                else:
                    raise NotADirectoryError(f"Output directory in which the logger file should be stored does not exist:\n\t{dir_file}\n"
                                             f"\nUse option 'b_create_out_dir=True' to automatically create it.")

            if os.path.isfile(logfile):
                if b_overwrite:
                    os.remove(logfile)
                else:
                    # Get directory for logfile
                    dir_file = os.path.dirname(logfile)
                    # Split extension from rest of filename
                    filename, ext = os.path.splitext(logfile)
                    filename = filename.replace(dir_file, '')[1:]
                    # Get index to be used, i.e., check how many index versions already exist
                    idx = 2  # Starting index, in case "logfile" already exists, but no other alternatives
                    for f in os.listdir(dir_file):
                        if f.startswith(filename + '_'):
                            f_name, f_ext = os.path.splitext(f)
                            # Check f_name equals form "filename_x", with x an integer
                            f_name_diff = f_name.replace(filename + '_', '')
                            if f_name_diff.isdigit():
                                idx += 1
                    # Define new "logfile"
                    self.logfile = os.path.join(dir_file, f"{filename}_{idx}{ext}")
                    # raise FileExistsError(f'Specified log file already exists: {logfile}')

    def print(self, s: str = '', **params):
        print(s, **params)
        self.write(s)

    def print_line(self, length=60, **params):
        """
        Print a line consisting of 'length' '-' characters

        :param length: number of consecutive '-' characters to print
        :return:
        """
        s = '-'*length
        self.print(s, **params)
        self.write(s)

    def print_dbl_line(self, length=60, **params):
        """
        Print a line consisting of 'length' '=' characters

        :param length: number of consecutive '=' characters to print
        :return:
        """
        s = '=' * length
        self.print(s, **params)
        self.write(s)

    def print_star_line(self, length=60, **params):
        """
        Print a line consisting of 'length' '*' characters

        :param length: number of consecutive '*' characters to print
        :return:
        """
        s = '*' * length
        self.print(s, **params)
        self.write(s)

    def write(self, s: str = ''):
        if self.b_log:
            with open(self.logfile, 'a') as fout:
                fout.write(s + '\n')

    # Write classifier results from training
    @staticmethod
    def open_classifier_results_csv(csv_file: str, b_overwrite=False):
        """
        Create results CSV file if it does not yet exist, and write header (i.e., column titles).
        If the file already exists, it is left untouched, unless you explicitly want it overwritten.

        :param csv_file: path to file
        :param b_overwrite: overwrite (i.e., replace) file if it already exists.
        :return:
        """
        if csv_file is not None:
            if b_overwrite or (not os.path.exists(csv_file)):
                file_dir = os.path.dirname(csv_file)
                if not os.path.isdir(file_dir):
                    os.makedirs(file_dir)
                with open(csv_file, 'w') as fout:
                    fout.write(f"datetime,model,loss,optimizer,lr,best_epoch,"
                               f"#train_samples,train_loss,train_acc,train_weighted_acc,train_macro_f1,train_weighted_f1,train_ap,"
                               f"#test_samples,test_loss,test_acc,test_weighted_acc,test_macro_f1,test_weighted_f1,test_ap,output_file\n")

    @staticmethod
    def write_classifier_result_to_csv(best_stats: dict, csv_file: str):
        """
        Append provided result to provided CSV file.

        :param best_stats: dictionary containing the result to be appended to the csv_file
        :param csv_file: path to file
        :return:
        """
        if csv_file is not None:
            with open(csv_file, 'a') as fout:
                fout.write(f"{best_stats['datetime']},{best_stats['model_name']},{best_stats['loss']},{best_stats['optimizer']},"
                           f"{best_stats['lr']},{best_stats['epoch']},"
                           f"{best_stats['#train_samples']},{best_stats['train_avg']:},"
                           f"{best_stats['train_acc']:.5f},{best_stats['train_weighted_acc']:.5f},"
                           f"{best_stats['train_macro_f1']:.5f},{best_stats['train_weighted_f1']:.5f},"
                           f"{best_stats['train_ap']:.5f},"
                           f"{best_stats['#test_samples']},{best_stats['test_avg']},"
                           f"{best_stats['test_acc']:.5f},{best_stats['test_weighted_acc']:.5f},"
                           f"{best_stats['test_macro_f1']:.5f},{best_stats['test_weighted_f1']:.5f},"
                           f"{best_stats['test_ap']:.5f},"
                           f"{best_stats['output_file']}\n")

    # Write regressor results from training
    @staticmethod
    def open_regressor_results_csv(csv_file: str, b_overwrite=False):
        """
        Create results CSV file if it does not yet exist, and write header (i.e., column titles).
        If the file already exists, it is left untouched, unless you explicitly want it overwritten.

        :param csv_file: path to file
        :param b_overwrite: overwrite (i.e., replace) file if it already exists.
        :return:
        """
        if csv_file is not None:
            if b_overwrite or (not os.path.exists(csv_file)):
                file_dir = os.path.dirname(csv_file)
                if not os.path.isdir(file_dir):
                    os.makedirs(file_dir)
                with open(csv_file, 'w') as fout:
                    fout.write(f"datetime,model,loss,optimizer,lr,best_epoch,"
                               f"#train_samples,train_loss,train_mae,train_spearman_r,train_spearman_r_p,"
                               f"train_pred_min,train_pred_max,train_pred_avg,train_pred_std,"
                               f"#test_samples,test_loss,test_mae,test_spearman_r,test_spearman_r_p,"
                               f"test_pred_min,test_pred_max,test_pred_avg,test_pred_std,output_file\n")

    @staticmethod
    def write_regressor_result_to_csv(best_stats: dict, csv_file: str):
        """
        Append provided result to provided CSV file.

        :param best_stats: dictionary containing the result to be appended to the csv_file
        :param csv_file: path to file
        :return:
        """
        if csv_file is not None:
            with open(csv_file, 'a') as fout:
                fout.write(f"{best_stats['datetime']},{best_stats['model_name']},{best_stats['loss']},{best_stats['optimizer']},"
                           f"{best_stats['lr']},{best_stats['epoch']},"
                           f"{best_stats['#train_samples']},{best_stats['train_avg']},"
                           f"{best_stats['train_mae']:.5f},"
                           f"{best_stats['train_spearman_r']:.5f},{best_stats['train_spearman_r_p']:.5f},"
                           f"{best_stats['train_pred_min']:.5f},{best_stats['train_pred_max']:.5f},"
                           f"{best_stats['train_pred_avg']:.5f},{best_stats['train_pred_std']:.5f},"
                           f"{best_stats['#test_samples']},{best_stats['test_avg']},"
                           f"{best_stats['test_mae']:.5f},"
                           f"{best_stats['test_spearman_r']:.5f},{best_stats['test_spearman_r_p']:.5f},"
                           f"{best_stats['test_pred_min']:.5f},{best_stats['test_pred_max']:.5f},"
                           f"{best_stats['test_pred_avg']:.5f},{best_stats['test_pred_std']:.5f},"
                           f"{best_stats['output_file']}\n")

    # Write confusion matrices and metrics per class table
    @staticmethod
    def open_mtx_file(mtx_file: str, b_overwrite=False):
        """
        Create confusion matrix text file if it does not yet exist.
        If the file already exists, it is left untouched, unless you explicitly want it overwritten.

        :param mtx_file: path to file
        :param b_overwrite: overwrite (i.e., replace) file if it already exists
        :return:
        """
        if mtx_file is not None:
            if b_overwrite or (not os.path.exists(mtx_file)):
                with open(mtx_file, 'w') as fout:
                    fout.write("Confusion matrices pertaining to best epochs per model.\n")

    @staticmethod
    def write_conf_mtx(best_stats: dict, mtx_file: str):
        """

        :param best_stats: dictionary containing {model_name: str, lr: float, train_c_mtx: str, test_c_mtx: str}
        :param mtx_file: path to file
        :return:
        """
        if mtx_file is not None:
            with open(mtx_file, 'a') as fout:
                fout.write("=" * 100 + '\n')
                fout.write("=" * 100 + '\n')
                fout.write(f"{best_stats['datetime']}\n")
                fout.write(f"Train {best_stats['model_name']} -- {best_stats['lr']}:\n")
                fout.write(f"{best_stats['train_c_mtx']}\n\n")
                fout.write(PrintTools.print_class_metrics_mtx(
                    prec=best_stats['prec_per_label'], rec=best_stats['rec_per_label'], f1=best_stats['f1_per_label']))
                fout.write("-" * 100 + '\n')
                fout.write(f"Test {best_stats['model_name']} -- {best_stats['lr']}:\n")
                fout.write(f"{best_stats['test_c_mtx']}\n\n")
                fout.write(PrintTools.print_class_metrics_mtx(
                    prec=best_stats['test_prec_per_label'], rec=best_stats['test_rec_per_label'], f1=best_stats['test_f1_per_label']))

    # Save Multimodal Layer1 bias and weights
    @staticmethod
    def write_layer_weights(epoch: int, layer: torch.nn.Linear, outfile: str):
        """
        Append bias and weights of provided layer to provided output file.

        :param epoch: training epoch, or random number to your liking of not training
        :param layer: the linear layer whose bias and weights will be written to outfile
        :param outfile: path to output file
        :return:
        """
        if outfile is not None:
            with open(outfile, 'a') as fout:
                fout.write(f"{epoch};{layer.bias.tolist()};{layer.weight.tolist()}\n")

    @staticmethod
    def get_nb_files(out_dir, gen_name):
        # Get index to be used, i.e., check how many index versions already exist
        idx = 1  # Starting index, in case "logfile" already exists, but no other alternatives
        for f in sorted(os.listdir(out_dir)):
            if f == f"{gen_name}.txt":
                idx += 1
            elif f.startswith(gen_name + '_'):
                f_name, f_ext = os.path.splitext(f)
                # Check f_name equals form "filename_x", with x an integer
                f_name_diff = f_name.replace(gen_name + '_', '')
                if f_name_diff.isdigit():
                    idx += 1
        return idx
