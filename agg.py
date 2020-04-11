import numpy as np
import matplotlib.pyplot as plt
from hyperparametrs import *

SAVING_AFTER = SAVE_STEPS/100

class Smoothing_aggregator:
    def __init__(self, path):
        self.aggregator_size = 0
        self.aggregator_max_size = 100
        self.aggregator = []
        self.aggregated_vals = []
        self.num_grapic = 0
        self.path = path

    def update(self, loss_arrays):
        print(loss_array.shape)
        self.aggregator.append(loss_arrays)

        self.aggregator_size+=1
        if self.aggregator_size >= self.aggregator_max_size:
            self.aggregated_vals.append(np.mean(np.asarray(self.aggregator)))
            self.aggregator = []
            self.aggregator_size = 0
            if len(self.aggregated_vals)%SAVING_AFTER==0:
                self.save_graphic()
                return True
        return False

    def save_graphic(self):
        plt.plot(self.aggregated_vals)
        plt.savefig(self.path + '/progress_' + str(len(self.aggregated_vals)) + '.png')

class Smoothing_aggregator_dual:
    def __init__(self, path):
        self.aggregator_size = 0
        self.aggregator_max_size = 100
        self.aggregator_stem = []
        self.aggregator_seg = []
        self.aggregated_vals_stem = []
        self.aggregated_vals_seg = []
        self.path = path

    def update(self, loss_stem, loss_seg):
        self.aggregator_stem.append(loss_stem)
        self.aggregator_seg.append(loss_seg)
        self.aggregator_size+=1
        if self.aggregator_size >= self.aggregator_max_size:
            self.aggregated_vals_stem.append(np.mean(np.asarray(self.aggregator_stem)))
            self.aggregated_vals_seg.append(np.mean(np.asarray(self.aggregator_seg)))
            self.aggregator_stem = []
            self.aggregator_seg = []
            self.aggregator_size = 0
            if len(self.aggregated_vals_stem)%SAVING_AFTER==0:
                self.save_graphic()
                return True
        return False

    def save_graphic(self):
        plt.subplot(211)
        plt.plot(self.aggregated_vals_stem)
        plt.gca().set_title('stem')
        plt.subplot(212)
        plt.plot(self.aggregated_vals_seg)
        plt.gca().set_title('seg')
        plt.savefig(self.path + '/progress_' + str(len(self.aggregated_vals_stem)) + '.png')
