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

    def update(self, loss_array):
        #print(loss_array.shape)
        self.aggregator.append(loss_array)

        self.aggregator_size+=1
        if self.aggregator_size >= self.aggregator_max_size:
            self.aggregated_vals.append(np.mean(np.asarray(self.aggregator)))
            print(self.aggregated_vals, '!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
            self.aggregator = []
            self.aggregator_size = 0
            if len(self.aggregated_vals)%SAVING_AFTER==0:
                self.save_graphic()
                return True
        return False

    def save_graphic(self):
        plt.plot(self.aggregated_vals)
        plt.savefig(self.path + '/progress_' + str(len(self.aggregated_vals)) + '.png')
