class smoothing_aggregator:
    def __init__(self, path):
        self.aggregator_size = 100
        self.aggregator = []
        self.aggregated_vals = []
        self.num_grapic = 0
        self.path = path

    def update(self, val):
        self.aggregator.append(val)
        if len(self.aggregator) >= self.aggregator_size:
            self.aggregated_vals.append(np.mean(np.asarray(self.aggregator)))
            self.aggregator = []
            if len(self.aggregated_vals%5==0):
                self.save_graphic()

    def save_graphic(self):
        plt.plot(self.aggregated_vals)
        plt.savefig('progress_' + str(len(self.aggregated_vals)) + '.png')
