

class Store:

    def __init__(self):
        self._train_hist = {}
        self._train_hist['D_losses'] = []
        self._train_hist['G_losses'] = []
        self._train_hist['per_epoch_ptimes'] = []
        self._train_hist['total_ptime'] = []

    def hist_append(self, key, value):
        self._train_hist[key].append(value)


    def retrieve(self, key):
        return self._train_hist[key]

