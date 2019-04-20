import math
import tensorflow as tf
import keras
import keras.backend as K
import numpy as np
import matplotlib.pyplot as plt


class CyclicLearningRateScheduler(keras.callbacks.History):
    """
    """

    def __init__(self, base_lr, max_lr, step_size, search_optimal_bounds=False):

        super(CyclicLearningRateScheduler, self).__init__()

        self.base_lr = base_lr
        self.max_lr = max_lr
        self.step_size = step_size
        self.search_optimal_bounds = search_optimal_bounds

    def on_train_begin(self, logs=None):
        K.set_value(self.model.optimizer.lr, self.base_lr)
        if self.search_optimal_bounds:
            n_epochs_for_search = 1
            iters_in_epoch = self.params['steps']
            self.search_lrs = np.linspace(self.base_lr,
                                          self.max_lr,
                                          n_epochs_for_search * iters_in_epoch + 1)
            print(self.search_lrs)
            self.search_iteration = 1
            self.losses = np.array([])

    def on_batch_end(self, batch, logs=None):
        updated_lr = None
        if self.search_optimal_bounds:
            self.losses = np.append(self.losses, logs.get('loss'))
            updated_lr = self.search_lrs[self.search_iteration]
            self.search_iteration += 1
        else:
            cycle = math.floor(1 + batch / (2 * self.step_size))
            x = abs(batch / self.step_size - 2 * cycle + 1)
            updated_lr = self.base_lr + (self.max_lr - self.base_lr) * max(0, 1 - x)
        K.set_value(self.model.optimizer.lr, updated_lr)

    def on_train_end(self, logs=None):
        if self.search_optimal_bounds:
            plt.plot(self.search_lrs, self.losses)