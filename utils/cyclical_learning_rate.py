import math
import tensorflow as tf
import keras
import keras.backend as K
import numpy as np
import matplotlib.pyplot as plt


class CyclicalLearningRateScheduler(keras.callbacks.History):
    """
    """

    def __init__(self, base_lr, max_lr, step_size, search_optimal_bounds=False):

        super(CyclicalLearningRateScheduler, self).__init__()

        self.base_lr = base_lr
        self.max_lr = max_lr
        self.step_size = step_size
        self.search_optimal_bounds = search_optimal_bounds

    def on_train_begin(self, logs=None):
        K.set_value(self.model.optimizer.lr, self.base_lr)
        if self.search_optimal_bounds:
            # Save model initial weights for further reinitialization
            self.initial_weights = self.model.get_weights()
            # Create array of learning rates to iterate over
            self.n_epochs_for_search = self.step_size // 2
            logs['epochs'] += self.n_epochs_for_search
            iters_in_epoch = self.params['steps']
            self.search_lrs = np.linspace(self.base_lr,
                                          self.max_lr,
                                          self.n_epochs_for_search * iters_in_epoch)
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

    def __calculate_optimal_bounds(self):
        self.search_lrs = np.reshape(self.search_lrs, (-1, 1))
        self.losses = np.reshape(self.losses, (-1, 1))
        averaged_losses = np.array(self.losses[0])

        # Calculate moving average for optimal bounds searching
        smooth = 0.05
        for i in range(1, self.losses.shape[0]):
            loss =  smooth * self.losses[i] + (1 - smooth) * averaged_losses[-1]
            averaged_losses = np.append(averaged_losses, loss)

        # Get learning rate with the highest loss and the lowest one
        lowest_loss_idx = np.argmin(averaged_losses)
        highest_loss_idx = np.argmax(averaged_losses)

        return self.search_lrs[highest_loss_idx], self.search_lrs[lowest_loss_idx]

    def on_epoch_end(self, epoch, logs=None):
        if self.search_optimal_bounds:
            if epoch == self.n_epochs_for_search:
                print("Stop searching for optimal bounds, calculating...")
                self.base_lr, self.max_lr = self.__calculate_optimal_bounds()
                print(f"Set base_lr: {self.base_lr}, max_lr: {self.max_lr}")
                self.search_optimal_bounds = False
                print(f"Reinitializing model...")
                self.model.set_weights(self.initial_weights)
                del self.initial_weights
