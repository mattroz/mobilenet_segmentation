import math
import tensorflow as tf
import keras
import keras.backend as K
import numpy as np


class CyclicLearningRateScheduler(keras.callbacks.History):
    """
    """

    def __init__(self, base_lr, max_lr, step_size):

        super(CyclicLearningRateScheduler, self).__init__()

        self.base_lr = base_lr
        self.max_lr = max_lr
        self.step_size = step_size

    def on_train_begin(self, logs=None):
        K.set_value(self.model.optimizer.lr, self.base_lr)

    def on_batch_end(self, batch, logs=None):
        cycle = math.floor(1 + batch / (2 * self.step_size))
        x = abs(batch / self.step_size - 2 * cycle + 1)
        updated_lr = self.base_lr + (self.max_lr - self.base_lr) * max(0, 1 - x)
        K.set_value(self.model.optimizer.lr, updated_lr)