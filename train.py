import os
import keras
import numpy as np

from math import ceil
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, ReduceLROnPlateau, TerminateOnNaN
from argparse import ArgumentParser

from data_generator.data_generator import COCODataLoader
from models.mobilenet_unet import MobilenetV2_base, relu6


BATCH_SIZE = 8
LR = 1e-4

if __name__ == '__main__':

    # Get the model
    mobilenet = MobilenetV2_base()
    mobilenet.build_model(keras.layers.Input(shape=(224,224,3)))

    # Define optimizer and compile model
    opt = keras.optimizers.Adam(lr=LR)
    mobilenet.model.compile(optimizer=opt, loss='binary_crossentropy')

    # Get data generators
    train_generator = COCODataLoader(
                    path_to_annotations='/home/matsvei.rozanau/hdd/datasets/coco_dataset/annotations/instances_train2017.json',
                    path_to_images='/home/matsvei.rozanau/hdd/datasets/coco_dataset/train2017/',
                    batch_size=BATCH_SIZE,
                    resize=(224,224),
                    augmentations=True)
    val_generator = COCODataLoader(
                    path_to_annotations='/home/matsvei.rozanau/hdd/datasets/coco_dataset/annotations/instances_val2017.json',
                    path_to_images='/home/matsvei.rozanau/hdd/datasets/coco_dataset/val2017/',
                    batch_size=BATCH_SIZE,
                    resize=(224,224),
                    augmentations=True)

    # Define callbacks
    model_checkpoint = ModelCheckpoint(
        filepath='./checkpoints/mobilenet-{epoch:02d}_loss-{loss:.4f}_val_loss-{val_loss:.4f}.h5',
        monitor = 'val_loss',
        verbose = 1,
        save_best_only = True,
        save_weights_only = False,
        mode = 'auto',
        period = 1)
    callbacks = [model_checkpoint]


    train_history = mobilenet.model.fit_generator(
        generator=train_generator,
        max_queue_size = 10,
        workers = 8,
        use_multiprocessing = True,
        steps_per_epoch = ceil(len(train_generator) / BATCH_SIZE),
        epochs = 5,
        callbacks = callbacks,
        validation_data = val_generator,
        validation_steps = ceil(len(val_generator) / BATCH_SIZE),
        initial_epoch = 0)

