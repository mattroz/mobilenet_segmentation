import os
import keras
import numpy as np

from math import ceil
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, ReduceLROnPlateau, CSVLogger
from argparse import ArgumentParser

from data_generator.data_generator import COCODataLoader
from models.mobilenet_unet import MobilenetV2_base, relu6
from utils.utils import iou_metric, dice_loss, bce_dice_loss, focal_dice_loss
from utils.cyclic_learning_rate import CyclicLearningRateScheduler


BATCH_SIZE = 6
LR = 1e-3
EPOCHS = 100
INPUT_SHAPE = (400, 400, 3)

if __name__ == '__main__':
    argparser = ArgumentParser()
    argparser.add_argument('--initial_epoch', type=int, required=True)
    argparser.add_argument('--final_epoch', type=int, required=False, default=EPOCHS)
    argparser.add_argument('--model', type=str, required=False, default=None)
    argparser.add_argument('--freeze_encoder', default=False, required=False, dest='freeze_encoder', action='store_true')
    argparser.add_argument('--lr', type=float, required=False, default=LR)
    args = argparser.parse_args()

    # Get the model
    mobilenet = MobilenetV2_base()
    mobilenet.build_model(keras.layers.Input(shape=INPUT_SHAPE))

    # Load saved model if specified
    if args.model is not None:
        mobilenet.model = keras.models.load_model(args.model,
                                                  custom_objects={'relu6' : relu6,
                                                                  'iou_metric' : iou_metric,
                                                                  'bce_dice_loss' : bce_dice_loss,
                                                                  'focal_dice_loss' : focal_dice_loss},
                                                  compile=False)

    # Freeze encoder layers which are pretrained
    if args.freeze_encoder:
        for layer in mobilenet.model.layers:
            if layer.name.startswith('enc'):
                layer.trainable=False
    else:
        for layer in mobilenet.model.layers:
            layer.trainable=True
    print(mobilenet.model.summary())

    # Define optimizer and compile model
    opt = keras.optimizers.Adam(lr=args.lr)
    mobilenet.model.compile(optimizer=opt, loss=bce_dice_loss, metrics=[iou_metric])

    # Get data generators
    train_generator = COCODataLoader(
                    path_to_annotations='/home/matsvei.rozanau/hdd/datasets/coco_dataset/annotations/instances_train2017.json',
                    path_to_images='/home/matsvei.rozanau/hdd/datasets/coco_dataset/train2017/',
                    batch_size=BATCH_SIZE,
                    resize=INPUT_SHAPE[:-1],
                    augmentations=True)
    val_generator = COCODataLoader(
                    path_to_annotations='/home/matsvei.rozanau/hdd/datasets/coco_dataset/annotations/instances_val2017.json',
                    path_to_images='/home/matsvei.rozanau/hdd/datasets/coco_dataset/val2017/',
                    batch_size=BATCH_SIZE,
                    resize=INPUT_SHAPE[:-1],
                    augmentations=False)

    # Define callbacks
    model_checkpoint = ModelCheckpoint(
        filepath='./checkpoints/mobilenet400_test-{epoch:02d}_loss-{loss:.4f}_val_loss-{val_loss:.4f}.h5',
        monitor = 'val_loss',
        verbose = 1,
        save_best_only = True,
        save_weights_only = False,
        mode = 'auto',
        period = 1)

    plateau_reducer_checkpoint = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.6,
        patience=5,
        verbose=1,
        min_lr=1e-9)

    cyclic_learning_rate = CyclicLearningRateScheduler(
        base_lr=1e-6,
        max_lr=1e-2,
        step_size=5 * ceil(len(train_generator) / BATCH_SIZE),
        search_optimal_bounds=True)

    callbacks = [model_checkpoint, plateau_reducer_checkpoint, cyclic_learning_rate]

    print('\nTraining...')
    train_history = mobilenet.model.fit_generator(
        generator=train_generator,
        max_queue_size = 10,
        workers = 8,
        use_multiprocessing = True,
        steps_per_epoch = ceil(len(train_generator) / BATCH_SIZE),
        initial_epoch = args.initial_epoch,
        epochs = args.final_epoch,
        callbacks = callbacks,
        validation_data = val_generator,
        validation_steps = ceil(len(val_generator) / BATCH_SIZE))

