import json

from math import ceil

import keras
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, ReduceLROnPlateau
from argparse import ArgumentParser

from data_generator.data_generator import COCODataLoader
from models.mobilenet_unet import MobilenetV2_base, relu6
from utils.losses import iou_metric, lovasz_hinge_loss, bce_dice_loss, focal_dice_loss
from utils.cyclical_learning_rate import CyclicalLearningRateScheduler


BATCH_SIZE = 6
LR = 1e-3
EPOCHS = 100
INPUT_SHAPE = (401, 401, 3)

if __name__ == '__main__':
    argparser = ArgumentParser()
    argparser.add_argument('--initial_epoch', type=int, required=True)
    argparser.add_argument('--final_epoch', type=int, required=False, default=EPOCHS)
    argparser.add_argument('--model', type=str, required=False, default=None)
    argparser.add_argument('--freeze_encoder', default=False, required=False, dest='freeze_encoder', action='store_true')
    argparser.add_argument('--lr', type=float, required=False, default=LR)
    argparser.add_argument('--loss', type=str, default='bce_dice', required=False)
    args = argparser.parse_args()

    config = None
    with open('./config.json', 'r') as f:
        config = json.load(f)

    # Get the model
    mobilenet = MobilenetV2_base()
    mobilenet.build_model(keras.layers.Input(shape=INPUT_SHAPE))

    # Load saved model if specified
    if args.model is not None:
        mobilenet.model = keras.models.load_model(
                args.model,
                custom_objects={'relu6': relu6,
                                'iou_metric': iou_metric,
                                'bce_dice_loss': bce_dice_loss,
                                'focal_dice_loss': focal_dice_loss},
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

    loss = None
    if args.loss == 'bce_dice':
        loss = bce_dice_loss
    elif args.loss.startswith('lovasz'):
        loss = lovasz_hinge_loss
    else:
        loss = bce_dice_loss

    mobilenet.model.compile(optimizer=opt, loss=bce_dice_loss, metrics=[iou_metric])

    # Get data generators
    train_generator = COCODataLoader(
                    path_to_annotations=config['path_to_train_annotations'],
                    path_to_images=config['path_to_train_images'],
                    batch_size=BATCH_SIZE,
                    resize=INPUT_SHAPE[:-1],
                    augmentations=True)
    val_generator = COCODataLoader(
                    path_to_annotations=config['path_to_val_annotations'],
                    path_to_images=config['path_to_val_images'],
                    batch_size=BATCH_SIZE,
                    resize=INPUT_SHAPE[:-1],
                    augmentations=False)

    # Define callbacks
    model_checkpoint = ModelCheckpoint(
        filepath='./checkpoints/mobilenet401_lovasz-{epoch:02d}_loss-{loss:.4f}_val_loss-{val_loss:.4f}.h5',
        monitor = 'val_loss',
        verbose = 1,
        save_best_only = True,
        save_weights_only = False,
        mode = 'auto',
        period = 1)

    plateau_reducer_checkpoint = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        verbose=1,
        min_lr=1e-8)

    cyclic_learning_rate = CyclicalLearningRateScheduler(
        base_lr=1e-7,
        max_lr=0.02,
        step_size=5 * ceil(len(train_generator) / BATCH_SIZE),
        search_optimal_bounds=False)

    callbacks = [model_checkpoint, plateau_reducer_checkpoint]#cyclic_learning_rate

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

