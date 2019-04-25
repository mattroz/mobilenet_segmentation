import json

import numpy as np
import keras
import tensorflow as tf

from tqdm import tqdm
from argparse import ArgumentParser

from data_generator.data_generator import COCODataLoader
from models.mobilenet_unet import MobilenetV2_base, relu6
from utils.losses import iou_metric, bce_dice_loss, get_precision, get_multi_threshold_precision


BATCH_SIZE = 32


def main():
    argp = ArgumentParser()
    argp.add_argument('--model', type=str, required=True)
    args = argp.parse_args()

    config = None
    with open('./config.json', 'r') as f:
        config = json.load(f)

    mobilenet = MobilenetV2_base()
    mobilenet.build_model(keras.layers.Input(shape=(400,400,3)))


    mobilenet.model = keras.models.load_model(
            args.model,
            custom_objects={'relu6': relu6,
                            'bce_dice_loss': bce_dice_loss,
                            'iou_metric': iou_metric})

    val_generator = COCODataLoader(
                        path_to_annotations=config['path_to_val_annotations'],
                        path_to_images=config['path_to_val_images'],
                        batch_size=BATCH_SIZE,
                        resize=(400,400),
                        augmentations=False,
                        shuffle=False)

    thresholds = np.arange(0.5, 1, 0.05)

    # Calculate mean intersection over union over all validation batches
    mean_iou = np.array([])

    print(f"\nEvaluating with batch size {BATCH_SIZE} ...")
    for index in tqdm(range(0, len(val_generator))):
        # Get predictions and prepare data for evaluating
        images, masks = val_generator[index]
        pred_mask = mobilenet.model.predict(images)
        pred_mask = keras.backend.cast(pred_mask, dtype=tf.float64)
        pred_mask = keras.backend.squeeze(pred_mask, axis=-1)
        masks = np.squeeze(masks)
        intersection_over_union = np.zeros((BATCH_SIZE,1))

        # Calculate IoU over all thresholds
        for threshold in thresholds:
            iou_over_threshold = np.reshape(get_precision(masks, pred_mask, threshold), (-1,1))
            intersection_over_union = np.concatenate((intersection_over_union, iou_over_threshold), axis=1)

        # Get mean IoU over thresholds over current batch
        mean_iou_over_threshold = np.mean(get_multi_threshold_precision(intersection_over_union[:, 1:]))
        mean_iou = np.append(mean_iou, mean_iou_over_threshold)
        with open('./results/metrics_log.txt', 'a') as f:
            f.write(f"{np.mean(mean_iou)}\n")

    print(f'Final mean IoU-over-threshold: {np.mean(mean_iou)}')


if __name__ == '__main__':
    main()
