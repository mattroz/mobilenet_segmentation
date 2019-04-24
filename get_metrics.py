import os
import keras
import gc
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from argparse import ArgumentParser

from data_generator.data_generator import COCODataLoader
from models.mobilenet_unet import MobilenetV2_base, relu6
from utils.losses import bce_dice_loss, get_precision, get_multi_threshold_precision

from utils.losses import iou_metric


BATCH_SIZE = 32

def main():
    argp = ArgumentParser()
    argp.add_argument('--model', type=str, required=True)
    args = argp.parse_args()

    mobilenet = MobilenetV2_base()
    mobilenet.build_model(keras.layers.Input(shape=(400,400,3)))


    mobilenet.model = keras.models.load_model(args.model,
                                              custom_objects={'relu6' : relu6,
                                                              'bce_dice_loss' : bce_dice_loss,
                                                              'iou_metric' : iou_metric})

    val_generator = COCODataLoader(
                        path_to_annotations='/home/matsvei.rozanau/hdd/datasets/coco_dataset/annotations/instances_val2017.json',
                        path_to_images='/home/matsvei.rozanau/hdd/datasets/coco_dataset/val2017/',
                        batch_size=BATCH_SIZE,
                        resize=(400,400),
                        augmentations=False,
                        shuffle=False)


    thresholds = np.arange(0.5, 1, 0.05)

    # Calculate mIoU over all validation batches
    mIoU = np.array([])

    import time
    print(f"\nEvaluating with batch size {BATCH_SIZE} ...")
    for i in tqdm(range(0, len(val_generator))):
        # Get predictions and prepare data for evaluating
        images, masks = val_generator[i]
        pred_mask = mobilenet.model.predict(images)
        pred_mask = keras.backend.cast(pred_mask, dtype=tf.float64)
        pred_mask = keras.backend.squeeze(pred_mask, axis=-1)
        masks = np.squeeze(masks)
        IoU = np.zeros((BATCH_SIZE,1))

        # Calculate IoU over all thresholds
        for threshold in thresholds:
            iou_over_threshold = np.reshape(get_precision(masks, pred_mask, threshold), (-1,1))
            IoU = np.concatenate((IoU, iou_over_threshold), axis=1)

        # Get mean IoU over thresholds over current batche
        mean_iou_over_threshold = np.mean(get_multi_threshold_precision(IoU[:, 1:]))
        mIoU = np.append(mIoU, mean_iou_over_threshold)
        with open('./results/metrics_log.txt', 'a') as f:
            f.write(f"{np.mean(mIoU)}\n")
        gc.collect()
    print(f'Final mean IoU-over-threshold: {np.mean(mIoU)}')


if __name__ == '__main__':
    main()
