import os
import keras
import gc
import numpy as np
import tensorflow as tf

from data_generator.data_generator import COCODataLoader
from models.mobilenet_unet import MobilenetV2_base, relu6
from utils.utils import visualize, bce_dice_loss, iou_metric, iou_for_image, get_precision, get_multi_threshold_precision

from utils.utils import iou_metric



mobilenet = MobilenetV2_base()
mobilenet.build_model(keras.layers.Input(shape=(400,400,3)))


mobilenet.model = keras.models.load_model('./checkpoints/mobilenet400_iou_no_dil-102_loss-0.4221_val_loss-0.3312.h5',
                                              custom_objects={'relu6' : relu6,
                                                              'bce_dice_loss' : bce_dice_loss,
                                                              'iou_metric' : iou_metric})

val_generator = COCODataLoader(
                    path_to_annotations='/home/matsvei.rozanau/hdd/datasets/coco_dataset/annotations/instances_val2017.json',
                    path_to_images='/home/matsvei.rozanau/hdd/datasets/coco_dataset/val2017/',
                    batch_size=16,
                    resize=(400,400),
                    augmentations=False,
                    shuffle=False)


thresholds = np.arange(0.5, 1, 0.05)
# Calculate mIoU over all validation batches
mIoU = np.array([])

for i in range(0, len(val_generator)):
    print(f"Processing batch {i}")
    images, masks = val_generator[i]
    IoU = np.zeros((16,1))
    for threshold in thresholds:
        pred_mask = mobilenet.model.predict(images)
        pred_mask = keras.backend.cast(pred_mask, dtype=tf.float64)
        pred_mask = keras.backend.squeeze(pred_mask, axis=-1)
        masks = np.squeeze(masks)
        iou_over_threshold = np.reshape(get_precision(masks, pred_mask, threshold), (-1,1))
        mean_iou_over_threshold = np.concatenate((IoU, iou_over_threshold), axis=1)
    mean_iou_over_threshold = np.mean(get_multi_threshold_precision(IoU[:, 1:]))
    mIoU = np.append(mIoU, mean_iou_over_threshold)
    with open('./results/metrics_log.txt', 'a') as f:
        f.write(f"mIoU on {i} batch = {mIoU.mean()}\n")