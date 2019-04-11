import os
import keras
import numpy as np
import skimage.transform as skt
from pycocotools.coco import COCO
from keras.utils import Sequence
from keras.preprocessing.image import img_to_array, load_img

from albumentations import (
    PadIfNeeded,
    HorizontalFlip,
    Compose,
    ElasticTransform,
    GridDistortion,
    RandomSizedCrop,
    OneOf,
    RandomBrightnessContrast,
    RandomGamma    ,
    ShiftScaleRotate,
)


class COCODataLoader(Sequence):
    """
    """

    def __init__(self, batch_size,
                 path_to_annotations,
                 path_to_images,
                 resize=(480,480),
                 shuffle=True,
                 augmentations=True):
        print(f'\nLoading COCO dataset from {path_to_images}')
        self.dataset = COCO(path_to_annotations)
        self.path_to_images = path_to_images
        self.batch_size = batch_size
        self.categories_ids = self.dataset.getCatIds(catNms=['person'])
        self.images_ids = self.dataset.getImgIds(catIds=self.categories_ids)
        self.images_descriptions = np.asanyarray(self.dataset.loadImgs(self.images_ids))
        self.resize = resize
        self.augmentations = augmentations
        self.shuffle = shuffle

        if self.shuffle:
            np.random.shuffle(self.images_descriptions)


    def __len__(self):
        return int(np.ceil(len(self.images_ids) / float(self.batch_size)))


    def __getitem__(self, index):
        images = []
        masks = []
        batch_descriptions = self.images_descriptions[index * self.batch_size: (index + 1) * self.batch_size]

        for desc in batch_descriptions:
            # Load image
            image_filename = os.path.join(self.path_to_images, desc['file_name'])
            image = img_to_array(load_img(image_filename))

            # Load masks for this image
            batch_annotations_ids = self.dataset.getAnnIds(imgIds=desc['id'], catIds=self.categories_ids, iscrowd=None)
            annotations = self.dataset.loadAnns(batch_annotations_ids)
            mask = self.dataset.annToMask(annotations[0])
            for i in range(len(annotations)):
                mask += self.dataset.annToMask(annotations[i])

            # Make binary mask
            mask = np.where(mask >= 1., 1., 0)

            # Resize image and mask
            if self.resize:
                image = skt.resize(image, self.resize, anti_aliasing=False)
                image /= 255.0
                mask = np.round(skt.resize(mask, self.resize, anti_aliasing=False))

            # Augmentations
            if self.augmentations:
                aug = Compose([
                    HorizontalFlip( p=.45),
                    RandomSizedCrop(p=.15, min_max_height=(10, 220), height=self.resize[0], width=self.resize[1]),
                    GridDistortion( p=.1, border_mode=0, distort_limit=0.1),
                    ElasticTransform(p=.1, alpha=10, sigma=120 * 0.5, alpha_affine=120 * 0.05),
                    ShiftScaleRotate(p=.3, border_mode=0, shift_limit=0.04, scale_limit=0.05),
                    OneOf([
                        RandomBrightnessContrast(p=.3),
                        RandomGamma(p=.3)
                    ], p=.3)
                ])
                augmented = aug(image=image, mask=mask)
                image = augmented['image']
                mask = augmented['mask']

            images.append(image)
            masks.append(mask)

        return np.asanyarray(images), np.expand_dims(masks, axis=-1)


    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.images_descriptions)