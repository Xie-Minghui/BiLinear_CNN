# -*- coding: utf-8 -*
"""This module is served as torchvision.datasets to load CUB200-2011.
CUB200-2011 dataset has 11,788 images of 200 bird species. The project page
is as follows.
    http://www.vision.caltech.edu/visipedia/CUB-200-2011.html
- Images are contained in the directory data/cub200/raw/images/,
  with 200 subdirectories.
- Format of images.txt: <image_id> <image_name>
- Format of train_test_split.txt: <image_id> <is_training_image>
- Format of classes.txt: <class_id> <class_name>
- Format of iamge_class_labels.txt: <image_id> <class_id>
This file is modified from:
    https://github.com/vishwakftw/vision.
"""


import os
import pickle
import re

import mindspore.dataset.transforms.py_transforms as P
import mindspore.dataset.vision.c_transforms as CV
import mindspore.dataset.vision.py_transforms as PY

import numpy as np
import PIL.Image
from config import config


class ModelDataProcessor():

    def __init__(self):
        self.train_transforms =P.Compose([
                CV.Resize(size=[448,448]),  # Let smaller edge match
                PY.RandomHorizontalFlip(),
                PY.RandomCrop(size=[448,448]),
                PY.ToTensor(),
                PY.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
            ])
        self.test_transforms = P.Compose([
            CV.Resize(size=[448,448]),
            PY.CenterCrop(size=[448,448]),
            PY.ToTensor(),
            PY.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        ])


    def extract(self):
        """Prepare the data for train/test split and save onto disk."""
        # print('Start extracting...')
        image_path = re.sub('\\\\', '/', os.path.join(config.path_data, 'raw/CUB_200_2011/images/'))
        # Format of images.txt: <image_id> <image_name>
        id2name = np.genfromtxt(re.sub('\\\\', '/', os.path.join(
            self._root, 'raw/CUB_200_2011/images.txt')), dtype=str)
        # Format of train_test_split.txt: <image_id> <is_training_image>
        id2train = np.genfromtxt(re.sub('\\\\', '/', os.path.join(
            self._root, 'raw/CUB_200_2011/train_test_split.txt')), dtype=int)

        train_data = []
        train_labels = []
        test_data = []
        test_labels = []
        for id_ in range(id2name.shape[0]):
            image = PIL.Image.open(re.sub('\\\\', '/', os.path.join(image_path, id2name[id_, 1])))
            label = int(id2name[id_, 1][:3]) - 1  # Label starts with 0

            # Convert gray scale image to RGB image.
            if image.getbands()[0] == 'L':
                image = image.convert('RGB')
            image_np = np.array(image)
            image.close()

            if id2train[id_, 1] == 1:
                train_data.append(image_np)
                train_labels.append(label)
            else:
                test_data.append(image_np)
                test_labels.append(label)

        pickle.dump((train_data, train_labels),
                    open(re.sub('\\\\', '/', os.path.join(self._root, 'processed/train.pkl')), 'wb'))
        pickle.dump((test_data, test_labels),
                    open(re.sub('\\\\', '/', os.path.join(self._root, 'processed/test.pkl')), 'wb'))

    def get_data(self):
            # Now load the picked data.
        train_data, train_labels = pickle.load(open(
            os.path.join(config.path_data, 'processed/train.pkl'), 'rb'))
        assert (len(train_data) == 5994
                and len(train_labels) == 5994)
        test_data, test_labels = pickle.load(open(
            os.path.join(config.path_data, 'processed/test.pkl'), 'rb'))
        assert (len(test_data) == 5794
                and len(test_labels) == 5794)
        
        return train_data, train_labels, test_data, test_labels 

    def get_batch(self, x, y, is_train=True):
        assert len(x) == len(y) , "error shape!"

        n_batches = int(len(x) / config.batch_size)  # 统计共几个完整的batch
        for i in range(n_batches - 1):
            x_batch = x[i*config.batch_size: (i + 1)*config.batch_size]
            y_batch = y[i*config.batch_size: (i + 1)*config.batch_size]
            x_batch = self.transform(x_batch, is_train)
            y_batch = self.transform_target(y_batch)
            lengths = [len(seq) for seq in x_batch]
            max_length = max(lengths)
            for i in range(len(x_batch)):
                x_batch[i] = x_batch[i] + [0 for j in range(max_length-len(x_batch[i]))]

            yield x_batch, y_batch

    def transform(self, x_batch, is_train):
        images = []
        for image in x_batch:
            image =  PIL.Image.fromarray(image)
            if is_train:
                image = self.train_transforms(image)
            else:
                image = self.test_transforms(image)
            images.append(image)
        return images


if __name__ == "__main__":
    data_processor = ModelDataProcessor()
    data_processor.extract()

