import os
import pickle
import re

import mindspore.dataset.transforms.py_transforms as P
import mindspore.dataset.vision.c_transforms as CV
import mindspore.dataset.vision.py_transforms as PY

import numpy as np
import mindspore.dataset as ds
import PIL.Image
from src.config import config
ds.config.set_num_parallel_workers(1)

class ModelDataProcessor():

    def __init__(self):

        self.train_transforms = P.Compose([
                PY.ToPIL(),  # 需要在这里将输入转化为image形式，不知道为什么在外面转换就失败了
                PY.Resize(size=448),  # Let smaller edge match
                PY.RandomHorizontalFlip(),
                PY.RandomCrop(size=448),
                PY.ToTensor(),
                PY.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
            ])
        self.test_transforms = P.Compose([
            PY.ToPIL(),
            PY.Resize(size=448),
            PY.CenterCrop(size=448),
            PY.ToTensor(),
            PY.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        ])
        


    def extract(self):
        """Prepare the data for train/test split and save onto disk."""
        # print('Start extracting...')
        image_path = re.sub('\\\\', '/', os.path.join(config.path_data, 'CUB_200_2011/images/'))
        # Format of images.txt: <image_id> <image_name>
        id2name = np.genfromtxt(re.sub('\\\\', '/', os.path.join(
            config.path_data, 'CUB_200_2011/images.txt')), dtype=str)
        # Format of train_test_split.txt: <image_id> <is_training_image>
        id2train = np.genfromtxt(re.sub('\\\\', '/', os.path.join(
            config.path_data, 'CUB_200_2011/train_test_split.txt')), dtype=int)

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
                    open(re.sub('\\\\', '/', os.path.join(config.path_data, 'processed/train.pkl')), 'wb'))
        pickle.dump((test_data, test_labels),
                    open(re.sub('\\\\', '/', os.path.join(config.path_data, 'processed/test.pkl')), 'wb'))

    def get_data(self):
            # Now load the picked data.
        train_data, train_labels = pickle.load(open(
            os.path.join(config.path_data, 'processed/train.pkl'), 'rb'))
        # assert (len(train_data) == 5994
                # and len(train_labels) == 5994)
        test_data, test_labels = pickle.load(open(
            os.path.join(config.path_data, 'processed/test.pkl'), 'rb'))
        # assert (len(test_data) == 5794
        #         and len(test_labels) == 5794)
        
        return train_data, train_labels, test_data, test_labels 
    
    def make_batch(self, X, y, is_train=True):
        # images = []
        # for image in X:
        #     image = PIL.Image.fromarray(image)
        #     images.append(image)
        # X = images
        dataset_generator = IterDatasetGenerator(X, y)
        
        dataset = ds.GeneratorDataset(dataset_generator, ["image", "label"], shuffle=False)
        if is_train:
            dataset1 = dataset.map(operations=self.train_transforms, input_columns=["image"])
        else:
             dataset1 = dataset.map(operations=self.test_transforms, input_columns=["image"])
        # for data in dataset1.create_dict_iterator():
        #     print(data["image"], data["label"])
        dataset1 = dataset1.shuffle(5000)
        dataset1 = dataset1.batch(batch_size=config.batch_size, drop_remainder=True)
        return dataset1


class IterDatasetGenerator:
    def __init__(self, X, y):
        self.__index = 0
        self.__data = X
        self.__label = y

    def __next__(self):
        if self.__index >= len(self.__data):
            raise StopIteration
        else:
            item = (self.__data[self.__index], self.__label[self.__index])
            self.__index += 1
            return item

    def __iter__(self):
        return self

    def __len__(self):
        return len(self.__data)


if __name__ == "__main__":
    data_processor = ModelDataProcessor()
    data_processor.extract()

