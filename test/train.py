import os
import re
import time
import random
from ast import literal_eval as liter
import numpy as np
import mindspore as ms
from mindspore import Tensor
from mindspore import context
from mindspore import dataset as de
from mindspore.communication.management import init, get_rank
from mindspore.nn.loss import SoftmaxCrossEntropyWithLogits
from mindspore.nn.metrics import Accuracy
from mindspore.nn.optim.adam import Adam
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig, LossMonitor, TimeMonitor
from mindspore.train.model import Model, ParallelMode
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from src.dataset_create import CUB200
import mindspore.dataset.transforms.py_transforms as P
import mindspore.dataset.vision.c_transforms as CV
import mindspore.dataset.vision.py_transforms as PY

ms.common.set_seed(0)

def train(project_root, path):

    train_transforms =P.Compose([
        CV.Resize(size=[448,448]),  # Let smaller edge match
        PY.RandomHorizontalFlip(),
        PY.RandomCrop(size=[448,448]),
        PY.ToTensor(),
        PY.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])
    test_transforms = P.Compose([
        CV.Resize(size=[448,448]),
        PY.CenterCrop(size=[448,448]),
        PY.ToTensor(),
        PY.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])

    train_loader = CUB200(
        root=path['cub200'], train=True, download=True)
    test_loader = CUB200(
        root=path['cub200'], train=False, download=True)

    def train_iter():
        for i in range(len(train_loader._train_data)):
            yield (train_loader._train_data[i], train_loader._train_labels[i])

    def test_iter():
        for i in range(len(test_loader._test_data)):
            yield (test_loader._test_data[i], test_loader._test_labels[i])

    train_data = de.GeneratorDataset(train_iter, ["image", "label"])
    test_data = de.GeneratorDataset(test_iter, ["image", "label"])

    #数据增强
    train_data = train_data.map(operations=train_transforms, input_columns=["image"])
    test_data = test_data.map(operations=test_transforms, input_columns=["image"])

if __name__ == "__main__":

    project_root = os.getcwd()
    path = {
        'cub200': re.sub('\\\\', '/', os.path.join(project_root, 'data/cub200')),
        'model': re.sub('\\\\', '/', os.path.join(project_root, 'model')),
    }

    train(project_root, path)
