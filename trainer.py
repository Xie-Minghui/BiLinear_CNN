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
import mindspore.dataset.transforms.py_transforms as P
import mindspore.dataset.vision.c_transforms as CV
import mindspore.dataset.vision.py_transforms as PY
import mindspore.nn as nn
import mindspore

import os, sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__)) # __file__获取执行文件相对路径，整行为取上一级目录
sys.path.append(BASE_DIR)

from src.config import config
# from src.bilinear_cnn import BCNN
from src.dpcnn import DPCNN
from src.cub200A import ModelDataProcessor

ms.common.set_seed(0)
# from mindspore import context
# context.set_context(mode=context.GRAPH_MODE, device_target="GPU")

class Trainer:
    def __init__(self):
        """Prepare the network, criterion, solver, and data.
        Args:
            options, dict: Hyperparameters.
        """
        print('Prepare the network and data.')
        # Network.
        self.model = DPCNN()
        self.data_processor = ModelDataProcessor()
        # Criterion.
        self.criterion = nn.SoftmaxCrossEntropyWithLogits(sparse=True,reduction='sum')
        self.optimizer = nn.Adam(self.model.trainable_params(), learning_rate=config.lr)
        self.X_train, self.y_train, self.X_test, self.y_test = self.data_processor.get_data()

    def controller(self):

        for epoch in range(config.epochs):
            print("Epoch: {}".format(epoch))
            self.train()

    def train(self):

        net_with_criterion = nn.WithLossCell(self.model, self.criterion)

        train_network = nn.TrainOneStepCell(net_with_criterion, self.optimizer)
        train_network.set_train()

        loss_total = 0.0
        correct_total = 0
        print("haha")
        self.data_processor.make_batch(self.X_train, self.y_train)  
        # for x_batch, y_batch in self.data_processor.get_batch(self.X_train, self.y_train, is_train=True):  # self.data_processor.
        #     x_batch = Tensor(x_batch, mindspore.int32)
        #     y_batch = Tensor(y_batch, mindspore.int32)
        #     print(x_batch)
 
        #     loss = train_network(x_batch, y_batch)
        #     loss_total += float(loss.asnumpy())
        #     predict = self.model(x_batch).asnumpy().argmax(1)
        #     correct = 0
        #     for i, j in zip(predict, y_batch):
        #         correct += (i==j.asnumpy())
        #     correct_total += correct

        # loss_total_final = loss_total
        # accuracy = correct_total / len(self.X_train)
        # print("train loss: {}, train accuracy: {}".format(loss_total_final, accuracy))


if __name__ == "__main__":

    trainer = Trainer()
    trainer.controller()
