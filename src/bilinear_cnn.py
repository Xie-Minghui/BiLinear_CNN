# Bilinear CNN 模型

import mindspore
import mindspore.nn as nn
import mindspore.ops as ops
from .vgg16 import Vgg
from mindspore import load_checkpoint, load_param_into_net
from .config import config

vgg16_ckpt = r'./vgg16_ascend_v120_imagenet2012_official_cv_bs32_acc73.ckpt'


class BiCNN(nn.Cell):
    def __init__(self):
        super().__init__()

        vgg16 = Vgg([64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
                    num_classes=1000, batch_norm=False)

        if not config.train_all:
            param_dict = load_checkpoint(vgg16_ckpt)
            load_param_into_net(vgg16, param_dict)

        self.sub_vgg16 = vgg16.layers[:-1]
        self.fc = nn.Dense(512 ** 2, 200)

        # 工具函数
        self.bmm = mindspore.ops.BatchMatMul()
        self.transpose = mindspore.ops.Transpose()
        self.sqrt = ops.Sqrt()
        self.l2_normalize = ops.L2Normalize(epsilon=1e-12)
        self.relu = nn.ReLU()

    def construct(self, x):
        """
        Args:
            x.shape: N*3*448*448
        Returns:
            Score.shape: N*200
        """
        N = x.shape[0]
        # print(x)
        # print("vgg")
        # assert x.shape == (N, 3, 448, 448)
        x = self.sub_vgg16(x)
        # print(x)
        # assert x.shape == (N, 512, 28, 28)
        x = self.relu(x)
        x = x.view((N, 512, 28 ** 2))
        # print("relu")
        # print(x)
        x = self.bmm(x, self.transpose(x, (0, 2, 1))) / (28 ** 2)
        # assert x.shape == (N, 512, 512)
        x = x.view(N, 512 ** 2)
        # print("mm")
        # print(x)
        x = self.sqrt(x + 1e-5)
        # print("sqrt")
        # print(x)
        x = self.l2_normalize(x)
        x = self.fc(x)
        # assert x.shape == (N, 200)
        return x


if __name__ == '__main__':
    # model:
    # input_size:(N, 3, 448, 448)
    # output_size:(N, 200)
    model = BiCNN()
    print(model.sub_vgg16[1].parameters_dict())

    # import numpy as np
    # x = Tensor(np.random.sample((5, 3, 448, 448)), ms.float32)
    # y = model(x)
    # print(y.shape)
