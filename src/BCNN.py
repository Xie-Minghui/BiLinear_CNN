import mindspore.nn as nn
from mindspore.common.initializer import Uniform
from mindspore.ops import operations as P
from .vgg import Vgg
from mindspore import load_checkpoint, load_param_into_net

class BCNN(nn.Cell):
    """
    Bilinear CNN model.
    """

    def __init__(self, image_size, num_classes=200):
        super(BCNN, self).__init__()
        self.num_classes = num_classes
        self.image_h = image_size[0]
        self.image_w = image_size[1]

        self.fc = nn.Dense(512 * 512, num_classes)

        self.relu = P.ReLU()
        self.reshape = P.Reshape()
        self.mul = P.BatchMatMul()
        self.transpose = P.Transpose()
        self.sqrt = P.Sqrt()
        self.normalize = P.L2Normalize()
        self.abs = P.Abs()

    def features(self, x):
        vgg16 = Vgg([64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
                    num_classes=1000, batch_norm=False)

        vgg16_ckpt = r'./vgg16_ascend_v120_imagenet2012_official_cv_bs32_acc73.ckpt'
        param_dict = load_checkpoint(vgg16_ckpt)
        load_param_into_net(vgg16, param_dict)

        self.vgg16 = vgg16.layers[:-1]

        out = vgg16(x)

        return out

    def construct(self, x):

        N = x.shape[0]
        assert x.shape == (N, 3, 448, 448)
        out = self.features(x)
        assert out.size() == (N, 512, 28, 28)
        # The main branch
        out = self.relu(out)
        assert out.size() == (N, 512, 28, 28)

        # Classical bilinear pooling.
        out = self.reshape(out, (N, 512, 28 * 28))
        out = self.mul(out, self.transpose(out, 1, 2)) / (28 * 28)
        assert out.size() == (N, 512, 512)
        out = self.reshape(out, (N, 512 * 512))

        # Normalization.
        #out = self.sign(out)
        out = self.sqrt(self.abs(out) + 1e-5)
        # out = self.normalize(out)

        # Classification.
        out = self.fc(out)

        return out