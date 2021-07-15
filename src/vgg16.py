import mindspore.nn as nn
from mindspore.common.initializer import initializer
import mindspore.common.dtype as mstype
import mindspore


def _make_layer(base, batch_norm):
    """Make stage network of VGG."""
    layers = []
    in_channels = 3
    for v in base:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            weight = 'ones'
            # if args.initialize_mode == "XavierUniform":
            # weight_shape = (v, in_channels, 3, 3)
            # weight = initializer('XavierUniform', shape=weight_shape, dtype=mindspore.float32)

            conv2d = nn.Conv2d(in_channels=in_channels,
                               out_channels=v,
                               kernel_size=3,
                               padding=1,
                               pad_mode='pad',
                               has_bias=False,
                               weight_init=weight)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU()]
            else:
                layers += [conv2d, nn.ReLU()]
            in_channels = v
    return nn.SequentialCell(layers)


class Vgg(nn.Cell):
    """
    VGG network definition.

    Args:
        base (list): Configuration for different layers, mainly the channel number of Conv layer.
        num_classes (int): Class numbers. Default: 1000.
        batch_norm (bool): Whether to do the batchnorm. Default: False.
        batch_size (int): Batch size. Default: 1.
        include_top(bool): Whether to include the 3 fully-connected layers at the top of the network. Default: True.

    Returns:
        Tensor, infer output tensor.

    Examples:
        >>> Vgg([64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
        >>>     num_classes=1000, batch_norm=False, batch_size=1)
    """

    def __init__(self, base, num_classes=1000, batch_norm=True, phase="train", include_top=True):
        super(Vgg, self).__init__()
        self.layers = _make_layer(base, batch_norm=batch_norm)
        self.include_top = include_top
        self.flatten = nn.Flatten()
        dropout_ratio = 0.5
        has_dropout = True
        if not has_dropout or phase == "test":
            dropout_ratio = 1.0
        self.classifier = nn.SequentialCell([
            nn.Dense(512 * 7 * 7, 4096),
            nn.ReLU(),
            nn.Dropout(dropout_ratio),
            nn.Dense(4096, 4096),
            nn.ReLU(),
            nn.Dropout(dropout_ratio),
            nn.Dense(4096, num_classes)])

    def construct(self, x):
        x = self.layers(x)
        if self.include_top:
            x = self.flatten(x)
            x = self.classifier(x)
        return x


cfg = {
    '11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    '13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    '16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    '19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

if __name__ == '__main__':
    from mindspore import load_checkpoint, load_param_into_net

    net = Vgg([64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
              num_classes=1000, batch_norm=False)
    ckpt_path = r'./vgg16_ascend_v120_imagenet2012_official_cv_bs32_acc73.ckpt'

    print(net.layers[0].weight.asnumpy())
    param_dict = load_checkpoint(ckpt_path)
    load_param_into_net(net, param_dict)
    print(net.layers[0].weight.asnumpy())
