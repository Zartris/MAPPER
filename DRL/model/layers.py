import torchvision
from torch import nn


class ConvBNReLU(nn.Module):
    def __init__(self, in_chan, out_chan, ks: int = 3, stride=1, padding=1,
                 dilation=1, groups=1, bias=False):
        """

        :param in_chan: number of channels (for remembering: 1 is gray, 3 is color)
        :param out_chan:
        """
        super(ConvBNReLU, self).__init__()
        self.conv = nn.Conv2d(
            in_chan, out_chan,
            kernel_size=ks,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias)
        self.bn = nn.BatchNorm2d(out_chan)
        self.relu = nn.ReLU6(inplace=True)

    def forward(self, x):
        feat = self.conv(x)
        feat = self.bn(feat)
        feat = self.relu(feat)
        return feat


def resnet_backbone_conv(backbone_name: str):
    # backbone
    if backbone_name == 'resnet_18':
        resnet_net = torchvision.models.resnet18(pretrained=True)
        modules = list(resnet_net.children())[:-2]
        backbone = nn.Sequential(*modules)
    elif backbone_name == 'resnet_34':
        resnet_net = torchvision.models.resnet34(pretrained=True)
        modules = list(resnet_net.children())[:-2]
        backbone = nn.Sequential(*modules)

    elif backbone_name == 'resnet_50':
        resnet_net = torchvision.models.resnet50(pretrained=True)
        modules = list(resnet_net.children())[:-2]
        backbone = nn.Sequential(*modules)

    elif backbone_name == 'resnet_101':
        resnet_net = torchvision.models.resnet101(pretrained=True)
        modules = list(resnet_net.children())[:-2]
        backbone = nn.Sequential(*modules)

    elif backbone_name == 'resnet_152':
        resnet_net = torchvision.models.resnet152(pretrained=True)
        modules = list(resnet_net.children())[:-2]
        backbone = nn.Sequential(*modules)

    elif backbone_name == 'resnet_50_modified_stride_1':
        resnet_net = resnet50(pretrained=True)
        modules = list(resnet_net.children())[:-2]
        backbone = nn.Sequential(*modules)

    elif backbone_name == 'resnext101_32x8d':
        resnet_net = torchvision.models.resnext101_32x8d(pretrained=True)
        modules = list(resnet_net.children())[:-2]
        backbone = nn.Sequential(*modules)
    return backbone
