import torch.nn as nn


# CDCN: conv-deconv network

# layer_num: the range of this value is 0-4
class CDCN(nn.Module):
    def __init__(self, layer_num=0, mean=0, std=0.015):
        super(CDCN, self).__init__()
        self.layer_num = layer_num
        self.CNN = nn.Sequential(
            # 224
            nn.Conv2d(3, 96, kernel_size=7, stride=2, padding=1),
            nn.ReLU(True),
            # 110
            nn.AvgPool2d(kernel_size=3, stride=2, padding=1),
            # 55
            nn.Conv2d(96, 256, kernel_size=5, stride=2, padding=1),
            nn.ReLU(True),
            # 27
            nn.AvgPool2d(kernel_size=3, stride=2, padding=1),
            # 14

        )
        self.DCNN = nn.Sequential(
            nn.ConvTranspose2d(256, 256, kernel_size=3, stride=2, padding=1),
            # 27
            nn.ConvTranspose2d(256, 96, kernel_size=5, stride=2, padding=1),
            # 55
            nn.ConvTranspose2d(96, 96, kernel_size=4, stride=2, padding=1),
            # 110
            nn.ConvTranspose2d(96, 3, kernel_size=8, stride=2, padding=1)
            # 224)
        )
        self.init_weight(mean, std)
        #self.train_limit()

    def forward(self, x):
        x = self.CNN(x)
        x = self.DCNN(x)
        # x = x.view(x.size(0), -1)
        return x

    def init_weight(self, mean, std):
        for i, m in enumerate(self.modules()):
            if i < 2:
                continue
            if i >= self.layer_num + 2:
                break
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(mean, std)
                if m.bias is not None:
                    m.bias.data.zero_()

    def train_limit(self):
        for i, item in enumerate(self.parameters()):
            if i >= self.layer_num - 2:
                break
            item.requires_grad = False

    def get_feature(self, input, layer_num):
        i = 0
        x = input
        for layer in self.modules():
            if isinstance(layer, nn.Conv2d):
                i += 1
                x = layer(x)
                if i == layer_num:
                    return x
            if isinstance(layer, nn.ReLU):
                x = layer(x)
            if isinstance(layer, nn.AvgPool2d):
                x = layer(x)
