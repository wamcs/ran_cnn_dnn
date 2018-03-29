import torch.nn as nn


# CDCN: conv-deconv network

# layer_num: the range of this value is 0-4
class CDCN(nn.Module):
    def __init__(self, layer_num=0):
        super(CDCN, self).__init__()
        self.train_limit()
        self.layer_num = layer_num
        self.layers = nn.Sequential(
            # 224
            nn.Conv2d(3, 96, kernel_size=7, stride=2, padding=1),
            # nn.ReLU(True),
            # 110
            nn.AvgPool2d(kernel_size=3, stride=2, padding=1),
            # 55
            nn.Conv2d(96, 256, kernel_size=5, stride=2, padding=1),
            # nn.ReLU(True),
            # 27
            nn.AvgPool2d(kernel_size=3, stride=2, padding=1),
            # 14
            nn.ConvTranspose2d(256, 256, kernel_size=3, stride=2, padding=1),
            # 27
            nn.ConvTranspose2d(256, 96, kernel_size=5, stride=2, padding=1),
            # 55
            nn.ConvTranspose2d(96, 96, kernel_size=4, stride=2, padding=1),
            # 110
            nn.ConvTranspose2d(96, 3, kernel_size=8, stride=2, padding=1)
            # 224
        )

    def forward(self, x):
        x = self.layers(x)
        # x = x.view(x.size(0), -1)
        return x

    def train_limit(self):
        for i, item in enumerate(self.parameters()):
            if i >= self.layer_num:
                break
            item.requires_grad = False
