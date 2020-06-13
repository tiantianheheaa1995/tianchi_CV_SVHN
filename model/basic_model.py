import torch
from torch import nn
import torchvision.models as models

class SVHM_Model1(nn.Module):
    def __init__(self):
        super(SVHM_Model1, self).__init__()

        # CNN，提取特征
        model_conv = models.resnet34(pretrained=True)
        model_conv.avgpool = nn.AdaptiveAvgPool2d(1)  # 更改了resnet中的avgpool层
        model_conv = nn.Sequential(*list(model_conv.children())[:-1])  # children获得网络的子层，list封装一下，[:-1]遍历全部元素
        self.cnn = model_conv

        # 全连接网络，分类
        self.fc1 = nn.Linear(512, 11)
        self.fc2 = nn.Linear(512, 11)
        self.fc3 = nn.Linear(512, 11)
        self.fc4 = nn.Linear(512, 11)
        # self.fc5 = nn.Linear(512, 11)

    def forward(self, img):  # img表示输入，用x表示也行
        feat = self.cnn(img)
        feat = feat.view(feat.shape[0], -1)

        c1 = self.fc1(feat)
        c2 = self.fc2(feat)
        c3 = self.fc3(feat)
        c4 = self.fc4(feat)
        # c5 = self.fc5(feat)

        return c1, c2, c3, c4