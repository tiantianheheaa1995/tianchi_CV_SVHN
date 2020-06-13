import torch
from torch.utils import data
from PIL import Image
from PIL import ImageChops
from PIL import ImageFilter
import numpy as np
import torchvision.transforms as transforms
import skimage
from skimage import util


class SVHNDataset(data.Dataset):
    def __init__(self, img_path, img_label, transform = None, train = True, test = False):
        self.img_path = img_path
        self.img_label = img_label
        self.test = test
        self.train = train

        if transform is None:
            if self.test or not train:  # 验证集和测试集的图像变换
                self.transform = transforms.Compose([
                    transforms.Resize((60, 120)),
                    transforms.ToTensor(),  # Image->Tensor
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet数据的标准化
                    ])
            else:  # 训练集的图像变换
                self.transform = transforms.Compose([
                    transforms.Resize((64, 128)),
                    transforms.RandomCrop((60, 120)),  # 随机裁剪为60*120
                    transforms.ColorJitter(0.3, 0.3, 0.2),  # 亮度，对比度，饱和度调节
                    transforms.RandomRotation(10),  # 随机旋转-5~5°，旋转角度不可太大。目标检测中bbox框也要跟着旋转
                    transforms.ToTensor(),  # Image->Tensor
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet数据的标准化
                    ])

    def __getitem__(self, index):
        img = Image.open(self.img_path[index]).convert('RGB')
        if self.train and not self.test:
            img = img.resize((128, 64))
            img = ImageChops.offset(img, np.random.randint(-35, 35), np.random.randint(-10, 10))  # 平移
            if np.random.randint(2):
                # img = np.array(img)
                # img = util.random_noise(img, mode='gaussian')  # 添加噪声
                # img = np.uint8(img*255);
                # img = Image.fromarray(img)
                img = img.filter(ImageFilter.BLUR)  # 图像模糊

        img = self.transform(img)

        lbl = np.array(self.img_label[index], dtype = np.int)
        lbl = list(lbl) + (4-len(lbl)) * [10]  # label长度不足5的部分，赋值为10。

        # getitem函数返回的是每张变换后的图片和对应的label
        return img, torch.from_numpy(np.array(lbl[:4]))  # 获取label的前5个字符，转换为Tensor，返回

    def __len__(self):
        # img_path是个列表，可以通过len返回元素数量
        return len(self.img_path)


