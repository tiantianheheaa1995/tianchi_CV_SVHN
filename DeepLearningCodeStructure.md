一个简单的深度学习工程结构，整体可以分为**数据data**、**神经网络模型model**和
**训练测试过程train&test**。
### 1、数据data可以分为DataSet和DataLoader两部分   
torch.utils.data中有两个类一个是DataSet类，一个是DataLoader类。这也是我们定义自己的数据集，需要做的两步。  
**1.1 DataSet**  
对于一些常用的数据集可以使用torchvision.datasets中提供的数据集。对于自己的数据集，可以继承data.DataSet类，写自己的DataSet类，重写init、getitem和len三个函数。对于不同的数据集，有不同的处理。例如，VOC数据集、COCO数据集、SVHM数据集等。  
**init函数**  
需要传入的参数是img_root(图片地址)、label_root(标注地址)、transform(图像变换)。<br>
init函数做的工作，就是根据img_root返回img_path图片列表，并进行排序。根据label_root返回label_path标注列表，并进行排序。定义好各种数据集的transform。<br>
函数返回值是排序后的img_path图片列表和label_path标注列表。  

对于img_path和label_path可以有两种写法。  
一种是使用os.path.join(路径连接)和os.listdir
(获得给定目录下的文件和文件夹名字列表)。
```python
img_path = [os.path.join(root, img) for img in os.listdir(root)]
```
一种是使用glob.glob(匹配所有符合条件的文件，并以list形式返回)。  
```python
img_path = glob.glob('root/*.jpg')
```
img_path和label_path获取后，需要sort或sorted排序，一般图片的名字就是序号。  

对于transform可以有多种写法。一是在init函数中定义，二是在创建DataSet对象的时候定义。更好的做法是在init函数中定义默认的transform，然后在创建dataset对象时可以使用默认参数，也可以传入新的参数定义新的变换。  
一般来说，训练集、验证集和测试集的transform是不同的。在创建train_dataset和test_dataset
的时候，可以定义不同的transform传入。在init中定义默认的transform时，可以设置train和test
标签来区分不同的数据集的transform。  
```python
class MyDataSet(data.Dataset):
    ...
    if transform is None:
        if self.train:
            self.transform = ...
        if self.test:
            self.transform = ...
    ...
```
**getitem函数**  
需要传入的参数是图片的index。<br>
getitem函数做的工作就是根据index，从img_path列表中获得一
张图片img_path[index]，然后对图像进行变换。根据index从label_path中获得该图片的label，即label_path[index]。<br>
函数返回值是每张图像变换后的图片和对应的label。  

img_path[index]返回的图片可以用Image.open()打开（PIL库）或者OpenCV方式打开，然后用上面定义好的transform
进行图像变换。  
label_path[index]返回的该图片的label。  

**len函数**  
返回的是img_path列表中的元素个数，即数据集中样本的总数。

**1.2 DataLoader**  
传入的参数是dataset，batch_size，shuffle(数据集是否打乱)，num_workers(加载数据使用的进程数)，pin_memory(是否使用锁页内存)。

### 2、神经网络模型model  
**2.1 预训练模型**  
torchvision.models提供了一些经典的神经网络结构和预训练模型(pretrained=True)。  
**2.2 自己定义的模型**  
继承nn.Module，重写init函数，定义网络的各层结构。重写fordward函数，定义前传播过程。  
有以下几点需要注意：  
（1）如果网络的上一层的输入直接传递到下一层，可以用nn.Sequential快速定义网络。从而不需要
在forward函数中写前向传递过程，默认是一层一层直接向前传递。  
(2)对于图像分类分类网络，用CNN进行特征提取，用全连接网络把特征维度映射到标签维度，进行分类。中间需要经过维度转换，即把所有通道的特征展开为一维向量。  
(3)对于模块化网络，例如Inception，ResNet，FPN等，可以先定义子模块，然后根据子模块构建整个网络。  
(4)对于各层网络的定义，需要学习参数的层用nn.Module(自动提取可学习参数)。不需要学习参数
的层，可以用nn.Module(在init函数中定义)，可以用nn.functional(在forward函数中定义)。


### 3、训练和测试过程  
**3.1 train和val的过程**   
1、数据准备，train_dataset，train_dataloader，val_dataset，val_dataloader。  
2、模型准备。  
3、损失函数和优化器。  
4、开始训练，外层循环是epoch，内层循环是dataloader。
```python
for epoch in range(10):
    model.train()
    for ii, (input, label) in enumerate(train_dataloader):
        ...
    model.eval()
    for ii, (input, label) in enumerate(val_dataloaer):
        ...
```
train_dataloader循环中，需要前向传播，梯度清零，计算损失，反向传播，参数更新。  
val_dataloader循环中，需要前向传播，计算损失。  
还有一种写法，把train_dataloader循环和val_dataloader循环，写成两个单独的函数。从而主函数只有一层epoch循环。

**3.2 val评测指标**<br>
**3.2.1 图片分类**<br>
score = 分类正确的图片数 / 测试集图片总数<br>
**基本指标**（二分类）：  
TP，True Positive，标签是正样本，分类是正样本。  
FN，False Negative，标签是正样本，分类是负样本。  
FP，False Positive，标签是负样本，分类是正样本。  
TN，True Negative，标签是负样本，分类是负样本。
对于多分类，标签对于的类别是正样本，其他的类别都是负样本。预测类别是预测概率最大的那个类别。  
**准确率Accuracy**  
Accuracy = 所有正确分类的样本 / 全部样本数 = (TP+TN) / (TP+FP+TN+FN)  
具有可以分为Top_1Accuracy和Top_5Accuracy。  
Top_1Accuracy就是上面介绍的，预测概率最大的那个类别和标签类别相同，就判断分类正确。  
Top_5Accuracy是每次预测概率最大的5个类别中包含标签类别，就判断为分类正确。从而Top_5Accuracy
指标更宽松一些，对应的score更高一些。  
**精确度Precision和召回率Recall**  
正样本的精确度Precision = TP / (TP + FP)，表示找回为正样本的样本中，有多少真正的正样本。  
正样本的召回率Recall = TP / (TP + FN)，表示所有的正样本中，有多少是被召回的。  
同样可以计算负样本的精确度和召回率。  
通常召回率越高，精确度越低。可以用PR曲线和坐标轴包围的面积来评价一个模型的好坏。  
**F1 score**  
如果同时关注精确度Precision和召回率Recall，F1 score是一个综合性能的指标。  
F1 score = 2 * Precision * Recall / (Precision + Recall)  
只有当精确度Precision和召回率Recall都很高的情况下，F1 score才会很高。  
**混淆矩阵Confusion Matrix**  
如果想要知道多个类别之间错分的情况，可以计算混淆矩阵。  
例如对于20类别的分类任务，混淆矩阵是20*20的矩阵。第i行和第j列表示第i类目标被分为第j类的概率。
混淆矩阵的对角线的值越大，表示分类器的性能越好。<br>

**ROC曲线和AUC指标**<br>
上面的准确率Accuracy，精确率Precison，召回率Recall, F1 score，混淆矩阵Confusion Matrix
都是单一数值指标。 如果想要观察分类算法在不同参数/阈值下的表现情况，就可以使用ROC曲线。  
ROC曲线中，每个点的横坐标是FPR(False Positive Rate)，纵坐标是TPR(True Positive Rate)。  
TPR = TP / (TP + FN)，表示预测的正样本中，实际正样本占所有正样本的比例。  
FPR = FP / (FP + TN)，表示预测的正样本中，实际负样本占所有负样本的比例。  
ROC曲线中的4个关键点：  
(0,0)，FPR = TPR = 0，分类器预测所有的样本都是负样本。  
(1,1)，FPR = TPR = 1，分类器预测所有的样本都是正样本。  
(0,1)，FPR = 0，TPR = 1，此时FN = 0，FP = 0，所有的样本都分类正确。  
(1,0)，FPR = 1，TPR = 0，此时TP = 0，TN = 0，所有样本分类错误。  
ROC曲线和PR曲线比较：  
ROC曲线对于正负样分布不均衡问题不敏感，即当测试集中正负样本分布发生变化时，ROC曲线保持不变。
例如负样本数据增大为原来的10倍，TPR不受影响，FPR也是成比例增加，不会有太大变化。  
AUC指标  
如果用ROC曲线定量评估两个分类器的性能，可以用AUC曲线，即ROC曲线和坐标轴围成的面积。

**3.2.2 字符识别**<br>
本次赛题街景字符识别使用的评测标准是Accuracy： score = 字符串正确识别的图片数 / 测试集图片总数

**3.2.3 目标检测**<br>
**IoU**  
IoU表示两个矩阵框的面积的交集和并集的比值。  
IoU = A ∩ B / A ∪ B   
一般设置IoU阈值为0.5，就认为召回。 IoU阈值越高，召回率就会下降，定位框也越精确。

**AP和mAP**  
AP是Average Precision，连续曲线求积分求平均值，即PR曲线下的面积。  
目标检测任务中，目标的数量可能有多个。预测的目标id越多，Recall就越高，Precision就越低。
从Top_1到Top_N都统计一遍，就得到了目标检测中的Precision和Recall，进而得到PR曲线。  
目标检测中的PR曲线和图片分类中的PR曲线的意义和整体趋势相同，但是计算方式却不同。

VOC2007计算AP, 将recall坐标轴进行10等分（11个点），每个recall节点，取大于该recall值的
最大precison，取11个点的precision值平均得到AP。<br>
VOC 2010提出了更精确的计算方法，不是固定的11个点计算，而是取所有不同的recall点的最大
precision，进行平均得到AP。  
假设有N个id，有M个label，取M个recall节点，然后从0到1进行1/M等间距划分，对每个recall节点，计算大于该recall值的最大precision。然后对M个precision值取平均获得AP值。

AP衡量的是模型在一个类别上的好坏，mAP是所有类别AP值的平均值，衡量的是模型在所有类别上的好坏。


**3.3 test过程**  
1、数据准备，test_dataset和test_dataloader。  
2、模型准备，用上面训练结果最好的参数初始化模型。  
3、只有一层循环test_dataloader，需要做的是前向传播。
