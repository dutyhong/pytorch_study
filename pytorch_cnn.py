import torch

# empty_tensor = torch.empty(5, 3)
# print(empty_tensor)
# data_tensor = torch.tensor([[1,3,3],[4,5,6]])
# data_shape = data_tensor.shape
# for x in data_shape:
#     print(x)
# cnt = data_shape.count(1)
# index = data_shape.index(3)
# print(data_shape)


x = torch.randn(32, 64)
y = torch.randn(32, 10)
##构建网络
model = torch.nn.Sequential(torch.nn.Linear(64, 1000), torch.nn.ReLU(), torch.nn.Linear(1000, 10))
##定义loss
loss_fn = torch.nn.MSELoss(reduction='sum')
##定义优化器
optim = torch.optim.Adam(model.parameters(), lr=1e-4)
##循环迭代
for i in range(500):
    y_pred = model(x)
    loss = loss_fn(y_pred, y)
    print(i, loss.item())
    optim.zero_grad()
##loss反向计算梯度
    loss.backward()
##根据优化器更新梯度
    optim.step()


torch.nn.RNN(10, 32, 2)







import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # 制定GPU
EPOCHES = 5

def show(img):
    """输入是一个batch的图片"""
    img = img / 2 + 0.5
    npimg = img.numpy()  # 输入是tensor，转成array
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # 输入图像channel: 1, 输出channel: 6; 5*5卷积核
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        # 仿射函数 y = Wx +b 也就是全连接
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def train(model, trainLoader, criterion, optimzer):
    for epoch in range(EPOCHES):
        running_loss = 0.0
        for i, data in enumerate(trainLoader, 0):
            # get the batch
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)  # 将输入送入GPU
            # 清空所有参数的梯度缓存
            optimzer.zero_grad()
            # forward + backward + optimze
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimzer.step()  # 更新参数 update parameter
            # print statistics
            running_loss += loss.item()
            # print every 2000 mini-batches 的平均损失
            if i % 2000 == 1999:
                print('[%d, %5d] loss: %.2f' % (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0
    print("Finish Trained")

def test(model, testLoader, classes):
    correct = 0
    total = 0
    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))
    # 也可以通过将代码块包装在 with torch.no_grad(): 中，来阻止autograd跟踪设置了 .requires_grad=True 的张量的历史记录。
    # 即只做计算，不更新计算图
    with torch.no_grad():
        for data in testLoader:
            images, labels = data
            inputs, labels = inputs.to(device), labels.to(device)  # 将输入送入GPU
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)  # 输入参数：0是每列的最大值，1是每行的最大值；输出：第一个是每行的最大值，第二个是最大值的索引
            total + labels.size()
            correct += (predicted == labels).sum().item()
            c = (predicted == labels).squeeze()
            for i in range(4):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1
    for i in range(10):
        print('Accuracy of %5s : %2d %%' % (classes[i], 100 * class_correct[i] / class_total[i]))
    print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))


def app():
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    trainSet = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    trainLoader = torch.utils.data.DataLoader(trainSet, batch_size=4, shuffle=True, num_workers=2)
    testSet = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    testLoader = torch.utils.data.DataLoader(testSet, batch_size=4, shuffle=False, num_workers=2)
    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    dataIter = iter(trainLoader)  # 生成一个迭代器
    images, labels = dataIter.next()
    # 将输入送入GPU

    # 显示图片
    show(torchvision.utils.make_grid(images))
    # 打印图片标签
    print(' '.join('%5s' % classes[labels[j]] for j in range(4)))

    # 定义模型，并使用GPU
    model = CNN()
    model.to(device)

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimzer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    # 训练模型
    train(model, trainLoader, criterion, optimzer)



if __name__ == '__main__':
    app()