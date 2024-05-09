import copy,torch,torchvision,os,warnings,math
import torchvision.transforms as transforms
from torchvision import models
from torch.utils.data import DataLoader
from torch import optim, nn
from torch.optim import lr_scheduler
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import f1_score,classification_report, confusion_matrix
import pandas as pd
import seaborn as sns


warnings.filterwarnings("ignore")

plt.rcParams['axes.unicode_minus'] = False

# set GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

import torch
from torchvision import datasets, transforms
import os

# dataset root
data_dir = "/home/kingargroo/seed/ablation1/allclass/images"

# 图像的大小
img_height = 640
img_width = 640

# 数据预处理
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(img_height),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize((img_height, img_width)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

# 加载数据集
full_dataset = datasets.ImageFolder(data_dir)

# 获取数据集的大小
full_size = len(full_dataset)
train_size = int(0.7 * full_size)
test_size=int(0.15*full_size)
val_size = full_size - train_size-test_size


torch.manual_seed(0)
train_dataset, val_dataset,test_dataset = torch.utils.data.random_split(full_dataset, [train_size,val_size,test_size])

# 将数据增强应用到训练集
train_dataset.dataset.transform = data_transforms['train']

# 创建数据加载器
batch_size = 32
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

dataloaders = {'train': train_dataloader, 'val': val_dataloader,'test': test_dataset}
dataset_sizes = {'train': len(train_dataset), 'val': len(val_dataset),'test': len(test_dataset)}
class_names = full_dataset.classes
print(class_names)
# ShuffleNet model
model = models.shufflenet_v2_x1_0(pretrained=True)
num_ftrs = model.fc.in_features

# 根据分类任务修改最后一层
model.fc = nn.Linear(num_ftrs, len(class_names))

model = model.to(device)

#print summary
print(model)

# loss function
criterion = nn.CrossEntropyLoss()

# optimizer
optimizer = optim.Adam(model.parameters())

# learning rate
exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

#  begin train
num_epochs = 10
best_model_wts = copy.deepcopy(model.state_dict())
best_acc = 0.0

# 初始化记录器
train_loss_history = []
train_acc_history = []
val_loss_history = []
val_acc_history = []

for epoch in range(num_epochs):
    print('Epoch {}/{}'.format(epoch, num_epochs - 1))
    print('-' * 10)

    # 每个epoch都有一个训练和验证阶段
    for phase in ['train', 'val']:
        if phase == 'train':
            model.train()  # Set model to training mode
        else:
            model.eval()  # Set model to evaluate mode

        running_loss = 0.0
        running_corrects = 0
        pred,true=np.array([]),np.array([])
        # 遍历数据
        for inputs, labels in dataloaders[phase]:
            inputs = inputs.to(device)
            labels = labels.to(device)

            # 零参数梯度
            optimizer.zero_grad()

            # 前向
            with torch.set_grad_enabled(phase == 'train'):
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)

                # 只在训练模式下进行反向和优化
                if phase == 'train':
                    loss.backward()
                    optimizer.step()

            # 统计
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
            pred_array=preds.cpu().numpy()
            label_array=labels.data.cpu().numpy()
            pred=np.concatenate([pred,pred_array])
            true=np.concatenate([true,label_array])


        epoch_loss = running_loss / dataset_sizes[phase]
        epoch_acc = (running_corrects.double() / dataset_sizes[phase]).item()

        # 记录每个epoch的loss和accuracy
        if phase == 'train':
            train_loss_history.append(epoch_loss)
            train_acc_history.append(epoch_acc)
        else:
            val_loss_history.append(epoch_loss)
            val_acc_history.append(epoch_acc)

        print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
        print("{} F1-Score:{:.4f}".format(phase,f1_score(true, pred,average='macro')))

        # 深拷贝模型
        if phase == 'val' and epoch_acc > best_acc:
            best_acc = epoch_acc
            best_model_wts = copy.deepcopy(model.state_dict())

    print()

print('Best val Acc: {:4f}'.format(best_acc))

# 加载最佳模型权重
# model.load_state_dict(best_model_wts)
# torch.save(model, 'shufflenet_best_model.pth')
# print("The trained model has been saved.")

epoch = range(1, len(train_loss_history) + 1)

fig, ax = plt.subplots(1, 2, figsize=(10, 4))
ax[0].plot(epoch, train_loss_history, label='Train loss')
ax[0].plot(epoch, val_loss_history, label='Validation loss')
ax[0].set_xlabel('Epochs')
ax[0].set_ylabel('Loss')
ax[0].legend()

ax[1].plot(epoch, train_acc_history, label='Train acc')
ax[1].plot(epoch, val_acc_history, label='Validation acc')
ax[1].set_xlabel('Epochs')
ax[1].set_ylabel('Accuracy')
ax[1].legend()

plt.savefig("loss-acc.pdf", dpi=300,format="pdf")





def plot_cm(labels, predictions):
    conf_numpy = confusion_matrix(labels, predictions,normalize='true')
    conf_df = pd.DataFrame(conf_numpy, index=class_names, columns=class_names)

    plt.figure(figsize=(8, 7))

    sns.heatmap(conf_df, annot=True, cmap="BuPu")

    plt.title('Confusion matrix', fontsize=15)
    plt.ylabel('Actual value', fontsize=14)
    plt.xlabel('Predictive value', fontsize=14)
    plt.savefig("confuse.jpg")


def evaluate_model(model, dataloader, device):
    model.eval()
    true_labels = []
    pred_labels = []
    # 遍历数据
    for inputs, labels in dataloader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        # 前向
        with torch.no_grad():
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

        true_labels.extend(labels.cpu().numpy())
        pred_labels.extend(preds.cpu().numpy())

    return true_labels, pred_labels


# 获取预测和真实标签
true_labels, pred_labels = evaluate_model(model, dataloaders['val'], device)

# 计算混淆矩阵
cm_val = confusion_matrix(true_labels, pred_labels)
a_val = cm_val[0, 0]
b_val = cm_val[0, 1]
c_val = cm_val[1, 0]
d_val = cm_val[1, 1]

# 计算各种性能指标
acc_val = (a_val + d_val) / (a_val + b_val + c_val + d_val)  # 准确率
error_rate_val = 1 - acc_val  # 错误率
sen_val = d_val / (d_val + c_val)  # 灵敏度
sep_val = a_val / (a_val + b_val)  # 特异度
precision_val = d_val / (b_val + d_val)  # 精确度
F1_val = (2 * precision_val * sen_val) / (precision_val + sen_val)  # F1值
MCC_val = (d_val * a_val - b_val * c_val) / (
    np.sqrt((d_val + b_val) * (d_val + c_val) * (a_val + b_val) * (a_val + c_val)))  # 马修斯相关系数

# 打印出性能指标
print("验证集的灵敏度为：", sen_val,
      "验证集的特异度为：", sep_val,
      "验证集的准确率为：", acc_val,
      "验证集的错误率为：", error_rate_val,
      "验证集的精确度为：", precision_val,
      "验证集的F1为：", F1_val,
      "验证集的MCC为：", MCC_val)

# 绘制混淆矩阵
plot_cm(true_labels, pred_labels)

# 获取预测和真实标签
train_true_labels, train_pred_labels = evaluate_model(model, dataloaders['train'], device)
# 计算混淆矩阵
cm_train = confusion_matrix(train_true_labels, train_pred_labels)
a_train = cm_train[0, 0]
b_train = cm_train[0, 1]
c_train = cm_train[1, 0]
d_train = cm_train[1, 1]
acc_train = (a_train + d_train) / (a_train + b_train + c_train + d_train)
error_rate_train = 1 - acc_train
sen_train = d_train / (d_train + c_train)
sep_train = a_train / (a_train + b_train)
precision_train = d_train / (b_train + d_train)
F1_train = (2 * precision_train * sen_train) / (precision_train + sen_train)
MCC_train = (d_train * a_train - b_train * c_train) / (
    math.sqrt((d_train + b_train) * (d_train + c_train) * (a_train + b_train) * (a_train + c_train)))
print("训练集的灵敏度为：", sen_train,
      "训练集的特异度为：", sep_train,
      "训练集的准确率为：", acc_train,
      "训练集的错误率为：", error_rate_train,
      "训练集的精确度为：", precision_train,
      "训练集的F1为：", F1_train,
      "训练集的MCC为：", MCC_train)

# 绘制混淆矩阵
plot_cm(train_true_labels, train_pred_labels)